import os
import logging
import json
import pickle


# ------------ 각 step에서 사용되는 default setting 불러오기 ------------
with open("./DEFAULT_SETTINGS.json", 'r') as file:
    DEFAULT_SETTINGS = json.load(file)


def input(pipeline: dict):
    from input.input import concat_dataframe, _read_csv, _read_parquet, read_file, filter_files

    logging.basicConfig(level=logging.DEBUG)  # DEBUG 레벨 이상의 메시지를 출력하도록 설정
    # logger = logging.getLogger(__name__)
    logger = logging.getLogger('TCR-ALO-v3')
    logger.debug("input")

    # GLOBAL VARIABLE
    CSV_FORMATS = ('csv', 'CSV')
    PARQUET_FORMATS = ("parquet", "parq", "pqt")
    PARQUET_ENGINE = "fastparquet"

    # 우선 pyarrow는 너무 무거우므로 fastparquet만 지원
    # 참고: https://blog.metafor.kr/237
    DEFAULT_CONFIG = DEFAULT_SETTINGS["input"]


    ext2func = {'csv': _read_csv, 'parquet': _read_parquet}
    ext2format = {'csv': CSV_FORMATS, 'parquet': PARQUET_FORMATS}  # 해당 부분에 확장자 확장 가능
    input_paths = os.listdir(pipeline['dataset']['workspace'])

    # 사용자 argument setting
    try:
        file_type = pipeline['input']['argument']['file_type'] 
        encoding = pipeline['input']['argument']['encoding']
    except:
        file_type = DEFAULT_CONFIG['file_type'] 
        encoding = DEFAULT_CONFIG['encoding']

    file_list = filter_files(input_paths, ext2format)
    df_list = [read_file(ext2func, os.path.join(pipeline['dataset']['workspace'], file_path), file_type, encoding, logger) for file_path in file_list]
    
    output_df = concat_dataframe(logger, df_list, input_paths)  # 하나의 데이터프레임으로 생성
    output_df.reset_index(drop=True, inplace=True)
    return {
        'dataframe': output_df  
    }


def readiness(pipeline: dict):
    from readiness.readiness import readiness_check, setting_configs

    logger = pipeline['logger']
    logger.debug("readiness")

    DEFAULT_ARGS = DEFAULT_SETTINGS["readiness"]

    input_data = pipeline['input']['result']['dataframe']  # output from 'input' step
    
    config = {}
    config = setting_configs(config=config, 
                             args=pipeline['readiness']['argument'], 
                             input_data=input_data, 
                             pipeline=pipeline, 
                             logger=logger,  default_args= DEFAULT_ARGS)

    rdc = readiness_check(logger)

    if pipeline['name'] == 'train':
        config, data = rdc.train(config, input_data)
        pipeline['model']['train_config'] = config['readiness']  # save model
    elif pipeline['name'] == 'inference':
        config, data = rdc.inference(config, input_data)

    return {
        'config': config,
        'dataframe': data
    }


def preprocess(pipeline: dict):
    from preprocess.preprocess import Preprocessor

    logger = pipeline['logger']
    logger.debug("preprocess")

    DEFAULT_ARGS = DEFAULT_SETTINGS["preprocess"]
    CONFIG_FILE_NAME = 'train_config.pkl'

    if pipeline['name'] == 'train':
        workflow_type = 'train_pipeline'
    else:
        workflow_type = 'inference_pipeline'

    pp = Preprocessor(workflow_type=workflow_type, 
                      model_path=pipeline['model'], 
                      default_args=DEFAULT_ARGS, 
                      user_args=pipeline['preprocess']['argument'],
                      config_file_name=CONFIG_FILE_NAME,
                      logger_method_dict=logger)
    
    data = pipeline['readiness']['result']['dataframe']
    config = pipeline['readiness']['result']['config']

    config, input_data = pp.prepare_config_data(config, data)
    config, input_data = pp.tcr_preprocess(config, input_data)

    if pipeline['name'] == 'train':  # inference시에 저장된 train-config를 그대로 사용
        save_path = pipeline['model'] + '/' + CONFIG_FILE_NAME
        with open(save_path, 'wb') as file:  # 'wb' 모드는 바이너리 쓰기 모드
            pickle.dump(config, file)

    return {
        'config': config,
        'dataframe': input_data
    }


def sampling(pipeline: dict):
    from sampling.sampling import get_output
    
    logger = pipeline['logger']
    logger.debug("preprocess")

    config = pipeline['preprocess']['result']['config']
    
    output_from_input = pipeline['preprocess']['result']['dataframe']
    config['sampling'] = pipeline['sampling']['argument']

    df, data_split = get_output(config, output_from_input, logger)

    res = dict()
    res['dataframe'] = df
    if data_split != None: 
        res['data_split'] = data_split

    return {
        'config': config,
        'dataframe': df,
        'data_split': data_split
    }


def train(pipeline: dict):
    # train step 산출물은 ~/.workspace/tcr/history/~/train 에서 확인
    from tcr_modeling.src_tcr.utils_asset import update_train_config, convert_columns_split, convert_columns, make_grouped_df, all_in_one, decode_target_pred, rename_columns, save_output, make_summary_classification, make_summary_regression, save_eval_result
    from tcr_modeling.tcr import TCR

    logger = pipeline['logger']
    logger.debug("train")

    DEFAULT_ARGS = DEFAULT_SETTINGS["train"]

    config = pipeline['sampling']['result']['config']

    input_data = pipeline['sampling']['result']['dataframe']
    data_split = None

    # split 데이터 (모델 학습 및 hpo용)
    if 'data_split' in config['sampling'].keys():
        data_split = pipeline['sampling']['result']['data_split']

    config = update_train_config(config, pipeline['train']['argument'], DEFAULT_ARGS, pipeline)

    # ---------------------------------- 1. data 준비 ----------------------------------
    # x_columns, y_column에 지정된 칼럼들의 명칭 변환 (한글/영문 상관없이 TCR_Input_x0, TCR_Input_x1, ... TCR_Input_Y 형식으로)
    # 일부 모델에 한글 칼럼명이 입력되는 경우 생기는 에러 방지용
    if data_split is not None:
        data_split = convert_columns_split(config, data_split)
    config, input_data = convert_columns(config, input_data, pipeline['name'])

    # ---------------------------------- 2. 모델 학습 및 HPO 진행 ----------------------------------
    is_groupkey = len(config['readiness']['groupkey_list']) > 0  # groupkey 옵션 사용 여부
    if is_groupkey:
        # df -> {groupA: df_A, groupB:, df_B, ...}
        input_data = make_grouped_df(input_data, config)

    tcr = TCR(df_retrain_by_groupkey=input_data,  # retrain용 (best모델 이용해서)
              cv_data_list=data_split,  # Hyper-Parameter Optimization용 (HPO)
              config=config, workflow_type=pipeline['name'], logger_method_dict=logger)

    # tcr의 train 함수 실행 결과 출력물
    # groupkey X: {'no_group_key': df}
    # groupkey O: {groupA: df_A, groupB: df_B, ...}
    output_df_list = tcr.run()

    # ---------------------------------- 3. 후처리 진행 ----------------------------------
    # 3.1 output_df_list를 하나의 데이터프레임으로 통합
    output = all_in_one(config, output_df_list)

    # 3.2 enoding된 y칼럼과 pred_칼럼의 값들을 원래 값으로 복원 (classification일 때)
    if config['readiness']['task_type'] == 'classification':
        output = decode_target_pred(config, output, pipeline['name'])

    # 3.3 1의 convert_columns를 통해 변환된 칼럼들의 명칭을 원상 복원
    output = rename_columns(config, output, pipeline['name'])

    # ---------------------------------- 4. train_config 저장 ----------------------------------
    train_config = config['train']  # 이후 asset_inference 단계에서 사용함
    save_file = 'train_config.json'
    train_model_path = pipeline['model']['workspace']
    with open(os.path.join(train_model_path , save_file), 'w') as f:
        json.dump(train_config, f, indent=4)

    # ---------------------------------- 5. output 저장 ----------------------------------
    output_path = os.path.join(pipeline['artifact']['workspace'], 'output.csv')
    output = save_output(config, output, pipeline['name'])
    output.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'{output_path}에 train 결과를 저장합니다.')

    # ---------------------------------- 6. train_summary 저장 ----------------------------------
    if config[pipeline['name']]['model_type'] == 'classification':  # classification용
        summary = make_summary_classification(config, output)
        logger.debug('Summary(classification) :', summary)
        result = {
            'summary' :{
                'result': summary['result'],
                'score': summary['score'],
                'note': summary['note'],
                'probability': summary['probability']
            }
        }

    elif config[pipeline['name']]['model_type'] == 'regression':  # regression용
        summary = make_summary_regression(config, output, pipeline['name'])
        logger.debug('Summary(Regression) :', summary)
        result = {
            'summary' :{
                'result': summary['result'],
                'score': summary['score'],
                'note': summary['note'],
                'probability': summary['probability']
            }
        }

    # ---------------------------------- 7. 평가 결과 저장 ----------------------------------
    train_external_path = pipeline['artifact']['workspace'][:-6]
    if len(config['readiness']['groupkey_list']) > 0:  # groupkey O
            groupkey_col = config['readiness']['groupkey_columns'][0]  
            for group in config['readiness']['groupkey_list']:
                output_group = output[output[groupkey_col] == group]
                eval_result = save_eval_result(config, output_group, pipeline['name'])
                eval_result.to_csv(train_external_path + f'eval_result_{group}.csv', index=False)

    else:  # groupkey X
        eval_result = save_eval_result(config, output, pipeline['name'])
        eval_result.to_csv(train_external_path + 'eval_result.csv', index=False)

    ## summary dict 다시 확인
    output = {
        'dataframe' : output,
        'data_split' : None
    }
    output.update(result)
    return output


def inference(pipeline: dict):
    from tcr_modeling.tcr import TCR
    from tcr_modeling.src_tcr.utils_asset import update_inference_config, convert_columns, make_grouped_df, all_in_one, decode_target_pred, rename_columns, save_output, make_summary_classification, make_summary_regression, save_eval_result

    logger = pipeline['logger']
    logger.debug("train")

    config = pipeline['sampling']['result']['config']
    config = update_inference_config(config, pipeline)  # train step에서 사용된 train_config 그대로 이용

    input_data = pipeline['sampling']['result']['dataframe']

    # ---------------------------------- 1. data_manager run ----------------------------------
    # 2.1 x_columns 칼럼명 변환 (한글/영문 상관없이 input_x0, input_x1... 형식으로)
    config, input_data = convert_columns(config, input_data, pipeline['name'])

    is_groupkey = len(config['readiness']['groupkey_list']) > 0
    if is_groupkey:
        input_data = make_grouped_df(input_data, config)

    # ---------------------------------- 2. TCR run ----------------------------------
    tcr = TCR(df_retrain_by_groupkey=input_data, cv_data_list=input_data, config=config, workflow_type=pipeline['name'], logger_method_dict=logger)
    # groupkey O : {group_A: dfA, group_B: dfB, ...} 형식
    # groupkey X: {no_group_key: df}
    output_df_list = tcr.run()  # tcr의 inference 함수 실행

    # ---------------------------------- 3. 후처리 진행 ----------------------------------
    # 3.1 output_df_list를 하나의 데이터프레임으로 통합
    output = all_in_one(config, output_df_list)

    # 3.2 enoding된 y칼럼과 pred_칼럼의 값들을 원래 값으로 복원 (classification일 때)
    if config['readiness']['task_type'] == 'classification':
        output = decode_target_pred(config, output, pipeline['name'])

    # 3.3 칼럼들의 명칭 복원    
    output = rename_columns(config, output, pipeline['name'])

    # ---------------------------------- 4. output 저장 ----------------------------------
    output = save_output(config, output, pipeline['name'])
    inference_output_path = os.path.join(pipeline['artifact']['workspace'], 'output.csv')
    output.to_csv(inference_output_path, index=False, encoding='utf-8-sig')
    logger.debug(f'{inference_output_path}에 inference 결과를 저장합니다.')

    # ---------------------------------- 5. inference_summary 저장 ----------------------------------
    if config[pipeline['name']]['model_type'] == 'classification':
        summary = make_summary_classification(config, output)
        logger.debug('Summary(classification) :', summary)
        result = {
            'summary' :{
                'result': summary['result'],
                'score': summary['score'],
                'note': summary['note'],
                'probability': summary['probability']
            }
        }
    
    elif config[pipeline['name']]['model_type'] == 'regression':  # regression용
        summary = make_summary_regression(config, output)
        logger.debug('Summary(classification) :', summary)
        result = {
            'summary' :{
                'result': summary['result'],
                'score': summary['score'],
                'note': summary['note'],
                'probability': summary['probability']
            }
        }
        
    # ---------------------------------- 7. 평가 결과 저장 ----------------------------------
    # !!! test 데이터에 타깃 칼럼이 존재할 때만 저장됨 !!!
    col_name_dict = {v: k for k, v in config[pipeline['name']]['column_mapping_dict'].items()}
    target_name_key = [k for k in col_name_dict.keys() if 'TCR_Target_Y' in k][0]
    target_name = col_name_dict[target_name_key]  # 타깃 칼럼 명 (readiness옵션의 y_column 설정과 동일)

    if target_name in output.columns:
        inference_external_path = inference_output_path[:-10]
        if len(config['readiness']['groupkey_list']) > 0:  # groupkey O
            groupkey_col = config['readiness']['groupkey_columns'][0]  
            for group in config['readiness']['groupkey_list']:
                output_group = output[output[groupkey_col] == group]
                eval_result = save_eval_result(config, output_group, pipeline['name'])
                eval_result.to_csv(inference_external_path + f'eval_result_{group}.csv', index=False)

        else:  # groupkey X
            eval_result = save_eval_result(config, output, pipeline['name'])
            eval_result.to_csv(inference_external_path + 'eval_result.csv', index=False)
    
    result = {
            'summary' :{
                'result': summary['result'],
                'score': summary['score'],
                'note': summary['note'],
                'probability': summary['probability']
            }
        }

    return result
