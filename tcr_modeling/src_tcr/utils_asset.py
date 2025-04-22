import os
import json
import pandas as pd
import numpy as np
import copy
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import entropy


def convert_columns(config_in, input_df, workflow_type=None):
    '''칼럼들의 명칭 변환 
           x_columns: colname_A, colname_B, ... -> TCR_Input_X0, TCR_Input_X1, ...
           y_column: colname_1, ... -> TCR_Target_Y0'''
    #config = config_in.copy()
    config = copy.deepcopy(config_in)
    input_data = input_df.copy()

    # X 칼럼 변환
    x_columns = config['readiness']['x_columns'] 
    convert_dict = {}
    for idx, x_col in enumerate(x_columns):
        convert_dict[x_col] = 'TCR_Input_X' + str(idx)
    
    input_data.rename(columns=convert_dict, inplace=True)
 
    if workflow_type:
        config['readiness']['x_columns'] = [convert_dict[x_col]for x_col in x_columns]
        config[workflow_type]['column_mapping_dict'] = convert_dict  # 원래 x 칼럼 명: 변환된 칼럼 명

    # Y 칼럼 변환
    y_column = [config['readiness']['y_column']] 
    convert_dict_y = {}
    for idx, y_col in enumerate(y_column):
        convert_dict_y[y_col] = 'TCR_Target_Y' + str(idx)
        
    input_data.rename(columns=convert_dict_y, inplace=True)
    if workflow_type:
        config['readiness']['y_column'] = [convert_dict_y[y_col]for y_col in y_column][0]
        config[workflow_type]['column_mapping_dict'].update(convert_dict_y)  # 원래 x 칼럼 명: 변환된 칼럼 명
    
    return config, input_data


def make_grouped_df(df, config): 
        '''df -> {groupA: df_A, groupB: df_B, ...}'''
        group_keys = config['readiness']['original_groupkey_columns']
        groupkey_list = config['readiness']['groupkey_list']
        groupkey_column = config['readiness']['groupkey_columns'][0]

        df_dict = {}
        for groupkey in groupkey_list:
            partial_df = df[df[groupkey_column] == groupkey]
            df_dict[groupkey] = partial_df

        return df_dict


def all_in_one(config, output_data):
        '''groupkey가 있는 경우 groupkey 조합에 의해 분할된 데이터프레임 결과들을 하나로 통합
           CASE 1. groupkey O: {groupA: df_A, groupB: df_B, ...} -> single dataframe
           CASE 2. groupkey X: {'no_group_key': df} -> single dataframe'''
        # CASE 1. groupkey O
        if len(config['readiness']['groupkey_list']) > 0:
            for idx, group in enumerate(config['readiness']['groupkey_list']):
                data_group = output_data[group]
                if idx == 0:
                    res = data_group
                else:
                    res = pd.concat([res, data_group], axis=0)
            return res

        # CASE 2. groupkey X
        else:
            return output_data['no_group_key']


def decode_target_pred(config, output_data, workflow_type):
    '''label encoding에 의해 encoding된 y 칼럼과 prediction의 값들을 원상 복원
        output_data
            - groupkey X -> output_data['no_groupkey']에 데이터프레임이 담겨있는 형태
            - groupkey O -> output_data[group]에 해당 group에 속하는 데이터프레임이 담겨 있음 (group은 여러개 일 수 있음)
            - TCR_Input, TCR_Target 등 변환된 x, y칼럼 명칭 사용
        workflow_type: train/inference
        
        출력: TCR_Target_Y, pred_TCR_Target_Y의 encoding된 라벨 값들을 원본 값으로 변환됨
            * workflow_type=inference시에는 prediction칼럼의 값들만 복원(y칼럼 X)'''
    
    def decode_dataframe(output_df, col, mapping_dict, idxs=None):
        '''mapping_dict: {encodin된 y: 원본 y}'''
        def decode_value(encoded_y_value):
            '''encoding된 y값을 원본 y값(decoding된 값)으로 변환'''
            decoded_y_value = mapping_dict[encoded_y_value]
            return decoded_y_value
        
        if idxs:
            output_df_copy = output_df.copy()
            values =  output_df_copy.loc[idxs, col].apply(decode_value)
            output_df_copy.loc[idxs, col] = values
            output_df = output_df_copy
        else:
            output_df[col] = output_df[col].apply(decode_value)
        
        return output_df
        
        
    # CASE 1. groupkey X
    if len(config['readiness']['original_groupkey_columns']) == 0:
        if workflow_type == 'train':
            # 1.1 target 칼럼 값 복원
            target_col = config['readiness']['y_column']  # TCR_Target_Y 형식
            output_data = decode_dataframe(output_data, target_col, config['preprocess']['y_mapping_table'])
    
        # 1.2 pred 칼럼 값 복원
        pred_col = [col for col in list(output_data.columns) if 'TCR-pred_' in col][0]  # pred_TCR_Target_Y 형식  
        output_data = decode_dataframe(output_data, pred_col, config['preprocess']['y_mapping_table'])

        return output_data
    
    
    # CASE 2. groupkey O
    else:
        groupkey_columns = config['readiness']['groupkey_columns'][0]
        groupkey_list = config['readiness']['groupkey_list']
        for idx, group in enumerate(groupkey_list):
            output_data_group = output_data[output_data[groupkey_columns] == group]
            row_idxs = list(output_data[output_data[groupkey_columns] == group].index)
    
            if workflow_type == 'train':
                # !! A value is trying to be set on a copy of a slice from a DataFrame. 에러를 방지하기위해 row_idxs설정
                # 2.1 target 칼럼 값 복원
                target_col = config['readiness']['y_column']  # TCR_Target_Y 형식  
                output_data_group = decode_dataframe(output_data_group, target_col, config['preprocess'][group]['y_mapping_table'], row_idxs)

            # 2.1 pred 칼럼 값 복원
            pred_col = [col for col in list(output_data_group.columns) if 'TCR-pred_' in col][0]  # pred_TCR_Target_Y 형식  
            output_data_group = decode_dataframe(output_data_group, pred_col, config['preprocess'][group]['y_mapping_table'], row_idxs)
    
            if idx == 0:
                outputs_total = output_data_group
            else:
                outputs_total = pd.concat([outputs_total, output_data_group], axis=0)

        return outputs_total


def rename_columns(config, output_data, workflow_type):
    '''칼럼들의 명칭 변환 (산출물 output.csv에 대한 유저 편의성 개선)
        1. 공통 (rename_commons)
            - TCR_Input_X0, TCR_Input_X1식으로 변환된 x_columns의 명칭들을 복원
            - TCR_Target_Y 형식으로 변환된 원본 y_column의 명칭 복원 (workflow_type=inference시에는 동작 X)
            - pred_TCR_Target_Y 형식으로 변환된 칼럼을 pred_{원본 y_column의 명칭}로 복원
        
        2. shapley=True인 경우 (rename_shap함수)
            - TCR-shap_TCR_Input_X0, shap_TCR_Input_X1, ... -> TCR-shap_{원래 x_column 명칭}로 복원
        
        3. classification 태스크인 경우 (rename_prob함수)
            - TCR-prob_TCR_Input_X0, prob_TCR_Input_X1, ... -> TCR-prob_{원래 x_column 명칭}로 복원'''
    
    def rename_commons(config, output, workflow_type):
        mapping_dict = config[workflow_type]['column_mapping_dict']
        mapping_dict = {v:k for k, v in mapping_dict.items()}  # 변환된 칼럼 명(TCR_~): 원래 칼럼 명
        mapping_dict.update({f'TCR-pred_{k}': f'TCR-pred_{v}' for k, v in mapping_dict.items() if 'TCR_Target' in k})
        
        output_data.rename(columns=mapping_dict, inplace=True)
        return output_data
    
    def rename_shap(config, output, workflow_type):
        mapping_dict = config[workflow_type]['column_mapping_dict']
        mapping_dict = {v:k for k, v in mapping_dict.items()}
        mapping_dict = convert_col(output_data, mapping_dict, name='TCR-shap')  
        
        output_data.rename(columns=mapping_dict, inplace=True)   
        return output_data
       
    def rename_prob(config, output, mapping_dict, workflow_type):
        mapping_dict = convert_col(output_data, mapping_dict, name='TCR-prob')  
        output_data.rename(columns=mapping_dict, inplace=True)     
        return output_data

    def convert_col(df, mapping_dict, name):
        '''TCR-shap_, TCR-prob_ 칼럼 변환용'''
        columns = [i for i in list(df.columns) if name in i]

        encoded_value = [i.split('_')[-1] for i in columns]
        if name == 'TCR-shap':
            encoded_value = [i.split('TCR-shap_')[1:][0] for i in columns]

        for idx, col in enumerate(encoded_value):
            try: encoded_value[idx] = int(col)
            except Exception: continue
            
        convert_col = [f'{name}_' + str(mapping_dict[val]) for val in encoded_value]
        convert_dict = {k: v for k, v in zip(columns, convert_col)}
        return convert_dict

    # 1. 공통
    output_data = rename_commons(config, output_data, workflow_type)
        
    # 2. shapley=True인 경우
    if config[workflow_type]['shapley_value']:
        output_data = rename_shap(config, output_data, workflow_type)
    
    # 3. classification 태스크인 경우
    if config[workflow_type]['model_type'] == 'classification':
        # CASE 1. groupkey X
        if len(config['readiness']['original_groupkey_columns']) == 0:
            mapping_dict = config['preprocess']['y_mapping_table']
            output_data = rename_prob(config, output_data, mapping_dict, workflow_type)
        
        # CASE 2. groupkey O
        else:
            # ! readiness의 check_y_group에 의해 특정 group에 속하는 데이터의 y칼럼 unique값이
            # 전체 데이터의 y칼럼 unique값과 다르면 groupkey 리스트에서 제거되기 때문에 아래와 같이 동작 가능
            group = list(config['preprocess'].keys())[0]
            mapping_dict = config['preprocess'][group]['y_mapping_table']
            output_data = rename_prob(config, output_data, mapping_dict, workflow_type)
            
    return output_data


def save_output(config, output, workflow_type):
    '''output 저장
        - default: 유저 편의성을 위해 pred 칼럼과 y 칼럼의 명칭에서 prep_제거
        - optional
            - 1. original index 정보 (원본 train/inference 데이터의 index정보)
            - 2. output_type: all(모든 칼럼 저장) / simple(예측과 관련된 칼럼 + target 칼럼 저장 + groupkey칼럼 저장)'''

    cols = list(output.columns)
    target = config['readiness']['original_y_column']
        
    cols = {col: col.replace('prep_', '') for col in cols if 'prep_' in col and target in col}
    output.rename(columns=cols, inplace=True)
    output = output.loc[:, ~output.columns.duplicated(keep='last')]
    
    # option 1. original index정보 추가 (output.csv 맨 앞 열에)
    output.insert(loc=0, column='origin_index', value=output.index)

    # option 2. 저장 옵션
    if config[workflow_type]['output_type'] == 'simple':
        if config['readiness']['original_groupkey_columns'] and workflow_type == 'train':  # groupkey 있으면 편의성을 위해 groupkey 정보도 저장
            groupkey_col = config['readiness']['groupkey_columns'][0]
            keywords = ['TCR-shap', 'TCR-pred', 'TCR-prob', 'origin_index', target, groupkey_col]
        else:  # groupkey X
            keywords = ['TCR-shap', 'TCR-pred', 'TCR-prob', 'origin_index', target]

        simple_cols = [col for col in output.columns if any(keyword in col for keyword in keywords)]              
        output = output[simple_cols]
    output.sort_values(['origin_index'], inplace=True) # origin index 기준으로 오름차순 sort
    
    return output


def make_summary_classification(config, output):

    num_classes = config['readiness']['num_classes'] 
    target_col = config['readiness']['original_y_column']

    prob_cols = [col for col in output.columns if 'TCR-prob' in col]
    uncertainty = output[prob_cols].T.apply(lambda x: entropy(x, base=num_classes))

    if len(output) == 1:
        result      = output['TCR-pred_'+target_col][0]
        probability = {col[len('TCR-prob_'):]: float(output[col][0]) for col in prob_cols}
    else: 
        if num_classes == 2:
            result = 'Binary-Classification'
        else:
            result = 'Multi-Classification'
        probability = {}

    summary_dict= {}
    summary_dict['result'] = result
    summary_dict['score'] = 1-float(np.mean(uncertainty))
    summary_dict['note'] = 'Confidence Score (The closer to 1, the better the predictive performance of the model)'
    summary_dict['probability'] = probability

    return summary_dict


def make_summary_regression(config, output_data, workflow_type):
    from sklearn.metrics import r2_score
    '''regression용 summary dictionary 만들기'''
    summary_dict = {}

    target_label_list = [config['readiness']['y_column']]

    if len(config['readiness']['groupkey_list']) > 0:
        key = config['readiness']['groupkey_list'][0]
        mapping_dict = config[workflow_type]['column_mapping_dict']
        mapping_dict = {v:k for k, v in mapping_dict.items()}
        target_label_list = [mapping_dict[i] for i in target_label_list]
    else:
        mapping_dict = config[workflow_type]['column_mapping_dict']
        mapping_dict = {v:k for k, v in mapping_dict.items()}
        target_label_list = [mapping_dict[i] for i in target_label_list]

    target_name = target_label_list[0]
    target_name = target_name.replace('prep_', '')
    pred_col = [col for col in list(output_data.columns) if target_name in col and 'TCR-pred_' in col][0]
    target_col =  [col for col in list(output_data.columns) if target_name in col and not 'TCR-pred' in col][0]

    y_pred = output_data[pred_col]
    y_target = output_data[target_col]

    # 회귀 결정계수 R^2계산 (이론상 0~1사이의 값) - 참고: https://ltlkodae.tistory.com/19
    R2 = r2_score(y_target, y_pred)
    if R2 < 0:
        score = 0
        note = 'WARNING!! R2<0: R2 had returned minus value. Reconsider your model or data'
    elif R2 == 1:
        score = 1
        note = 'WARNING!! R2=1: The model may be overfitted to the training data. Reconsider your data'
    else:
        score = round(float(R2), 4)
        note = f'The independent variables(x_columns) you used and the model account for about {score}% of the data'

    summary_dict['result'] = 'regression'
    summary_dict['score'] = score
    summary_dict['note'] = note
    summary_dict['probability'] = {}

    return summary_dict


def convert_columns_split(config, data_split):
    '''칼럼들의 명칭 변환 (train_test/KFold로 split된 데이터용)
           x_columns: colname_A, colname_B, ... -> TCR_Input_X0, TCR_Input_X1, ...
           y_column: colname_1, ... -> TCR_Target_Y0'''
    data_split_method = config['sampling']['data_split']['method']

    is_groupkey = len(config['readiness']['groupkey_list']) > 0
    if is_groupkey:
        groupkey_list = config['readiness']['groupkey_list']
        group_col = config['readiness']['groupkey_columns'][0]

    def process_cross_validation(data_set):
        for cv_idx in range(len(data_set)):
            data = data_set[cv_idx]  # [df_train_i, df_val_i]
            data_train = data[0] ; data_val = data[1]

            _, data_train = convert_columns(config, data_train, None) 
            _, data_val = convert_columns(config, data_val, None) 
            data_set[cv_idx][0] = data_train ; data_set[cv_idx][1] = data_val
        return data_set

    def process_train_test(data_set):
        data_train = data_set[0] ; data_val = data_set[1]

        _, data_train = convert_columns(config, data_train, None) 
        _, data_val = convert_columns(config, data_val, None) 
        data_set[0] = data_train ; data_set[1] = data_val
        return data_set

    # CASE 1.1 groupkey X & cross-val 
    if data_split_method == 'cross_validation' and not is_groupkey:
        data_split = process_cross_validation(data_split)

    # CASE 1.2 groupkey X & train-test split
    elif data_split_method == 'train_test' and not is_groupkey:
        data_split = process_train_test(data_split)

    # CASE 2.1 groupkey O & cross-val 
    elif data_split_method == 'cross_validation' and is_groupkey:
        for group in groupkey_list:
            data_group = data_split[group]
            data_group_converted = process_cross_validation(data_group)

            data_split[group] = data_group_converted

    # CASE 2.2 groupkey O & train-test split
    elif data_split_method == 'train_test' and is_groupkey:
        for group in groupkey_list:
            data_group = data_split[group]
            data_group_converted = process_train_test(data_group)

            data_split[group] = data_group_converted

    return data_split


def save_eval_result(config, output, workflow_type):
    ''' 평가 결과 저장 (asset_inference에서는 테스트 데이터에 타깃 칼럼 있을 때만 실행)
       - classification: Accuracy, F1, Recall, Precision 저장
       - regression: mae, mase, rmse, r2 저장'''
    
    col_name_dict = {v: k for k, v in config[workflow_type]['column_mapping_dict'].items()}
    target_name_key = [k for k in col_name_dict.keys() if 'TCR_Target_Y' in k][0]
    target_name = col_name_dict[target_name_key].split('prep_')[-1]  # 타깃 칼럼 명 (readiness옵션의 y_column 설정과 동일)
    pred_name = [col for col in output.columns if 'TCR-pred_' in col][0]  # 예측 칼럼 명

    y_target = output[target_name]  # 예측값
    y_pred = output[pred_name]  # GT

    if config['readiness']['task_type'] == 'classification':    
        # labels = list(config['preprocess']['y_mapping_table'].values())  
        labels = list(y_pred.unique())  
        report = classification_report(y_target, y_pred, labels=labels, output_dict=True)

        summary = pd.DataFrame()
        for idx, label in enumerate(labels):
            res = pd.DataFrame()
            label = str(label)
            report_label = report[label]
            data = {    
                        'label': label,
                        'Accuracy': accuracy_score(y_target, y_pred),
                        'F1-Score': report_label['f1-score'],
                        'Recall': report_label['recall'],
                        'Precision': report_label['precision']
                }
            res = pd.DataFrame(data, index=[idx])
            summary = pd.concat([summary, res], axis=0)

    elif config['readiness']['task_type'] == 'regression':
        mae = mean_absolute_error(y_target, y_pred)
        mse = mean_squared_error(y_target, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_target, y_pred)
        
        summary = pd.DataFrame(
            {
                'MAE': [mae],
                'MSE': [mse],
                'RMSE': [rmse],
                'R2': [r2]
            }
        )

    return summary


def update_train_config(config, args, DEFAULT_ARGS, pipeline):
    '''config에 args를 참고하여 train 옵션 세팅하기 / args에 없으면 default 옵션을 세팅'''
    workflow_type = pipeline['name']
    config[workflow_type] = copy.deepcopy(DEFAULT_ARGS)
    for arg_name, user_value in args.items():
        if user_value:  # experimental_yaml에서 빈 list나 ''등을 넣으면 default값이 세팅됨
            config[workflow_type][arg_name] = user_value

    config[workflow_type]['model_type'] = config['readiness']['task_type']  # classification / regression
    config[workflow_type]['data_split'] = config['sampling']['data_split']  # data_split 옵션

    # 평가 지표
    if config[workflow_type]['model_type'] == 'classification' and config[workflow_type]['evaluation_metric'] == 'auto':
        config[workflow_type]['evaluation_metric'] = 'accuracy'
    elif config[workflow_type]['model_type'] == 'regression' and config[workflow_type]['evaluation_metric'] == 'auto':
        config[workflow_type]['evaluation_metric'] = 'mse'

    config[workflow_type]['model_path'] = pipeline['model']['workspace']
    config[workflow_type]['external_path'] = pipeline['extra_output']['workspace']

    return config


def update_inference_config(config, pipeline):
    '''config[inference]에 asset_train에 사용된 옵션 세팅하기
        (inference argument에 유저가 입력해도 반영 안됨, train argument만 이용)'''
    # asset_train에서 사용된 config 불러오기
    with open(os.path.join(pipeline['model']['workspace'] + '/train_config.json'), 'r') as file:
        config[pipeline['name']] = json.load(file)

    # if args:
    #     for arg_name, user_value in args.items():
    #         config[workflow_type][arg_name] = user_value

    config[pipeline['name']]['model_path'] = pipeline['model']['workspace']
    config[pipeline['name']]['external_path'] = pipeline['extra_output']['workspace']

    return config
    
    
