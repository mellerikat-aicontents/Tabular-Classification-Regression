from collections import defaultdict
import itertools
import psutil

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve, mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

def parse_hpo_settings(model_hpo_setting: dict) -> list:
    '''
    - 모델 별 hpo_setting parsing 진행

    Args:
        model_hpo_setting (dict)    : 각 model 별 hpo setting (e.g. hpo_settings['rf'])

    Returns:
        parsed_hpo_setting (list)   : model_params를 담은 list (model_params는 모델의 **kwargs)
    '''

    # HPO 진행 할 parameter, 고정 할 paramter와 HPO 방법 (one_to_one, grid serach) 구분
    # Type이 list일 경우 HPO 진행 / list가 아닐 경우 fixed parameter / 방법은 key: 'tcr_param_mix'로 구분
    # tcr_param_mix가 주어지지 않은 경우 --> raise error? one_to_one으로 진행? (현 code는 one_to_one으로 진행)
    hpo_params = {param: param_list for param, param_list in model_hpo_setting.items() if (isinstance(param_list, list) and len(param_list) > 1)}
    fixed_params = {param: value for param, value in model_hpo_setting.items() if not isinstance(value, list) and param != 'tcr_param_mix'}
    fixed_params.update({param: param_list[0] for param, param_list in model_hpo_setting.items() if (isinstance(param_list, list) and len(param_list) == 1)})
    tcr_param_mix = model_hpo_setting.get('tcr_param_mix', 'one_to_one')

    # one_to_one은 zip을 사용 / all(grid-search)은 itertools.product 사용
    # 이외의 값을 넣는 경우 --> raise error?
    if hpo_params:
        if tcr_param_mix == 'one_to_one':
            hpo_combinations = list(zip(*hpo_params.values()))
        elif tcr_param_mix == 'all':
            hpo_combinations = list(itertools.product(*hpo_params.values()))

    else:
        hpo_combinations = [fixed_params]

    # 모델 별 HPO case에 대한 parameter list return
    # dict unpacking(**)을 사용하여 fixed_params, hpo_params로 만든 dict merge
    # **로 2개의 dict를 unzip후 하나의 dict로 합침
    parsed_hpo_setting = [{**fixed_params, **dict(zip(hpo_params.keys(), hpo_comb))} for hpo_comb in hpo_combinations]

    return parsed_hpo_setting

def make_df_from_list(df_list, index_valid_fold):
    '''
    Description:
        - df_list를 input으로 받아 validation할 fold를 df_train, df_infer로 Split함

    Args:
        df_list (list)          : K-Fold로 Split된 df를 담고 있는 list
        index_valid_fold (int)  : Validation에 사용할 df의 index

    Return:
        df_train (Dataframe)    : Training Set
        df_valid (Dataframe)    : Validation Set

    '''

    # best_model_train (best model 전체 데이터 재학습인 case)
    if index_valid_fold is None:
        df_train = pd.concat(df_list)
        df_valid = None

    # 원본 list는 다른 실험 세팅에서 사용되므로 modify 불가
    # cv_data_list에서 K-Fold에 대한 정보를 list로 담아서 주는데, pd.concat은 original df copy
    # indexing방식은 view를 return할 것으로 예상했으나 선택된 columns의 type이 모두 동일해야함
    # train asset에서의 X_feature 컬럼은 float 혹은 int type임 (numeric)
    # └ 만약 전부 float type으로 들어온다면 --> pd.concat이 아닌 indexing으로 새로운 memory 할당 없이 가능!
    else:
        df_train = df_list.copy()
        df_train.pop(index_valid_fold)
        df_train = pd.concat(df_train)
        df_valid = df_list[index_valid_fold]

    return df_train, df_valid

def get_eval_score(model_type, classification_type, pred, true_label, target_label):
    '''
    - Classification / Regression에 해당하는 evaluation score dict를 return
    - Classification: f1, precision, recall, accuracy 지원
    - Regression: MSE, MAE, R-squared 지원

    Args:
        model_type (str)            : Classification / Regression
        classification_type (str)   : Classification일 때, binary / multi
        pred (object)               : model의 예측 값 numpy Array
        true_label (object)         : 실제 타겟 Label pandas Series
        target_label (object)       : evaluation score를 측정할 label (해당 label에 대한 결과 값 return)

    Returns:
        eval_score (dict)       : evaluation 결과 dict e.g.) {f1: 0.5, recall: 0.5, precision: 0.5, accuracy: 0.5}, ...
    '''
    eval_score = {}

    if model_type == 'classification':
        kwargs = {
            'binary': {'y_true':true_label, 'y_pred':pred, 'average': 'binary', 'pos_label': target_label[0]}, #binary인데 len(target_label)>2인 경우 없음 
            'multi': {'y_true':true_label, 'y_pred':pred, 'average':'micro', 'labels':target_label}
        }
        eval_score['f1'] = f1_score(**(kwargs[classification_type]))
        eval_score['recall'] = recall_score(**(kwargs[classification_type]))
        eval_score['precision'] = precision_score(**(kwargs[classification_type]))
        eval_score['accuracy'] = accuracy_score(true_label, pred)

    elif model_type == 'regression':
        eval_score['r2'] = r2_score(true_label, pred)
        eval_score['mse'] = mean_squared_error(true_label, pred)
        eval_score['mae'] = mean_absolute_error(true_label, pred)

    return eval_score

def aggregate_and_sort_results(exp_results, eval_metric, in_ascending_order):
    '''
    - HPO 수행
    - Cross validation 결과 계산 (K개 실험 evaluation score 평균)
    - groupkey_val로 split된 df별 evaluation score 정렬

    Args:
        exp_results (list)              : 전체 실험 결과
        eval_metric (str)               : Sorting에 사용할 evalustion
        in_ascending_order (bool)       : FLAG (sort in ascending order)
                                          Classification의 경우 점수가 높을수록 (score)
                                          Regression의 경우 오차가 작을수록 (loss)

    Returns:
        exp_result_by_groupkey (dict)   : HPO 결과
                                          key   - groupkey_val
                                          value - model_name, model_params, mean_eval_score (K-Fold 평균값)
    '''
    def create_model_priority(user_models):
        # 기본 우선순위 정의
        base_priority = {
            'rf': 0,
            'lgbm': 1,
            'gbm': 2,
            'xgb': 3,
            'cb': 4
        }
        
        # 사용자 모델의 우선순위 설정
        priority = {}
        next_priority = 1  # 사용자 모델의 우선순위는 1부터 시작

        # 사용자 모델 우선순위 설정
        for model in user_models:
            priority[model] = next_priority
            next_priority += 1

        # 기본 모델의 우선순위 추가
        for model in base_priority:
            if model not in priority:
                priority[model] = base_priority[model] + next_priority
        return priority
    
    def create_param_score(param_lists):
        # max_depth, n_estimators가 있을 경우 해당 값을 곱하여 score생성(오름차순 정렬 위함. )
        # 해당 param없으면 사용자 정의 모델로 판단하고 임의값 1을 넣음
        cnt = 1
        if ('max_depth' in param_lists.keys()) and ('n_estimators' in param_lists.keys()):
            cnt = cnt*param_lists['max_depth']*param_lists['n_estimators']
        return cnt
    
    aggregated_results = defaultdict(list)
    exp_result_by_groupkey = defaultdict(list)

    # Cross Validation 진행 시 groupekey_val, model_name, model_params 별로 K개의 실험 진행
    # 실험 세팅 별로 K개의 결과 담은 리스트 mapping ()
    # eval_score는 지원하는 evaluation method 전체의 정보를 담은 dict
    for groupkey_val, model_name, model_params, eval_scores in exp_results:
        key = (groupkey_val, model_name, tuple(sorted(model_params.items())), create_param_score(model_params))
        aggregated_results[key].append(eval_scores)  # Append the entire dictionary
    
    # 실험 세팅 별 evaluation 값 평균내기
    mean_scores = []
    for key, scores in aggregated_results.items():
        # scores: list[eval_score_1 (dict), eval_score_2 (dict), ..., eval_score_K (dict)]
        mean_score = {metric: np.mean([eval_score[metric] for eval_score in scores]) for metric in scores[0]}
        mean_scores.append((key[0], key[1], key[2], mean_score, key[3]))

    classification_metric = ['accuracy', 'f1', 'recall', 'precision']
    regression_metric = ['r2', 'mse', 'mae', 'rmse']
    
    user_model = sorted(set([x[1] for x in mean_scores]) - set(['rf','lgbm','gbm','xgb','cb']))
    model_priority = create_model_priority(user_model)
    
    if eval_metric in classification_metric:
        classification_metric.remove(eval_metric)
        other_metric = [m for m in classification_metric if m != eval_metric]
    elif eval_metric in regression_metric:
        other_metric = [m for m in regression_metric if m != eval_metric]
        
    mean_scores.sort(key=lambda x: (x[0], x[3][eval_metric], x[3][other_metric[0]], x[3][other_metric[1]], x[3][other_metric[2]], x[4], model_priority[x[1]]) 
                     if in_ascending_order else (x[0], -x[3][eval_metric], -x[3][other_metric[0]], -x[3][other_metric[1]], -x[3][other_metric[2]], x[4], model_priority[x[1]]))
    # mean_scores.sort(key=lambda x: (x[0], x[3][eval_metric], x[3][other_metric[0]], x[3][other_metric[1]], x[3][other_metric[2]], x[2]) if in_ascending_order else (x[0], -x[3][eval_metric], -x[3][other_metric[0]], -x[3][other_metric[1]], -x[3][other_metric[2]], x[2]))

    keys = ('model_name', 'model_params', 'evaluation_score')
    for groupkey_val, model_name, model_params, mean_eval_scores, _ in mean_scores:
        values = (model_name, dict(model_params), mean_eval_scores)
        exp_result_by_groupkey[groupkey_val].append(dict(zip(keys, values)))

    return exp_result_by_groupkey

def get_kwargs_for_model_selection(data_lists_by_groupkey, model_list, hpo_settings, data_split):
    '''
    - tcr.py의 model_selection에서 사용될 target function kwargs list return

    Args:
        data_lists_by_groupkey (dict)   : groupkey 별 df_train, df_valid를 담은 dict (없을 시 key 값 no_group_key)
        model_list (list)               : 학습에 사용할 model name list e.g.) ['cb', 'lgbm', ...]
        hpo_settings (dict)             : 학습에 사용할 모델 별 parameter dict
        data_split (str)                : Validation 방법론 (train_test / cross_validation)

    Returns:
        kwargs_processed (list)         : tcr.py의 model_selection에서 사용될 target function kwargs list
    '''

    keys = ('groupkey_val', 'df_list', 'model_name', 'model_params')
    kwargs_processed = []

    # groupkey_col을 설정하지 않은 경우 key: groupkey_val 'no_group_key'로 설정
    # groupkey_col을 설정 & 값이 'no_group_key'인 경우 --> tcr에서 type을 통해 확인 가능
    data_lists_by_groupkey = data_lists_by_groupkey if isinstance(data_lists_by_groupkey, dict) else {'no_group_key': data_lists_by_groupkey}

    # ({groupkey_val: df_list}, model, model_params, idx_valid_split)
    for groupkey_val, df_lists in data_lists_by_groupkey.items():
        for model in model_list:
            for model_params in hpo_settings[model]:
                # data_split이 train_test인 경우, [df_train, df_test] --> idx_valid_split: 1 로 고정
                if data_split == 'train_test':
                    df_train, df_test = df_lists[0], df_lists[0]
                    df_train = df_train[df_train['tcr_sampled'].isin(['sampled', 'over'])] if 'tcr_sampled' in df_train.columns else df_train
                    args = (groupkey_val, [df_train, df_test], model, model_params)
                    kwargs = dict(zip(keys, args))
                    kwargs_processed.append(kwargs)
                    continue

                # CV K-Fold인 경우 {groupkey_val: [df_0, df_1, df_2, ..., df_K], ...}
                for df_list in df_lists:
                    df_train, df_valid = df_list[0], df_list[1]
                    df_train = df_train[df_train['tcr_sampled'].isin(['sampled', 'over'])] if 'tcr_sampled' in df_train.columns else df_train
                    args = (groupkey_val, [df_train, df_valid], model, model_params)
                    kwargs = dict(zip(keys, args))
                    kwargs_processed.append(kwargs)

    return kwargs_processed

def get_kwargs_for_best_model_train(df_retrain_by_groupkey, best_model_result):
    """
    - tcr.py의 best_model_train에서 사용될 target function kwargs list return

    Args:
        df_retrain_by_groupkey (dict)   : groupkey 별 전체 학습 DataFrame (없을 시 key값 no_group_key)
        best_model_result (dict)        : groupkey 별 best_model 결과값 (model name과 parameter 필요 / 없을 시 key값 no_group_key)

    Returns:
        kwargs_processed (list)         : tcr.py의 best_model_train에서 사용될 target function kwargs list
    """

    df_retrain_by_groupkey = {'no_group_key': df_retrain_by_groupkey} if not isinstance(df_retrain_by_groupkey, dict) else df_retrain_by_groupkey
    keys = ('groupkey_val', 'df_list', 'model_name', 'model_params')
    kwargs_processes = []

    # args = tuple(data_list, model_name, model_params, idx_df_valid, is_retrain_model)
    for groupkey_val, result in best_model_result.items():
        df_retrain = df_retrain_by_groupkey[groupkey_val]
        df_train = df_retrain[df_retrain['tcr_sampled'].isin(['sampled', 'over'])] if 'tcr_sampled' in df_retrain.columns else df_retrain
        df_list = [df_train, df_retrain]
        model_name, model_params = result['model_name'], result['model_params']
        values = (groupkey_val, df_list, model_name, model_params)
        kwargs_processes.append(dict(zip(keys, values)))

    return kwargs_processes

def get_kwargs_for_create_df_output(df_retrain_by_groupkey, df_shapley_by_groupkey, df_pred_by_groupkey, y_col):
    """
    - tcr.py의 create_df_output에서 사용될 target function kwargs list return

    Args:
        df_retrain_by_groupkey (dict)   : groupkey 별 전체 학습 DataFrame (없을 시 key값 no_group_key)
        df_shapley_by_groupkey (dict)   : groupkey 별 shapley value DataFrame (없을 시 key값 no_group_key)
        df_pred_by_groupkey (dict)      : groupkey 별 예측값 DataFrame (없을 시 key값 no_group_key)
        y_col (str)                     : label 컬럼명

    Returns:
        kwargs_processed (list)         : tcr.py의 create_df_output에서 사용될 target function kwargs list
    """

    df_retrain_by_groupkey = df_retrain_by_groupkey if isinstance(df_retrain_by_groupkey, dict) else {'no_group_key': df_retrain_by_groupkey}
    kwargs_processes = []
    keywords = ['groupkey', 'df_retrain', 'df_shapley', 'df_pred', 'y_col']

    for groupkey in df_retrain_by_groupkey:
        params = [groupkey, df_retrain_by_groupkey[groupkey], df_shapley_by_groupkey[groupkey], df_pred_by_groupkey[groupkey], y_col]
        kwargs_processes.append(dict(zip(keywords, params)))

    return kwargs_processes

def concat_dataframes(df_retrain, df_shapley, df_pred, y_col):
    """
    - output DataFrame concat [df_retrain, df_shapley, df_pred]
    - 컬럼 순서 일부 수정

    Args:
        df_retrain (object) : 원본 DataFrame
        df_shapley (object) : Shapley Value DataFrame
        df_pred (object)    : 예측값 DataFrame
        y_col (str)         : label 컬럼명

    Returns:
        df_output (object)  : output DataFrame
    """

    cols_input = [col for col in df_retrain.columns if col not in [y_col]]
    cols_y_pred = list(df_pred.columns)

    if df_shapley is None:
        if y_col in df_retrain.columns:
            cols_reindexed = cols_input + cols_y_pred + [y_col]
        else:
            cols_reindexed = cols_input + cols_y_pred

        df_output = pd.concat((df_retrain, df_pred), axis=1)
        df_output = df_output[cols_reindexed]

    else:
        cols_shapley = list(df_shapley.columns)
        if y_col in df_retrain.columns:
            cols_reindexed = cols_input + cols_shapley + cols_y_pred + [y_col]
        else:
            cols_reindexed = cols_input + cols_shapley + cols_y_pred

        df_output = pd.concat((df_retrain, df_shapley, df_pred), axis=1)
        df_output = df_output[cols_reindexed]

    return df_output

def get_model_selection_json(groupkey_val, start_time, config, hpo_result):
    """
    - Hyper Parameter Optimization 결과값 JSON dump 용 dict 생성

    Args:
        groupkey_val (str)  : groupkey 값
        start_time (str)    : tcr.py init 시각
        config (dict)       : tcr 설정 값 dict
        hpo_result (dict)   : Hyper Parameter Optimization 결과 dict

    Returns:
        model_selection (dict)  : Hyper Parameter Optimization 결과값 JSON dump 용 dict
    """

    # Initialize the base structure of the model selection dictionary
    model_name, model_params, eval_score = hpo_result[0].values()
    best_model_info = {'model_name': model_name, 'model_type': config['readiness']['task_type'], 'model_params': model_params}

    model_selection = {
        'groupkey': groupkey_val,
        'best_model_info': best_model_info,
        'task_type': config['readiness']['task_type'],
        'data_split': config['train']['data_split'],
        'evaluation_metric': config['train']['evaluation_metric'],
        'target_label': config['readiness']['target_label'],
        'time': start_time,
        'x_columns': [col_prep for col_prep, col_TCR in config['train']['column_mapping_dict'].items() if 'TCR_Input_X' in str(col_TCR)],
        'model_score': {}
    }

    modeL_name_count = {}

    for result in hpo_result:
        model_name, model_params, eval_score = result.values()

        if model_name in modeL_name_count:
            modeL_name_count[model_name] = modeL_name_count[model_name] + 1
        else:
            modeL_name_count[model_name] = 0

        model_set_key = f'{model_name}_set{modeL_name_count[model_name]}'

        model_info = {
            'parameters': model_params,
            'score': eval_score
        }
        model_selection['model_score'][model_set_key] = model_info

    return model_selection

def get_trained_model_name(groupkey_vals, path, is_groupkey_used):

    model_name_dict = {}

    if is_groupkey_used:
        for groupkey_val in groupkey_vals:
            model_path = path / groupkey_val

            for item in model_path.glob('*best_model*'):
                if item.is_file():
                    model_name = item.name.split('_')[-1].split('.')[0]
                    model_name_dict[groupkey_val] = model_name

        return model_name_dict

    for item in path.glob('*best_model*'):
        if item.is_file():
            model_name = item.name.split('_')[-1].split('.')[0]
            model_name_dict['no_group_key'] = model_name

    return model_name_dict

def init_tcr_config(cv_data_list, config):
    """
    - tcr 테스트용 config init
    """
    tcr_config = {
        'train': {
            'update_hpo_settings': {
                'hpo_setting_user': None,
                'hpo_setting_default': None,
                'models_not_in_config': None,
                'hpo_settings': None
            },
            'model_selection': {
                'df_list_by_groupkey': None,
                'kwargs_processes': None,
                'exp_results': None,
                'exp_result_by_groupkey': None,
                'best_model_result': None
            },
            'best_model_train': {
                'best_model_result': None,
                'kwargs_processes': None,
                'best_model_by_groupkey': None,
                'train_result_by_groupkey': None
            },
            'best_model_save': {
                'model_selection': None
            },
        },
        'inference': {
            'best_model_inference': {
                'groupkey_vals': None,
                'kwargs_processes': None,
                'inf_result_by_groupkey': None,
                'inf_model_by_groupkey': None
            }
        }
    }

    for pipeline in tcr_config:
        tcr_config[pipeline]['calc_shapley_value'] = {
            'best_model_by_groupkey': None,
            'external_path': None,
            'kwargs_processes': None,
            'df_shap_by_groupkey': None,
        }
        tcr_config[pipeline]['create_df_output'] = {
            'df_input_list_by_groupkey': None,
            'df_shapley_by_groupkey': None,
            'df_pred_by_groupkey': None,
            'kwargs_processes': None,
            'df_output_by_groupkey': None
        }

    return tcr_config


def save_model_selection(json_data):
    '''HPO 결과가 저장된 model_selection.json을 csv 형식으로 알아보기 변환 (asset_train 전용)'''
    set_evaluation_metric =json_data["evaluation_metric"]
    model_df = pd.json_normalize(json_data["model_score"])

    trans_model = model_df.transpose().reset_index()
    model_index =  trans_model['index'].str.split('.')
    trans_model['model'] = model_index.str.get(0)
    trans_model['category'] = model_index.str.get(1)

    trans_model_parameter = trans_model[trans_model['category']=='parameters'].copy()
    trans_model_parameter['parameters'] = model_index.str.get(2)
    trans_model_parameter_copy = trans_model_parameter.copy()
    trans_model_parameter = trans_model_parameter_copy.rename(columns={0:'value'})

    summary_parameter= trans_model_parameter[['model','parameters','value']].copy()
    summary_parameter = pd.crosstab(index=summary_parameter.model, columns=summary_parameter.parameters, values=summary_parameter.value, aggfunc='sum')
    summary_parameter.reset_index(inplace=True)

    trans_model_score = trans_model[trans_model['category']=='score'].copy()
    trans_model_score['evaluation_metric'] = model_index.str.get(2)
    trans_model_score_copy = trans_model_score.copy()
    trans_model_score = trans_model_score_copy.rename(columns={0:'score'})

    summary_score = trans_model_score[['model','evaluation_metric','score']]
    summary_score = pd.crosstab(index=summary_score.model, columns=summary_score.evaluation_metric, values=summary_score.score, aggfunc='sum')
    summary_score.reset_index(inplace=True)

    total_summary = pd.merge(summary_score, summary_parameter, on='model',how='inner')
    total_summary = total_summary.fillna('-')

    if set_evaluation_metric in ['accuracy', 'r2', 'f1', 'recall', 'precision']:
        ascending = False
    else:
        ascending = True
    total_summary = total_summary.sort_values(by=set_evaluation_metric, ascending = ascending)

    eval_cols = [col for col in list(total_summary.columns) if col in ['model', 'accuracy', 'r2', 'f1', 'recall', 'precision']]
    common_cols = [col for col in list(total_summary.columns) if '-' not in  total_summary[col].unique() and col not in eval_cols]
    others = list(set(total_summary.columns) - set(eval_cols) - set(common_cols))

    total_summary = total_summary[eval_cols + common_cols + others]

    return total_summary
