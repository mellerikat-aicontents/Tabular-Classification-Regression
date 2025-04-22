# __init__.py로 패키지임을 linter에게 알려줌 & file의 경로 기준으로 상대경로 import
import sys
import shutil
import importlib
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime

CURRENT_PATH = str(Path(__file__).parent)
if CURRENT_PATH not in sys.path:
    sys.path.append(CURRENT_PATH)

from src_tcr.model_manager import Model_Manager
from src_tcr.multiprocessor import Multiprocessor
from src_tcr.utils_tcr import parse_hpo_settings, get_kwargs_for_model_selection, get_kwargs_for_best_model_train, init_tcr_config
from src_tcr.utils_tcr import get_eval_score, aggregate_and_sort_results, get_model_selection_json, concat_dataframes
from src_tcr.utils_tcr import get_kwargs_for_create_df_output

class TCR:
    '''
    Tabular Data Classificaion / Regression 진행
    HPO(model_selection) 진행 후, best_model 선정
    best_model로 전체 데이터에 대해 재학습

    Args:
        df_retrain_by_groupkey (object)     : 재학습 시 사용될 전체 dataset
        cv_data_list (object)               : Validation을 위해 나누어진 DataFrame list를 담고 있음
        config (dict)                       : TCR에서 사용하는 config dictionary
        workflow_type (str)                 : train / inference pipeline 구분자
        logger_method_dict (dict, optional) : print문을 처리할 logger, alolib logger를 사용 / Defaults to None

    '''

    def __init__(self, df_retrain_by_groupkey: object, cv_data_list: object, config: dict, workflow_type: str, logger_method_dict):
        '''df_retrain_by_groupkey :
               - groupkey X: single dataframe
               - groupkey O: 그룹키 명칭을 key로 그룹키에 해당하는 dataframe을 value로 하는 dictionary (ex. {groupA: df_A, groupB:, df_B, ...}) 
           cv_data_list: 모델 학습 및 HPO용 data_split list
               - 1. groupkey X & cross-val split: [[df_train1, df_val1], [df_train2, df_val2], ...]
               - 2. groupkey X & train_test split: [df_train, df_val]
               - 3. groupkey O & cross-val split: {groupA: [[df_A_train1, df_A_val1], [df_A_train2, df_A_val2]...]
                                                   groupB: [[df_B_train1, df_B_val1], [df_B_train2, df_B_val2]...], ...}
               - 4. groupkey O & train_test split: {groupA: [df_A_train, df_A_val], groupB: [df_B_train, df_B_val], ...}'''
        # DataFrame
        self.df_retrain_by_groupkey = df_retrain_by_groupkey # dataframe or dict
        self.cv_data_list = cv_data_list # cv --> nested_list, train_test --> list | groupkey 있을 경우 dict

        # Config
        self.config = config # asset_train/inference.py에서 입력받음 (config 준비)
        self.model_type = config['readiness']['task_type'] # classification / regression
        self.model_list = list(set(config[workflow_type]['model_list'])) # default: [rf, gb, lgbm, cb, xgb]
        self.hpo_settings = config[workflow_type]['hpo_settings']  # {} -> update_hpo_settings를 통해 갱신
        self.eval_metric = config[workflow_type]['evaluation_metric']
        self.target_label = config['readiness']['target_label']
        self.X_cols = config['readiness']['x_columns']
        self.y_col = config['readiness']['y_column']

        # Multiprocessing
        self.use_multiprocessing = config[workflow_type]['multiprocessing']
        self.num_cpu_core = config[workflow_type]['num_cpu_core']
        self.multiprocessor = Multiprocessor(self.num_cpu_core)

        self.data_split = self.config[workflow_type]['data_split']['method'] # 필요없어짐
        self.is_groupkey_used = isinstance(cv_data_list, dict) # 수정 필요할 수도
        self.encoding = 'utf-8' # config 추가될 시 수정할 것
        if self.model_type == 'classification':
            self.classification_type = config['readiness']['classification_type']
        else:
            self.classification_type = None

        # Model_Manager는 self.config를 그대로 가져감
        self.time_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.hpo_result = {}
        self.best_model_result = []

        # For Result Tracking & Test
        self.pipeline       = workflow_type
        self.tcr_config     = init_tcr_config(cv_data_list, config)

        # For logging
        self.alolib_logger = logger_method_dict
        self.logger = logger_method_dict
        self.model_manager = Model_Manager(self.config, workflow_type, self.logger)

        # Reset result
        model_path, externel_path = Path(config[workflow_type]['model_path']), Path(config[workflow_type]['external_path'])

        if logger_method_dict is None:
            if workflow_type == 'train':
                if model_path.exists():
                    self.save_warning("The saved model will be reset.\n")
                    shutil.rmtree(model_path)

                if externel_path.exists():
                    self.save_warning('warning', 'The experimental results will be reset.\n')
                    shutil.rmtree(externel_path)

            elif workflow_type == 'inference':
                if externel_path.exists():
                    self.save_warning('warning', 'The experimental results will be reset.\n')
                    shutil.rmtree(externel_path)

    # def logger(self, log_type, msg):
    #     '''
    #     info, warning, error 세 종류의 log를 출력

    #     Args:
    #         log_type (str)  : info, warning, error 세 가지 값을 가짐
    #         msg (str)       : 출력할 log message

    #     Raises:
    #         ValueError      : alolib logger를 사용하지 않고 tcr.py만 따로 쓸 때, ValueError를 raise함

    #     '''
    #     if self.alolib_logger is None:
    #         if log_type == 'error':
    #             print(f'{[log_type]}: {msg}')
    #             raise ValueError

    #         print(f'[{log_type}]: {msg}')

    #     else:
    #         #self.alolib_logger[log_type](msg)
    #         print(f'[{log_type}]: {msg}')

    
    def save_info(self, msg):
        self.logger.info(msg)

    def save_warning(self, msg):
        self.logger.warning(msg)

    def save_error(self, msg):
        self.logger.error(msg)

    # CLM 구분에서 '학습 config parsing'에 해당함
    def update_hpo_settings(self, hpo_setting_user):
        '''
        - 입력된 hpo_setting을 hyper parameter optimization을 진행할 수 있는 형태로 parsing 진행
        - Model_Manager에서 hpo_settings를 return받아 parsing (self.config를 Update)
        - parsing된 hpo_settings는 Model_Manager로 넘어가 각 setting 별로 training 진행
        - 'one_to_one'은 동일 길이 list별 1대1 mapping
        - 'all'은 Grid Search

        Args:
            hpo_setting_user (dict)     : User가 설정한 HPO 설정
                                          key:
                                            - model_name
                                          value:
                                            - HPO 진행할 parameter (list)
                                            - Fix할 parameter (str, float, int, boolean, ...)

        Returns:
            hpo_settings (dict)         : 모델 별 모든 실험에 대한 **kwargs의 형태로 update

        '''

        self.tcr_config[self.pipeline]['update_hpo_settings']['hpo_setting_user'] = hpo_setting_user # Update to tcr_config (코드 유지 보수)

        hpo_settings = {}
    
        self.save_info(f'The model {self.model_list} to be used for learning is loaded.')
        hpo_setting_default = self.model_manager.get_default_hpo_settings(self.model_list)
        self.save_info(f'{self.model_list} loading completed')
        self.save_info(f'The loading of {self.model_list} has been completed.')
        self.tcr_config[self.pipeline]['update_hpo_settings']['hpo_setting_default'] = hpo_setting_default # Update to tcr_config (코드 유지 보수)

        # User가 설정한 HPO 설정에 model_list에 넣어주지 않은 모델이 있을 경우 체크
        # alolib.logger의 ProcessLogger 사용할 것
        models_not_in_model_list = set(hpo_setting_user.keys()) - set(self.model_list)

        # hpo_settting_user에 default HPO settings update
        self.save_info('The default parameters for each model are being updated.')
        self.save_info('=' * 100 + '\n')
        for model in self.model_list:
            if model not in hpo_setting_user:
                hpo_setting_user[model] = hpo_setting_default[model]
                continue

            for param in hpo_setting_default[model]:
                if param in hpo_setting_user[model]:
                    continue

                hpo_setting_user[model][param] = hpo_setting_default[model][param]

        # self.hpo_settings에 모델 별 param_list parsing (list of dict)
        for model, model_params in hpo_setting_user.items():
            if model in models_not_in_model_list:
                # logger 추가
                continue

            hpo_settings[model] = parse_hpo_settings(model_params)

        # self.config update
        self.config[self.pipeline]['hpo_settings'] = hpo_settings
        self.tcr_config[self.pipeline]['update_hpo_settings']['hpo_settings'] = hpo_settings # Update to tcr_config (코드 유지 보수)
        return hpo_settings

    def model_selection(self, df_list_by_groupkey, use_multiprocessing=True):
        '''
        - model_manager 호출 및 HPO 실행
        - best model 선택

        Args:
            df_list_by_groupkey (list)              : Cross-validation data list
            use_multiprocessing (bool, optional)    : multiprocessing 사용 여부. Defaults to True.

        Returns:
            best_model_result (dict)    : key값으로 groupkey_col의 value값, value로는 (model_name, model_params) tuple을 가짐
                                          model_name (str)      : best_model에 쓰인 모델 이름 e.g.) rf, gbm, lgbm, ...
                                          model_params (dict)   : model의 입력으로 사용할 **kwargs
        '''

        self.tcr_config[self.pipeline]['model_selection']['df_list_by_groupkey'] = df_list_by_groupkey # Update to tcr_config (코드 유지 보수)

        # Multiprocessing에 필요한 args list 생성
        # groupkey_col으로 구분 된 df list (K-Fold의 각 split이 element), model_name, model_params, idx_df_valid를 담고 있음
        # idx_df_valid는 df_list에서 validation에 사용할 df의 index
        # 현재 input args: cv_data_list, model_list, hpo_settings, data_split
        kwargs_processes = get_kwargs_for_model_selection(df_list_by_groupkey, self.model_list, self.hpo_settings, self.data_split)
        # kwargs_processes[idx_list].keys(): dict_keys(['groupkey_val', 'df_list', 'model_name', 'model_params'])
        self.tcr_config[self.pipeline]['model_selection']['kwargs_processes'] = kwargs_processes # Update to tcr_config (코드 유지 보수)

        best_model_result = {}

        def job_train_and_eval(groupkey_val, df_list, model_name, model_params):
            '''
            Description:
                - pred결과에 실험 정보를 같이 저장하기 위한 Wrapper Function
                - pred값과 비교할 target label을 mapping하기 위해서는 실험 정보가 필요하기에 같이 return

            '''

            # 모델 class 객체를 불러오고 (hpo 세팅 반영) 학습 진행(fit)
            _, pred, _ = self.model_manager.model_train(df_list, model_name, model_params)
            if self.model_type == 'classification':
                pred = pred[:, -1] # TODO 추후 y_label encoding이 추가 될 시 변경할 것

            # df_list: [df_train, df_valid]
            true_label = df_list[1][self.y_col]
            eval_score = get_eval_score(self.model_type, self.classification_type, pred, true_label, self.target_label)
            return groupkey_val, model_name, model_params, eval_score

        # 각 실험 세팅에 대한 train & evaluate을 병렬적으로 수행
        # 하나의 sub process라도 error 발생 시 다른 process를 할당하지 않고 바로 멈춤
        if use_multiprocessing:
            self.save_info('The training is starting using multiprocessing.')
            self.multiprocessor.set_target_function(job_train_and_eval)
            exp_results = self.multiprocessor.run(kwargs_processes) #

        else:
            self.save_info('The training is starting sequentially.')
            exp_results = [job_train_and_eval(**kwargs) for kwargs in tqdm(kwargs_processes)]

        self.save_info('The training has been completed.')
        self.tcr_config[self.pipeline]['model_selection']['exp_results'] = exp_results # Update to tcr_config (코드 유지 보수)

        # regression일 경우 ascending order / classification일 경우 descending order로 sort
        # key: groupkey_val / value: list of dict key (model_name, model_params, mean_eval_score)
        self.save_info(f'Sorting the entire experiment results based on the {self.eval_metric}.\n')
        in_ascending_order = self.model_type == 'regression' and self.eval_metric != 'r2'
        exp_result_by_groupkey = aggregate_and_sort_results(exp_results, self.eval_metric, in_ascending_order)
        self.tcr_config[self.pipeline]['model_selection']['exp_result_by_groupkey'] = exp_result_by_groupkey # Update to tcr_config (코드 유지 보수)

        # 전체 HPO 결과 Attribute로 저장
        self.hpo_result = exp_result_by_groupkey

        # key: groupkey_val value: dict (model_name, model_params, eval_score)
        # only return best_model_result --> exp_result[0]

        self.save_info('The Hyper Parameter Optimization has been completed and the results are being output.')

        for groupkey_val, exp_result in exp_result_by_groupkey.items():
            best_model_result[groupkey_val] = exp_result[0]
            model_name, _, eval_score = exp_result[0].values()

            if self.is_groupkey_used:
                self.save_info(f'groupkey: {groupkey_val}')
            self.save_info(f'best model: {model_name}, {self.eval_metric}: {eval_score[self.eval_metric]:.4f}')

        self.save_info('=' * 100 + '\n')
        self.tcr_config[self.pipeline]['model_selection']['best_model_result'] = best_model_result # Update to tcr_config (코드 유지 보수)
        return best_model_result

    # TODO Sampling Asset 적용 시 --> val_data_list=None, is_sampling_applied=False (추후 고려할 것)
    def best_model_train(self, best_model_result, df_retrain_by_groupkey, use_multiprocessing=True):
        '''
        - best model parameter 세팅으로 전체 학습 데이터 재학습
        - model_selection.json 및 best_model instance 저장

        Args:
            best_model_result (dict)                : model_selection을 통해 반환 받은 최적 parameter dictionary
            df_retrain_by_groupkey (dict)           : groupkey 별 HPO 후 best model로 재학습에 사용할 전체 DataFrame (없을 시 key값 no_group_key)
            use_multiprocessing (bool, optional)    : Multiprocessing 사용 여부 (실제 run method에서 call될 때, groupkey 있는 경우에만 True)

        Returns:
            train_result_by_groupkey (dict)         : groupkey 별 전체 데이터 학습결과 (tcr_sampled 컬럼 none일 경우, 학습X 추론만 진행)
            best_model_by_groupkey (dict)           : groupkey 별 best model instance (ML Model)
        '''

        self.tcr_config[self.pipeline]['best_model_train']['best_model_result'] = best_model_result # Update to tcr_config (코드 유지 보수)

        # Sampling 적용 시 원본 df에 학습 결과 붙이는 것은..?
        # pd.merge 사용하려면 결국 id 컬럼이 붙어 있어야해 --> 없는 경우엔 만들자?
        # 추후 수정할 것
        kwargs_processes = get_kwargs_for_best_model_train(df_retrain_by_groupkey, best_model_result)
        self.tcr_config[self.pipeline]['best_model_train']['kwargs_processes'] = kwargs_processes # Update to tcr_config (코드 유지 보수)

        def job_best_model_train(groupkey_val, df_list, model_name, model_params):
            best_model, pred, indices = self.model_manager.model_train(df_list, model_name, model_params)
            col_names = best_model.get_col_names(self.y_col)
            df_pred = pd.DataFrame(pred, columns=col_names, index=indices)
            return groupkey_val, best_model, df_pred

        self.save_info('Starting retraining on the entire training data.')

        # Multiprocessing
        if use_multiprocessing:
            self.multiprocessor = Multiprocessor() # Reset Multiprocessor
            self.multiprocessor.set_target_function(job_best_model_train)
            best_model_results = self.multiprocessor.run(kwargs_processes)

        # Multiprocessing 사용 X
        else:
            best_model_results = []
            for kwargs in tqdm(kwargs_processes):
                best_model_results.append(job_best_model_train(**kwargs))

        self.save_info('The retraining has been completed.')
        self.save_info('=' * 100 + '\n')
        best_model_by_groupkey = {}
        train_result_by_groupkey = {}

        # groupkey 사용 X인 경우, groupkey
        for groupkey_val, best_model, df_pred in best_model_results:
            # 실제 Return dict로 변환
            best_model_by_groupkey[groupkey_val] = best_model
            train_result_by_groupkey[groupkey_val] = df_pred

        self.tcr_config[self.pipeline]['best_model_train']['best_model_by_groupkey'] = best_model_by_groupkey # Update to tcr_config (코드 유지 보수)
        self.tcr_config[self.pipeline]['best_model_train']['train_result_by_groupkey'] = train_result_by_groupkey # Update to tcr_config (코드 유지 보수)
        return train_result_by_groupkey, best_model_by_groupkey

    def best_model_save(self, best_model_by_groupkey, hpo_result, use_multiprocessing=True):
        '''
        - model manager의 model_save method를 사용하여 groupkey 별 best model을 저장
        - 전체 HPO 결과 --> model_selection.json에 저장

        Args:
            best_model_by_groupkey (dict)           : groupkey별 best model instance (ML Model)
            hpo_result (dict)                       : Hyperparameter Optimization Result Dict
            use_multiprocessing (bool, optional)    : Multiprocessing 사용 여부 (실제 run method에서 call될 때, groupkey 있는 경우에만 True)
        '''

        train_result_path = Path(self.config[self.pipeline]['model_path'])
        kwargs_processes = [{'groupkey': groupkey, 'best_model': best_model} for groupkey, best_model in best_model_by_groupkey.items()]

        def job_model_save(groupkey, best_model):
            model_selection = get_model_selection_json(groupkey, self.time_start, self.config, hpo_result[groupkey])
            model_path = train_result_path / groupkey if self.is_groupkey_used else train_result_path
            self.model_manager.model_save(best_model, model_path, model_selection, self.encoding)

        # Multiprocessing
        if use_multiprocessing:
            self.multiprocessor = Multiprocessor() # Reset Multiprocessor
            self.multiprocessor.set_target_function(job_model_save)
            best_model_results = self.multiprocessor.run(kwargs_processes)

        # Multiprocessing 사용 X
        else:
            best_model_results = []
            for kwargs in tqdm(kwargs_processes):
                best_model_results.append(job_model_save(**kwargs))

        self.save_info('The results of the Best Model have been saved.')
        self.save_info('=' * 100 + '\n')

    #TODO: shapley_sampling asset에서 검증을 따로 하는지
    def calc_shapley_value(self, best_model_by_groupkey, df_retrain_by_groupkey, external_path, save_plot=True, use_multiprocessing=True):
        '''
        - shapley value 값을 계산
        - plot 파일 저장 (Optional)

        Args:
            best_model_by_groupkey (dict)           : groupkey 별 best model instance (ML Model)
            df_retrain_by_groupkey (dict)           : groupkey 별 HPO 후 best model로 재학습에 사용된 전체 DataFrame (없을 시 key값 no_group_key)
            external_path (str)                     : plot 파일 저장 경로
            save_plot (bool, optional)              : plot 파일 저장 Flag
            use_multiprocessing (bool, optional)    : Multiprocessing 사용 여부 (실제 run method에서 call될 때, groupkey 있는 경우에만 True)

        Returns:
            df_shapley_by_groupkey (dict)   : groupkey 별 XAI(shapley) 결과 값 DataFrame (없을 시 key값 no_group_key)
        '''

        df_retrain_by_groupkey = {'no_group_key': df_retrain_by_groupkey} if not isinstance(df_retrain_by_groupkey, dict) else df_retrain_by_groupkey

        pipeline = self.pipeline
        self.tcr_config[pipeline]['calc_shapley_value']['best_model_by_groupkey'] = best_model_by_groupkey # Update to tcr_config (코드 유지 보수)

        external_path = Path(external_path)
        self.tcr_config[pipeline]['calc_shapley_value']['external_path'] = external_path # Update to tcr_config (코드 유지 보수)

        kwargs_processes = []
        keys = ('groupkey', 'best_model', 'df_retrain', 'path_plot', 'save_plot')

        for groupkey, best_model in best_model_by_groupkey.items():
            if self.is_groupkey_used:
                values = (groupkey, best_model, df_retrain_by_groupkey[groupkey], external_path / groupkey, save_plot)
            else:
                values = (groupkey, best_model, df_retrain_by_groupkey[groupkey], external_path, save_plot)
            kwargs_processes.append(dict(zip(keys, values)))

        self.tcr_config[pipeline]['calc_shapley_value']['kwargs_processes'] = kwargs_processes # Update to tcr_config (코드 유지 보수)

        def job_calc_shapley(groupkey, best_model, df_retrain, path_plot, save_plot):
            df_shapley = self.model_manager.calc_shap_value(best_model, df_retrain, path_plot, save_plot)
            return groupkey, df_shapley

        if use_multiprocessing:
            self.multiprocessor = Multiprocessor() # Reset Multiprocessor
            self.multiprocessor.set_target_function(job_calc_shapley)
            shap_results = self.multiprocessor.run(kwargs_processes)

        # Multiprocessing 사용 X
        else:
            shap_results = []
            for kwargs in tqdm(kwargs_processes):
                shap_results.append(job_calc_shapley(**kwargs))

        if save_plot:
            self.save_info(f'The summary plot for {self.pipeline.capitalize()} data has been saved.')

        self.save_info('=' * 100 + '\n')

        df_shap_by_groupkey = dict(shap_results)
        self.tcr_config[pipeline]['calc_shapley_value']['df_shap_by_groupkey'] = df_shap_by_groupkey # Update to tcr_config (코드 유지 보수)
        return df_shap_by_groupkey

    def best_model_inference(self, inference_data_list, use_multiprocessing=True):
        '''
        - model manager 호출 및 best_model load
        - inference 데이터에 대한 예측

        Args:
            inference_data_list (object)            : inference에 사용될 DataFrame (no groupkey) 혹은 DataFrame을 담은 dict (groupkey 별 DataFrame)
            use_multiprocessing (bool, optional)    : Multiprocessing 사용 여부 (실제 run method에서 call될 때, groupkey 있는 경우에만 True)

        Returns:
            inf_result_by_groupkey (dict)   : groupkey 별 inference 데이터에 대한 prediction 결과 값 (없을 시 key값 no_group_key)
            inf_model_by_groupkey (dict)    : groupkey 별 inference model (없을 시 key값 no_group_key)
        '''

        model_path = Path(self.config[self.pipeline]['model_path'])

        groupkey_vals = inference_data_list.keys() if isinstance(inference_data_list, dict) else ['no_group_key']
        self.tcr_config['inference']['best_model_inference']['groupkey_vals'] = groupkey_vals # Update to tcr_config (코드 유지 보수)

        def job_model_inference(groupkey_val, df_inf, model):
            # train과정에서 저장된 best model을 불러와서 inference수행
            model, pred, indices = self.model_manager.model_inference(df_inf, model)

            # y_col이 multi (list) 인 경우까지 포함해서 전부 df_inf에 존재할 시 eval_score 계산
            y_col_set = self.y_col if isinstance(self.y_col, list) else [self.y_col]

            if set(y_col_set).issubset(df_inf.columns):
                #eval_score = get_eval_score(model.model_type, pred[:, -1], df_inf[self.y_col], self.target_label) # multi일 시 -1에서 바뀌어야함
                eval_score = None
            else:
                eval_score = None

            col_names = model.get_col_names(self.y_col)
            df_pred = pd.DataFrame(pred, columns=col_names, index=indices)
            return groupkey_val, model, df_pred, eval_score

        kwargs_processes = []
        keys = ('groupkey_val', 'df_inf', 'model')

        # 지정된 경로에(model_path) 저장된 best model 불러오기
        for groupkey_val in groupkey_vals:
            if self.is_groupkey_used:
                model = self.model_manager.model_load(model_path / groupkey_val, self.encoding)
                values = (groupkey_val, inference_data_list[groupkey_val], model)
            else:
                model = self.model_manager.model_load(model_path, self.encoding)
                values = (groupkey_val, inference_data_list, model)

            kwargs_processes.append(dict(zip(keys, values)))

        self.tcr_config['inference']['best_model_inference']['kwargs_processes'] = kwargs_processes # Update to tcr_config (코드 유지 보수)

        if use_multiprocessing:
            self.multiprocessor = Multiprocessor()
            self.multiprocessor.set_target_function(job_model_inference)
            inference_results = self.multiprocessor.run(kwargs_processes)

        else:
            inference_results = []
            for kwargs in kwargs_processes:
                inference_results.append(job_model_inference(**kwargs))

        self.save_info('Inference completed')
        self.save_info('=' * 100 + '\n')
        inf_result_by_groupkey, inf_model_by_groupkey = {}, {}

        for groupkey_val, model, df_pred, eval_score in inference_results:
            inf_model_by_groupkey[groupkey_val] = model
            inf_result_by_groupkey[groupkey_val] = {
                'df_pred': df_pred,
                'eval_score': eval_score
            }

        self.tcr_config['inference']['best_model_inference']['inf_result_by_groupkey'] = inf_result_by_groupkey # Update to tcr_config (코드 유지 보수)
        self.tcr_config['inference']['best_model_inference']['inf_model_by_groupkey'] = inf_model_by_groupkey # Update to tcr_config (코드 유지 보수)
        return inf_result_by_groupkey, inf_model_by_groupkey

    def create_df_output(self, df_retrain_by_groupkey, df_shapley_by_groupkey, df_pred_by_groupkey, use_multiprocessing=True):
        """
        - output DataFrame 생성
        - 원본 DataFrame / Shapley Value DataFrame / Prediction DataFrame의 형태로 구성

        Args:
            df_retrain_by_groupkey (dict)           : groupkey 별 HPO 후 best model로 재학습에 사용된 전체 DataFrame (없을 시 key값 no_group_key)
            df_shapley_by_groupkey (dict)           : groupkey 별 XAI(shapley) 결과 값 DataFrame (없을 시 key값 no_group_key)
            df_pred_by_groupkey (dict)              : groupkey 별 best model의 prediction DataFrame (없을 시 key값 no_group_key)
            use_multiprocessing (bool, optional)    : Multiprocessing 사용 여부 (실제 run method에서 call될 때, groupkey 있는 경우에만 True)

        Returns:
            df_output_by_groupkey (dict)    : groupkey 별 output DataFrame (없을 시 key값 no_group_key)
        """

        df_retrain_by_groupkey = {'no_group_key': df_retrain_by_groupkey} if not isinstance(df_retrain_by_groupkey, dict) else df_retrain_by_groupkey

        self.tcr_config[self.pipeline]['create_df_output']['df_input_list_by_groupkey'] = df_retrain_by_groupkey
        self.tcr_config[self.pipeline]['create_df_output']['df_shapley_by_groupkey'] = df_shapley_by_groupkey
        self.tcr_config[self.pipeline]['create_df_output']['df_pred_by_groupkey'] = df_pred_by_groupkey

        df_output_by_groupkey = {}

        args = df_retrain_by_groupkey, df_shapley_by_groupkey, df_pred_by_groupkey, self.y_col
        kwargs_processes = get_kwargs_for_create_df_output(*args)
        self.tcr_config[self.pipeline]['create_df_output']['kwargs_processes'] = kwargs_processes

        def job_concat_dataframes(groupkey, df_retrain, df_shapley, df_pred, y_col):
            df_output = concat_dataframes(df_retrain, df_shapley, df_pred, y_col)
            return groupkey, df_output

        if use_multiprocessing:
            self.multiprocessor = Multiprocessor()
            self.multiprocessor.set_target_function(job_concat_dataframes)
            df_outputs = self.multiprocessor.run(kwargs_processes)

        else:
            df_outputs = []
            for kwargs in kwargs_processes:
                df_outputs.append(job_concat_dataframes(**kwargs))

        df_output_by_groupkey = dict(df_outputs)
        self.tcr_config[self.pipeline]['create_df_output']['df_output_by_groupkey'] = df_output_by_groupkey
        self.save_info('The output DataFrame has been successfully created.')
        self.save_info('=' * 100 + '\n')

        return df_output_by_groupkey

    def run(self):
        """
        - TCR 실행

        Returns:
            df_output_by_groupkey (dict)    : groupkey 별 output DataFrame (없을 시 key값 no_group_key)
        """

        if self.pipeline == 'train':
            # hpo setting parsing 진행
            self.save_info('=' * 100)
            self.save_info('Parsing the input Hyper Parameter setting values.')
            # {model명: [{hpo세팅1}, {hpo세팅2}, ...]} 형식
            # ex) {'lgbm': [{'max_depth': 5, 'n_estimators': 300, ...}, {'max_depth': 3, 'n_estimators': 300, ...}, ...], ...}
            self.hpo_settings = self.update_hpo_settings(
                hpo_setting_user            = self.hpo_settings
            )

            # HPO 실행 후 best_model 정보 return / multiprocessing은 무조건 실행
            self.save_info('=' * 100)
            self.save_info('Proceeding with Hyper Parameter Optimization.')
            best_model_result = self.model_selection(  # 'model_params'에 최적의 hyper parameter 세팅 저장
                df_list_by_groupkey         = self.cv_data_list,  # split 데이터
                use_multiprocessing         = self.use_multiprocessing
            )

            # best_model 전체 데이터로 재학습
            self.save_info('=' * 100)
            self.save_info('Re-training the model selected through HPO on the entire training data.')
            # df_pred_by_groupkey: best모델을 활용한 prediction 결과 (classification인 경우 probability값도 반환)
            #  best_model_by_groupkey: best모델 클래스 객체  ex) <classification_model.lgbm.TCR_model object at 0x7fd54fadca90>
            df_pred_by_groupkey, best_model_by_groupkey = self.best_model_train(
                best_model_result           = best_model_result,
                df_retrain_by_groupkey      = self.df_retrain_by_groupkey,  # retrain용 데이터
                use_multiprocessing         = self.is_groupkey_used and self.use_multiprocessing
            )

            # train 결과 및 best_model 저장
            # lgbm은 joblib dump시 multiprocessing Pool과 충돌
            is_lgbm_in_any_best_model = any(model.model_name == 'lgbm' for model in best_model_by_groupkey.values())
            use_multiprocessing = self.is_groupkey_used and (not is_lgbm_in_any_best_model) and self.use_multiprocessing
            self.save_info('=' * 100)
            self.save_info('Saving the selected best model.')
            self.best_model_save(
                best_model_by_groupkey      = best_model_by_groupkey,
                hpo_result                  = self.hpo_result,
                use_multiprocessing         = use_multiprocessing
            )

            # Shap Value 계산
            save_shap_plot = self.config[self.pipeline]['shapley_value']
            self.save_info('=' * 100)
            self.save_info('Conducting Explainable AI (XAI) using SHAP (SHapley Additive exPlanations) values.')
            df_shapley_by_groupkey = self.calc_shapley_value(
                best_model_by_groupkey      = best_model_by_groupkey,
                df_retrain_by_groupkey      = self.df_retrain_by_groupkey,
                external_path               = self.config[self.pipeline]['external_path'],
                save_plot                   = save_shap_plot,
                use_multiprocessing         = use_multiprocessing
            )

            # output dataframe 생성
            self.save_info('=' * 100)
            self.save_info('Creating the output DataFrame.')
            self.save_info('The structure is composed in the order of original data, SHAP results, Prediction, Label, and Data Split.')
            df_output_by_groupkey = self.create_df_output(
                df_retrain_by_groupkey   = self.df_retrain_by_groupkey,
                df_shapley_by_groupkey      = df_shapley_by_groupkey,
                df_pred_by_groupkey         = df_pred_by_groupkey,
                use_multiprocessing         = self.is_groupkey_used and self.use_multiprocessing
            )

            return df_output_by_groupkey

        elif self.pipeline == 'inference':
            # best_model 불러와서 inference 진행
            self.save_info('=' * 100)
            self.save_info('Start the prediction by loading the saved best model.')
            inf_result_by_groupkey, inf_model_by_groupkey = self.best_model_inference(
                inference_data_list         = self.df_retrain_by_groupkey,
                use_multiprocessing         = self.is_groupkey_used and self.use_multiprocessing
            )

            # inference result에서 결과 값 뽑아낸 후 shap value 계산
            self.save_info('=' * 100)
            self.save_info('Conducting XAI using SHAP values.')
            df_pred_by_groupkey = {groupkey: result['df_pred'] for groupkey, result in inf_result_by_groupkey.items()}
            is_lgbm_in_any_best_model = any(model.model_name == 'lgbm' for model in inf_model_by_groupkey.values())
            use_multiprocessing = self.is_groupkey_used and (not is_lgbm_in_any_best_model) and self.use_multiprocessing
            df_shapley_by_groupkey = self.calc_shapley_value(
                best_model_by_groupkey      = inf_model_by_groupkey,
                df_retrain_by_groupkey      = self.df_retrain_by_groupkey,
                external_path               = self.config[self.pipeline]['external_path'],
                save_plot                   = True,
                use_multiprocessing         = use_multiprocessing
            )

            # output.csv 파일 생성
            self.save_info('=' * 100)
            self.save_info('Creating the output DataFrame.')
            self.save_info('The structure is composed in the order of original data, SHAP results, Prediction, Label, and Data Split.')
            df_output_by_groupkey = self.create_df_output(
                df_retrain_by_groupkey   = self.df_retrain_by_groupkey,
                df_shapley_by_groupkey      = df_shapley_by_groupkey,
                df_pred_by_groupkey         = df_pred_by_groupkey,
                use_multiprocessing         = self.is_groupkey_used and self.use_multiprocessing
            )

            return df_output_by_groupkey
