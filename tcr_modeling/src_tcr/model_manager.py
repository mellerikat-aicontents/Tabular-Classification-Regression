import sys
import json
import joblib

from importlib import import_module
from pathlib import Path
from collections import defaultdict

import shap
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager


def set_hangul():
    """
    - matplotlib에 한글 font 추가
    """
    flist = font_manager.findSystemFonts()

    for v in flist:
        try:
            fprop = font_manager.FontProperties(fname=v)
            fname = fprop.get_name()
            ffile = fprop.get_file()
            if 'gothic' in fname.lower() or 'nanumsquare' in fname.lower():
                mpl.rcParams['font.family'] = fname
                plt.rc('axes', unicode_minus=False) # 마이너스 출력되도록
                print('font:', mpl.rcParams['font.family']) 
                return
        except:
            continue
set_hangul()

# ALO에서 AI Contents를 실행 시, alo 폴더에서 실행됨 (동적 import를 위해서는)
# AI Contents가 위치한 폴더의 경우, alo/assets/train/asset_{name}.py
# __init__.py를 통해서 Linter에게 패키지임을 알려줘도 importlib 사용을 위해 추가
CURRENT_PATH = str(Path(__file__).parent)
if CURRENT_PATH not in sys.path:
    sys.path.append(CURRENT_PATH)

from utils_tcr import make_df_from_list
from utils_tcr import save_model_selection

class Wrapper:
    '''
    - Classification / Regression 모두 공통적으로 사용되는 Model Wrapper
    - {model}.py의 Super Class에 해당
    - {model}.py은 필요할 경우, override

    Args:
        model_name (str)    : 모델명 e.g.) cb, lgbm, xgb, ...
        model_type (str)    : Task Type (Classification / Regression)
        model (object)      : ML Model Instance
    '''

    def __init__(self, model_name, model_type, model):
        self.model_name = model_name
        self.model_type = model_type
        self.model = model
        self.trained = False

    def fit(self, train_data, label):
        '''
        Model 학습 (필요할 경우 각 모델 파일에서 override)

        Args:
            train_data (object) : 학습에 사용될 DataFrame
            label (object)      : 학습에 사용될 Label 컬럼

        '''
        self.model.fit(train_data, label)
        self.trained = True

    def predict(self, inference_data):
        '''
        Model 예측 (필요할 경우 각 모델 파일에서 override)

        Args:
            inference_data (object) : Inference에 사용될 DataFrame

        Returns:
            y_pred (object)         : Model의 예측 값 DataFrame
        '''
        y_pred = self.model.predict(inference_data)
        return y_pred

    def load(self, path):
        '''
        Model 로드 (필요할 경우 각 모델 파일에서 override)

        Args:
            path (object)   : 로드할 폴더 경로
        '''
        self.model = joblib.load(str(path))

    def save(self, path):
        '''
        Model 저장 (필요할 경우 각 모델 파일에서 override)

        Args:
            path (object)   : 저장할 폴더 경로
        '''
        if not self.trained:
            raise ValueError('Tried to save untrained model. You need to fit your model first')

        if not path.exists():
            path.mkdir(parents=True)

        model_path = path / 'best_model.pkl'
        joblib.dump(self.model, model_path)

    # Should go to utils_tcr.py (only getting prefix)
    def get_col_names(self, y_cols):
        """
        - 모델 예측 DataFrame의 컬럼에 prefix 붙이기

        Args:
            y_cols (object)  : Label명

        Returns:
            (list)  : prefix 붙은 Label 컬럼명
        """
        if isinstance(y_cols, str):
            return [f'TCR-pred_{y_cols}']
        
        if hasattr(y_cols, "__iter__"):
            return [f'TCR-pred_{y}' for y in y_cols]

        else:
            raise ValueError("y_cols should be string or iterable. y_cols:{y_cols}")

class Model_Manager:
    '''
    - Model을 관리하는 Manager
    - model instance 생성, default param 세팅
    - model train, model inference
    - model save, model load

    Args:
        config (dict)       : Train / Inference에 필요한 config dict
        workflow_type (str) : Pipeline (Train / Infernce)
        logger (object)     : info, warning, error를 logging 하는 object (alolib logger 혹은 단순 print문)
    '''

    def __init__(self, config, workflow_type, logger):
        self.config = config
        self.model_type = config['readiness']['task_type'] # classification / regression
        self.model_name_list = config[workflow_type]['model_list'] # [rf, gb, lgbm, cb, xgb]
        self.hpo_settings = config[workflow_type]['hpo_settings']
        self.eval_metric = config[workflow_type]['evaluation_metric']
        self.X_features = config['readiness']['x_columns']
        self.y_cols = config['readiness']['y_column']
        self.y_encoding = 'label' # 추후 y가 one-hot encoding 등의 방법을 사용하는 경우에 변경

        # get_default_hpo_settings에서 update
        self.model_modules = {}
        self.hpo_settings = defaultdict(dict)

        # only for classification
        self.classes = None
        self.workflow_type = workflow_type

        # logging method
        self.logger = logger

        # columns_mapping_dict
        self.column_mapping_dict = config[workflow_type]['column_mapping_dict']

    # CLM에서 create models에 해당
    def get_default_hpo_settings(self, model_name_list: list) -> dict:
        '''
        - Model List에 작성된 모델 load
        - Model 파일에 기재된 default parameter load

        Args:
            model_name_list (list)  : 학습 시 사용 될 모델명 list e.g.) ['rf','gb','lgbm','cb','xgb']

        Returns:
            hpo_settings (dict)     : config parsing을 위한 model HPO 정보를 담은 dict
        '''
        
        # 모델 class객체 불러오기
        # import {model}.py ()
        for model_name in model_name_list:
            model_module = f'{self.model_type}_model.{model_name}'  # ex) 'classification_model.lgbm'
            model_module = import_module(model_module)  # ex) <module 'classification_model.lgbm' from '/home/jovyan/venv0415/tcr_sampling/assets/train/src_tcr/classification_model/lgbm.py'>
            model_HPO = model_module.DEFAULT_PARAM
            self.model_modules[model_name] = model_module
            self.hpo_settings[model_name] = model_HPO

        return self.hpo_settings

    # CLM에서 model train에 해당
    def model_train(self, df_list, model_name, model_params):
        '''
        - model.py 파일에서 모델 class를 call
        - model.py의 model class의 fit()/pred()을 통해 학습/예측 (df_train / df_valid)

        Args:
            df_list (list)      : [df_train, df_valid] 형태의 list
            model_name (str)    : 모델명 e.g.) cb, lgbm, xgb, ...
            model_params (dict) : 모델 **kwargs

        Returns:
            model (object)          : 학습된 model instance
            pred (object)           : 모델 학습 결과 Numpy Array (tcr_sampled컬럼 none일 경우 추론)
            X_valid.index (object)  : 모델 학습 결과 DataFrame을 생성할 때 필요한 index (tcr.py에서 DataFrame 생성)
        '''

        # If testset mode로 설정한 경우, tcr.py에서 cv_list_by_sud_df를 len 2, idx_df_valid를 1로 설정
        model = self.model_modules[model_name].TCR_model(model_name, self.model_type, model_params)  # ex) <classification_model.lgbm.TCR_model object at 0x7f5d9d04b910>
        df_train, df_valid = df_list[0], df_list[1]
        X_train, y_train = df_train[self.X_features], df_train[self.y_cols]
        X_valid = df_valid[self.X_features]

        # 각 모델 별 fit, pred 진행
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)

        return model, pred, X_valid.index # 재학습 시 X_train.index에서 X_valid.index로 변경

    def model_inference(self, df_inf, model):
        '''
        - model.py 파일에서 모델 class를 call
        - train asset에서 저장된 best_model을 load하여 inference
        - 입력 받은 inf_data_list의 inference 진행

        Args:
            df_inf (object) : inference DataFrame
            model (object)  : inference model instance

        Returns:
            model (object)      : inference model instance
            y_pred (object)     : inference 결과 Numpy Array
            indices (object)    : inference 결과 DataFrame을 생성할 때 필요한 index (tcr.py에서 DataFrame 생성)
        '''

        X_inf = df_inf[self.X_features]

        y_pred = model.predict(X_inf)
        indices = df_inf.index

        return model, y_pred, indices

    def model_save(self, model, path, model_selection, encoding):
        '''
        - model 1개를 주어진 경로에 save
        - self.model을 path에 저장
        - dump model_selection.json to file

        Args:
            model (object)          : best model instance
            path (object)           : best model 저장 경로
            model_selection (dict)  : HPO 결과 dict
            encoding (str)          : encoding type
        '''

        # model instance 저장
        # 기본은 f'{str(path / fname)}.pkl'로 저장
        model.save(path)

        # model_selection.json dump
        model_selection_fname = path / 'model_selection.json'
        with open(model_selection_fname, 'w', encoding=encoding) as f:
            json.dump(model_selection, f, indent=4, ensure_ascii=False)

        model_selection_fname = path / 'model_selection.csv'
        output = save_model_selection(model_selection)
        output.to_csv(model_selection_fname, index=False)

        

    def model_load(self, path, encoding):
        '''
        - 저장된 best model load

        Args:
            path (object)   : best model load 경로
            encoding (str)  : encoding type

        Returns:
            model (object)  : best model instance
        '''

        with open(path / 'model_selection.json', 'r', encoding=encoding) as f:
            model_selection = json.load(f)

        # model_name: lgbm/rf/...
        # model_type: classification / regression
        # model_params: HPO를 통해 선별된 best model의 hyper-parameter
        model_name, model_type, model_params = model_selection['best_model_info'].values()
        model = getattr(import_module(f"{model_selection['best_model_info']['model_type']}_model.{model_name}"), "TCR_model")(model_name, model_type, model_params)

        for item in path.iterdir():
            if item.is_file() and (item.stem == 'best_model'):
                model_path = path / item.name

        model.load(model_path)
        return model

    def calc_shap_value(self, best_model, df_retrain, path_plot, save_plot=True):
        """
        - 전체 학습 데이터셋 중 일부를 Sampling하여 XAI(Shapley) Value를 계산

        Args:
            best_model (object)         : best model instance
            df_retrain (object)         : 전체 학습 DataFrame
            path_plot (object)          : plot 파일 저장 경로
            save_plot (bool, optional)  : plot 파일 저장 Flag

        Returns:
            df_test_shap (object)       : Shapley Value DataFrame
        """

        shapley_sampling = self.config[self.workflow_type]['shapley_sampling']
        shapley_value = self.config[self.workflow_type]['shapley_value']
        x_cols = self.X_features
        shap_cols = [f'TCR-shap_{col}' for col in x_cols]

        #TODO: shapley_sampling asset에서 검증을 따로 하는지
        #(shapley_sampling <= 0이면 안된다 OR shapley_sampling > 1 and isinstance(shapley_sampling, int))
        # shapley_sampling이 0 이거나 False일 경우 전부 nan return
        if not shapley_value:
            self.save_info(f'We will not proceed with setting the Shapley Value to {shapley_value}.')
            return None

        if shapley_sampling <= 0:
            self.save_warning(f'We will not proceed with setting the Shapley Value to {shapley_value}.')
            self.save_warning('Please enter a decimal number between 0 and 1 or a natural number.')
            return None

        df_trained = df_retrain[df_retrain['tcr_sampled'].isin(['sampled', 'over'])] if 'tcr_sampled' in df_retrain.columns else df_retrain
        
        import random 
        
        if 0 < shapley_sampling < 1:
            # df_sampled = df_trained.sample(frac=shapley_sampling, random_state=1024)

            sampled_idx = random.sample(range(len(df_trained)),int(shapley_sampling*len(df_trained)))
        
        elif 1 < shapley_sampling < len(df_trained):
            # df_sampled = df_trained.sample(n=int(shapley_sampling), random_state=1024)
            
            sampled_idx = random.sample(range(len(df_trained)),int(shapley_sampling))

        else:
            # self.logger('warning', '데이터 갯수보다 높은 값을 설정하여 전체 데이터에 대해 진행합니다')
            self.save_warning('We will proceed with the entire data as a higher value than the number of data has been set.')
            # df_sampled = df_trained
            
            sampled_idx = list(range(len(df_trained)))

        # df_sampled_x = df_sampled[x_cols]

        df_sampled_x = df_trained.iloc[sampled_idx][x_cols]

        # Explainer 선언
        df_sampled_np = df_sampled_x.to_numpy()
        try:
            explainer = shap.Explainer(best_model.model)
            shap_values = explainer.shap_values(df_sampled_np)
        except:
            self.save_warning(f'The {best_model.model_name} is not registered in the shap library, so we use the ExactExplainer.')
            self.save_warning(f'Using the ExactExplainer can take a long time.')
            explainer = shap.Explainer(best_model.model.predict, df_sampled_np)
            shap_result = explainer(df_sampled_np)
            shap_values = shap_result.values

        # summary chart 를 파일로 저장
        # TODO: path_model 인자

        if save_plot:
            if not path_plot.exists():
                path_plot.mkdir(parents=True)
            
            value2key = {v:k for k,v in self.column_mapping_dict.items()}
            feature_names = [value2key[col] for col in df_sampled_x.columns]
            shap.summary_plot(shap_values=shap_values, features=df_sampled_x, feature_names=feature_names, plot_type = "bar", plot_size = (25, 8), show=False)
            plt.savefig(path_plot / 'summary_plot.png', bbox_inches='tight')

        ########################################
        ## TODO: 여러 케이스로 테스트 후 최적화 예정
        # output dataframe 저장을 위한 Shapley_df 생성
        if self.model_type == "regression":
            vals = np.abs(shap_values).mean(0)
            shap_data = shap_values
        else: ## classification
            if isinstance(shap_values, list):
                vals = np.abs(shap_values[0]).mean(0)
                shap_data = shap_values[0]
            elif isinstance(shap_values, np.ndarray): ## ndarray
                if shap_values.ndim > 2:  ## class 별로 shap 이 나오는 모델인 경우
                    vals = np.abs(shap_values[0]).mean(0)
                    shap_data = shap_values[0]
                else:
                    vals = np.abs(shap_values).mean(0)
                    shap_data = shap_values

        ########################################
        # iloc index 사용 시 중복 index 처리는 못 함 --> 한 번 reset_index를 하자
        shap_cols = [f'TCR-shap_{col}' for col in x_cols]
        df_test_shap = pd.DataFrame(columns=shap_cols, data=np.full((len(df_retrain), len(x_cols)), np.nan), index=df_retrain.index)
        # df_test_shap.loc[df_sampled.index, :] = shap_data
        df_test_shap.iloc[sampled_idx] = shap_data

        return df_test_shap
    
    def save_info(self, msg):
        self.logger.info(msg)

    def save_warning(self, msg):
        self.logger.warning(msg)

    def save_error(self, msg):
        self.logger.error(msg)
