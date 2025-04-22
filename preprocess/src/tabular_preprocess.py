import os
import sys
import pickle
import numpy as np
import pandas as pd
import itertools
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, FunctionTransformer
from category_encoders import BinaryEncoder, CatBoostEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class CustomLabelEncoder(FunctionTransformer):
    def __init__(self):
        super().__init__()
        self.classes_dict = {}
        self.i = 0
        self.j = 0
        
    def fit(self, X, y=None):
        if type(X)==type(pd.DataFrame()):
            for col_name in X.columns:
                le = LabelEncoder()
                le.fit(X[col_name])
                self.classes_dict[col_name] = le.classes_
        elif type(X)==type(pd.Series()):
            le = LabelEncoder()
            le.fit(X)
            self.classes_dict[X.name] = le.classes_
        else:
            le = LabelEncoder()
            le.fit(X)
            self.classes_dict[self.i] = le.classes_
            self.i += 1

        return self
    
    def transform(self, X ,y=None):
        if type(X)==type(pd.DataFrame()):
            for col_name in X.columns:
                le = LabelEncoder()
                le.classes_ = self.classes_dict[col_name]
                X = X.copy()
                X[col_name] = le.transform(X[col_name])
        elif type(X)==type(pd.Series()):
            le = LabelEncoder()
            le.classes_ = self.classes_dict[X.name]
            X = le.transform(X)
        else:
            le = LabelEncoder()
            le.classes_ = self.classes_dict[self.j]
            X = le.transform(X)
            self.j += 1

        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
class NormalOutlier(FunctionTransformer):
    def __init__(self):
        super().__init__()
    
    def fit(self, X, y=None):
        sigma = 3
        m = X.mean()
        std = X.std()
        self.upper = m + sigma * std
        self.lower = m - sigma * std
        return self

    def transform(self, X ,y=None):
        if type(X)==type(pd.DataFrame()):
            condition = ((self.upper > X) & (self.lower < X)).all(axis=1)
        else: 
            condition = ((self.upper > X) & (self.lower < X))
        X[~condition] = float('nan')
        return X
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

class DropNa(FunctionTransformer):
    def __init__(self):
        super().__init__()
    
    def fit(self, X, y=None):
        
        return self
    
    def transform(self, X ,y=None):

        if type(X)==type(pd.DataFrame()):
            X = X.dropna()

        elif type(X)==type(pd.Series()):
            X = X.dropna()
        else:
            X = pd.DataFrame(X).dropna().values
        return X
    

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class TabularPreprocess:
    def __init__(self, logger_method_dict, df=None, config_detail=None, workflow_type='train_pipeline'): 
        if df is None:
            self.df = pd.read_csv('/nas001/users/sujin2.lee/share/sample_data/titanic/train/train.csv') # 타이타닉 read
            self.df['Pclass'].iloc[[0,20,40,60,100]] = np.nan
            self.df['Fare'].iloc[[0,20,40,60,100]] = np.nan
            self.df['SibSp'].iloc[[0,20,40,60,100]] = np.nan
            
        else:
            self.df = df

        if config_detail is None:
            self.config = {
                'categorical_columns'      : ['Sex','Pclass','Survived','Embarked','SibSp','Parch'],
                'numeric_columns'       : ['Fare','Age'],
                'other_columns'         : ['PassengerId', 'Name', 'Ticket', 'Cabin'], # 학습에 사용하지 않은 항목(없어야 정상)
                'categorical_encoding' : {'onehot':['Sex','Pclass'], 'label':['Embarked']}, # binary는 요청 시 추가
                'handle_missing'        : {'frequent':['Pclass', 'Embarked'], 'fill_1':['SibSp'], 'drop':['Fare','Age']},
                'numeric_outlier'       : {'normal':['Fare']}, # {'normal':['Fare'], 'quantile':['Age']},
                'numeric_scaler'        : {'standard':['Fare','Age']}, # standard, minmax, maxabs, robust, normalizer
            }
        else:
            self.config = config_detail # config['preprocess'] or config['preprocess']['groupkey']
        self.config['before_train_columns'] = self.df.columns.tolist()
        self.logger_method_dict = logger_method_dict

        self.numeric_outlier_level = 99
        
        self.categorical_encoding = self.config['categorical_encoding']
        self.category_columns     = self.config['categorical_columns']
        self.numeric_columns      = self.config['numeric_columns']
        # self.other_columns        = self.config['other_columns']
        self.handle_missing       = self.config['handle_missing']
        self.handle_outlier       = self.config['numeric_outlier']
        self.handle_scaler        = self.config['numeric_scaler']

        if 'catboost' in self.categorical_encoding.keys():
            if workflow_type == 'train_pipeline':
                total_columns = self.category_columns + self.numeric_columns + [self.config['train_y_column']]
            else:
                total_columns = self.category_columns + self.numeric_columns
        else:
            total_columns = self.category_columns + self.numeric_columns # + self.other_columns
        self.df = self.df[total_columns]
        self.column_to_index = {col:idx for idx, col in enumerate(total_columns)}

        # Add identity(to keep columns during column transformer)
        self.add_identity_columns()
        # self.user_function = config['user_function']

    def run(self, pipeline):
        if pipeline == 'train_pipeline':
            steps = []
            if len(self.handle_missing) > 0       :   steps.append(self.make_missing_handlers())
            if len(self.handle_outlier) > 0       :   steps.append(self.make_outlier_handlers())
            steps.append(self.drop_nan())
            if len(self.handle_scaler) > 0        :   steps.append(self.make_scalers()) # scaler의 경우 outlier제거한 뒤 적용
            if len(self.categorical_encoding) > 0 :   steps.append(self.make_encoders()) # ohe는 차원이 증가하기 때문에 마지막에 수행
            
            transformer = Pipeline(steps)
            if 'catboost' in self.categorical_encoding.keys():
                # CatBoost 인코딩의 경우 타깃 칼럼에 대한 정보가 반드시 필요함
                y_col = self.config['train_y_column']
                y = self.df[[y_col]]
                le = CustomLabelEncoder()
                y_encoded = le.fit_transform(y)
                
                X_cols =  self.config['categorical_columns'] + self.config['numeric_columns']
                X = self.df[X_cols]
                transformer.fit(X, y_encoded)  
                df = transformer.transform(X)
                self.config['before_train_columns'] = X_cols
  
            else:
                df = transformer.fit_transform(self.df)

            self.config['after_train_columns'] = df.columns.tolist()
            self.transformer = transformer
            # self.save_transformer(transformer)

        else:
            # load pipeline
            self.df = self.df[self.config['before_train_columns']]
            transformer = self.transformer

            # pandas method로 실행되는 처리(missing, outlier 등) 수행
            if len(self.handle_missing) > 0       :   self.make_missing_handlers()
            if len(self.handle_outlier) > 0       :   self.make_outlier_handlers()
            if len(self.handle_scaler) > 0        :   self.make_scalers() # scaler의 경우 outlier제거한 뒤 적용
            if len(self.categorical_encoding) > 0 :   self.make_encoders() # ohe는 차원이 증가하기 때문에 마지막에 수행

            df = transformer.transform(self.df)

        ### Pipeline 저장

        return df, transformer
    
    def add_identity_columns(self):
        total_columns = self.numeric_columns + self.category_columns # + self.other_columns

        if len(self.categorical_encoding) > 0:
            categorical_encoding_columns = list(itertools.chain(*self.categorical_encoding.values()))
            self.categorical_encoding['identity'] = sorted([item for item in total_columns if item not in categorical_encoding_columns])
        
        if len(self.handle_missing) > 0:
            handle_missing_columns = list(itertools.chain(*self.handle_missing.values()))
            self.handle_missing['identity'] = sorted([item for item in total_columns if item not in handle_missing_columns])
        
        if len(self.handle_outlier) > 0:
            handle_outlier_columns = list(itertools.chain(*self.handle_outlier.values()))    
            self.handle_outlier['identity'] = sorted([item for item in total_columns if item not in handle_outlier_columns])
        
        if len(self.handle_scaler) > 0:
            handle_scaler_columns = list(itertools.chain(*self.handle_scaler.values())) 
            self.handle_scaler['identity'] = sorted([item for item in total_columns if item not in handle_scaler_columns])

        # if len(self.dropna_dict) > 0:
        #     handle_dropna_columns = list(itertools.chain(*self.dropna_dict.values())) 
        #     self.dropna_dict['identity'] = sorted([item for item in total_columns if item not in handle_dropna_columns])

    def make_encoders(self):
        # self.save_info(f'>>>>> categorical_encoding을 시작합니다.')
        self.save_info(f'>>>>> Starting the categorical encoding process.')
        encoding_list = []
        column_list = []
        for method, col_names in self.categorical_encoding.items():
            if method != 'identity' and len(col_names) > 0:
                # self.save_info(f'{col_names} 컬럼에 {method} encoding 방법론을 적용합니다. ')
                self.save_info(f'Applying {method} encoding methodology to the {col_names} column(s).')
            col_idxs = [self.column_to_index[item] for item in col_names]
            column_list.extend(col_names)
            if method == 'onehot':
                encoding_list.append((method, OneHotEncoder(sparse_output=False), col_idxs))

            elif method == 'label':
                encoding_list.append((method, CustomLabelEncoder(), col_idxs))
                
            elif method == 'binary':
                encoding_list.append((method, BinaryEncoder(), col_idxs)) 
                # BinaryEncoder -> https://contrib.scikit-learn.org/category_encoders/ category_encoders라이브러리 받으면 있는듯..? fit/transform 형태

            elif method == 'catboost':
                encoding_list.append((method, CatBoostEncoder(), col_idxs)) 
            
            elif method == 'identity':
                encoding_list.append(("identity", FunctionTransformer(feature_names_out='one-to-one'), col_idxs))
        self.column_to_index = {item:i for i, item in enumerate(column_list)}
        return ('categorical_encoding', 
        ColumnTransformer(encoding_list, verbose_feature_names_out=False, sparse_threshold=0).set_output(transform='pandas'))
    
    def make_missing_handlers(self):
        # self.save_info(f'>>>>> 결측치 처리(handle_missing)를 시작합니다.')
        self.save_info(f'>>>>> Starting missing value handling (handle_missing).')

        handle_missing_list = []
        column_list = []

        for method, col_names in self.handle_missing.items():
            if method != 'identity' and len(col_names) > 0:
                # self.save_info(f'{col_names} 컬럼에 {method} 결측치 처리 방법론을 적용합니다. ')
                self.save_info(f'Applying {method} missing value handling methodology to the {col_names} column(s).')

            col_idxs = [self.column_to_index[item] for item in col_names]
            column_list.extend(col_names)

            if method == 'frequent':
                handle_missing_list.append((method, SimpleImputer(strategy='most_frequent'), col_idxs))

            elif method == 'mean':
                handle_missing_list.append((method, SimpleImputer(strategy='mean'), col_idxs))

            elif method == 'median':
                handle_missing_list.append((method, SimpleImputer(strategy='median'), col_idxs))

            elif method[:4] == 'fill':
                if method.split('_')[1] == 'category':
                    handle_missing_list.append((method, SimpleImputer(strategy='constant', fill_value=method.split('_')[-1]), col_idxs))
                else:
                    handle_missing_list.append((method, SimpleImputer(strategy='constant', fill_value=np.float64(method.split('_')[-1])), col_idxs))

            elif method == 'interpolation':
                self.df[col_names] = self.df[col_names].interpolate(limit_direction='both')
                handle_missing_list.append(("{}_pandas".format(method), FunctionTransformer(feature_names_out='one-to-one'), col_idxs))

            elif method == 'drop':
                self.df = self.df.dropna(subset=col_names)
                handle_missing_list.append(("{}_pandas".format(method), FunctionTransformer(feature_names_out='one-to-one'), col_idxs))

            elif method == 'identity':
                handle_missing_list.append(("identity", FunctionTransformer(feature_names_out='one-to-one'), col_idxs))
        
        self.column_to_index = {item:i for i, item in enumerate(column_list)}
        return ('handle_missing',
        ColumnTransformer(handle_missing_list, verbose_feature_names_out=False, sparse_threshold=0).set_output(transform='pandas'))
    

    def drop_nan(self):
        return ('drop_nan', DropNa())

    def make_scalers(self):
        # self.save_info(f'>>>>> 수치 데이터 scaling(numeric_scaler)을 시작합니다.')
        self.save_info(f'>>>>> Starting numeric data scaling (numeric_scaler).')

        scale_list = []
        column_list = []

        for method, col_names in self.handle_scaler.items():
            if method != 'identity' and len(col_names) > 0:
                # self.save_info(f'{col_names} 컬럼에 {method} scaler 방법론을 적용합니다. ')
                self.save_info(f'Applying {method} scaler methodology to the {col_names} column(s).')

            col_idxs = [self.column_to_index[item] for item in col_names]
            column_list.extend(col_names)

            if method == 'standard':
                scale_list.append((method, StandardScaler(), col_idxs))

            elif method == 'minmax':
                scale_list.append((method, MinMaxScaler(), col_idxs))

            elif method == 'maxabs':
                scale_list.append((method, MaxAbsScaler(), col_idxs))

            elif method == 'robust':
                scale_list.append((method, RobustScaler(), col_idxs))

            elif method == 'normalizer':
                scale_list.append((method, Normalizer(), col_idxs))

            elif method == 'identity':
                scale_list.append(("identity", FunctionTransformer(feature_names_out='one-to-one'), col_idxs))
        
        self.column_to_index = {item:i for i, item in enumerate(column_list)}
        return ('numeric_scaler', 
        ColumnTransformer(scale_list, verbose_feature_names_out=False, sparse_threshold=0).set_output(transform='pandas'))

    def make_outlier_handlers(self):
        # self.save_info(f'>>>>> 수치 데이터 outlier 제거(numeric_outlier)를 시작합니다.')
        self.save_info(f'>>>>> Starting numeric data outlier removal (numeric_outlier).')


        outlier_list = []
        column_list = []

        for method, col_names in self.handle_outlier.items():
            if method != 'identity' and len(col_names) > 0:
                # self.save_info(f'{col_names} 컬럼에 {method} outlier 처리 방법론을 적용합니다. ')
                self.save_info(f'Applying {method} outlier handling methodology to the {col_names} column(s).')

            col_idxs = [self.column_to_index[item] for item in col_names]
            column_list.extend(col_names)
            
            if method == 'normal': 
                # self.df = self.outlier_normal(self.df, col_names) # nan처리 안된 상태의 self.df를 바꿈  
                
                # outlier_list.append(("normal", FunctionTransformer(self.outlier_normal, kw_args={"condition": self.condition}, feature_names_out='one-to-one'), col_idxs))
                
                outlier_list.append(("normal", NormalOutlier(), col_idxs))
            
            elif method == 'identity':
                outlier_list.append(("identity", FunctionTransformer(feature_names_out='one-to-one'), col_idxs))
        
        self.column_to_index = {item:i for i, item in enumerate(column_list)}
        return ('handle_outlier', 
        ColumnTransformer(outlier_list, verbose_feature_names_out=False, sparse_threshold=0).set_output(transform='pandas'))
    
    def outlier_normal(self, X, sigma=3, condition=None):
        col_names = X.columns
        m = X[col_names].mean()
        std = X[col_names].std()
        upper = m + sigma * std
        lower = m - sigma * std
        if condition is not None:
            self.condition = ((upper > X[col_names]) & (lower < X[col_names])).all(axis=1)
        else:
            self.condition = condition

        X = X[self.condition]

    def save_transformer(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.transformer, f, pickle.HIGHEST_PROTOCOL)

    def load_transformer(self, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        self.transformer = obj
        return obj

    def save_info(self, msg):
        # self.logger_method_dict['info'](msg)
        self.logger_method_dict.info(msg)
     
    def save_warning(self, msg):
        # self.logger_method_dict['warning'](msg)
        self.logger_method_dict.warning(msg)
 
    def save_error(self, msg):
        # self.logger_method_dict['error'](msg)
        self.logger_method_dict.error(msg)

# if __name__=='__main__':
#     # train pipeline
#     dp = DataPreprocess(None, None)
    
#     # train pipeline
#     df, transformer = dp.run('train_pipeline')

#     print('변환 후', df.shape)
#     print('config', dp.config)

#     dp.save_transformer('temp.pkl')

#     ## inference pipeline
#     dp = DataPreprocess(None, None)

#     dp.load_transformer('temp.pkl')

#     df, transformer = dp.run('inference_pipeline')

#     print('after inference pipeline', df.shape)

#     print(dp.df.shape)