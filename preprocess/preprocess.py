import numpy as np
import pandas as pd
import pickle
from multiprocessing import Manager # multiprocessing
import multiprocessing, traceback
from tqdm import tqdm
import copy
import os
import sys
sys.path.append(os.getcwd()+'/preprocess/src')
from tabular_preprocess import TabularPreprocess


class Preprocessor():
    def __init__(self, workflow_type, model_path, default_args, user_args, config_file_name, logger_method_dict):
        self.workflow_type      = workflow_type
        self.model_path         = model_path
        self.default_args       = default_args
        self.user_args          = user_args
        self.config_file_name   = config_file_name
        self.logger_method_dict = logger_method_dict

        self.handle_missing_categorical_method = ['frequent']
        self.handle_missing_numeric_method = ['interpolation','mean','median']
    
    ### preprocess 전 data, config 준비 
    def prepare_config_data(self, config, input_data):
        '''
            preprocess 전 config와 data를 준비합니다. 
            1. readiness에서 받은 정보로 input data filtering
            2. config에 default config 세팅 후 user arguments로 config 업데이트
            3. readiness의 categorical/numeric 컬럼 정보로 input data 형변환
            4. default preprocess logic에 따라 config update
        '''
        self.save_info(f"categorical_columns: {config['readiness']['categorical_columns']}")
        self.save_info(f"numeric_columns: {config['readiness']['numeric_columns']}")

        # input_data          = self.readiness_update_input_data(config, input_data, self.workflow_type)
        config              = self.setting_default_configs(config, self.user_args, self.workflow_type)
        config, input_data  = self.change_column_types(config, input_data, self.workflow_type)
        input_data          = self.readiness_update_input_data(config, input_data, self.workflow_type)

        if self.workflow_type == 'train_pipeline':
            config          = self.setting_configs(config, input_data)

        return config, input_data

    ### tabular preprocess 진행
    def tcr_preprocess(self, config, input_data):
        config, prep_output = self.tcr_run_preprocess(config, input_data, self.workflow_type, self.model_path)
        config, output_data = self.make_output(config, input_data, prep_output, self.workflow_type)
        if self.workflow_type == 'train_pipeline':
            with open(file=self.model_path + self.config_file_name, mode='wb') as f:
                pickle.dump(config['preprocess'], f)
        return config, output_data

    ############################################################################################ 
    def convert_new_categories_to_NaN(self, config, input_data):
        '''학습에 사용되지 않은 새로운 범주형 변수값을 결측치로 대체합니다. 
           !!!! 범주형 변수 인코딩 방식이 catboost가 아닌 경우에만 적용됨 !!!'''

        col_encoding_methods = {}
        
        # groupkey X
        if len(config['readiness']['groupkey_columns']) == 0:  
            group = None
            categorical_encoding = config['preprocess']['categorical_encoding']
            res = []
            if 'catboost' in categorical_encoding.keys():
                catboost_cols = categorical_encoding['catboost']
                catboost_cols = [col.split('prep_')[-1] for col in catboost_cols]
                for catboost_col in catboost_cols:
                        if catboost_col in list(config['readiness']['new_category_row_index'].keys()):
                            res.append(catboost_col)
                self.save_warning(f'A new categorical value that was not used in training appears in {res}, but we handled it using CatBoost encoding.')

        # groupkey O
        else:  
            group = list(config['preprocess'].keys())[0]  # group마다 동일한 encoding 방식을 적용한다고 가정
            categorical_encoding = config['preprocess'][group]['categorical_encoding']
            
            groupkeys = list(config['preprocess'].keys())
            for groupkey in groupkeys:
                res = []
                cat_encoding = config['preprocess'][groupkey]['categorical_encoding']
                if 'catboost' in cat_encoding.keys():
                    catboost_cols = cat_encoding['catboost']
                    catboost_cols = [col.split('prep_')[-1] for col in catboost_cols]
                    for catboost_col in catboost_cols:
                        if catboost_col in list(config['readiness']['new_category_row_index'].keys()):
                            res.append(catboost_col)
                    self.save_warning(f'In {groupkey} data: a new categorical value that was not used in training appears in {res}, but we handled it using CatBoost encoding.')
        
        for key, values in categorical_encoding.items():
            for value in values:
                col_encoding_methods[value] = key

        if len(config['readiness']['new_category_row_index']) > 0:
            new_cat_cols = list(config['readiness']['new_category_row_index'].keys())
            res_ = []
            for new_cat_col in new_cat_cols:
                new_cat_col_prep = 'prep_' + new_cat_col
                if col_encoding_methods[new_cat_col_prep] != 'catboost':
                    new_cat_idx = config['readiness']['new_category_row_index'][new_cat_col]
                    input_data.loc[new_cat_idx, new_cat_col_prep] = np.nan
                    res_.append(new_cat_col)
                    # self.save_warning(f'For categorical column {new_cat_col}, a total of {len(new_cat_idx)} data point(s) have been converted to NaN')
            self.save_warning(f'Categorical value(s) that did not appear during training were replaced with NaN for the following categorical columns. : {res_}')
        return input_data


    def readiness_update_input_data(self, config, input_data, workflow_type):
        '''
            asset readiness에서 넘어온 config를 확인하여 데이터를 필터링합니다.
            - train: 
                - (그룹키가 있는 경우) groupkey_list에 있는 groupkey 데이터만 남김
                - (그룹키가 있는 경우) 결측 채우기 
            - inference: 
                - inference_new_data_idx 삭제(categorical column에 inference시 새로운 value 값 들어옴)
                - (그룹키가 있는 경우) groupkey_list에 있는 groupkey 데이터만 남김
        '''
        # self.save_info('input data 필터링 전 shape: ' + str(input_data.shape))
        self.save_info('shape of input data before filtering: ' + str(input_data.shape))
        if workflow_type == 'inference_pipeline':
            # del_idx = config['readiness']['del_idx_list'] <- 삭제된 기능 (get_remove_index)
            # input_data = input_data.drop(del_idx)

            # 학습에 사용되지 않은 새로운 범주형 변수값이 등장한 경우, 이를 결측치로 변환하여 처리
            # catboost를 사용하여 인코딩하는 경우에는 적용 X
            if config['readiness']['ignore_new_category'] == True:
                input_data = self.convert_new_categories_to_NaN(config, input_data)

        if len(config['readiness']['groupkey_columns']) > 0:
            groupkey_column, groupkey_list = config['readiness']['groupkey_columns'][0], config['readiness']['groupkey_list']
            # groupkey 컬럼 별 결측치 있는 경우 처리
            tmp_dict = {groupkey: cols for groupkey, cols in config['readiness']['groupkey_fill_columns'].items() if groupkey in groupkey_list}
            for groupkey, col_list in tmp_dict.items():
                prep_col_list = ['prep_'+c for c in col_list]

                for col in prep_col_list:
                    tmp_idx = input_data[input_data[groupkey_column] == groupkey].index
                    tmp_data = input_data[col].copy() # SettingWithCopyWarning 워닝 안나게 하기 위해 copy
                    if col in config['readiness']['categorical_columns']:
                        tmp_value = input_data[col].value_counts().index[0] # 가장 frequent 한 값을 넣음
                    elif col in config['readiness']['numeric_columns']:
                        tmp_value = input_data[col].max() # 가장 max 값을 넣음
                    else: continue
                    tmp_data.loc[tmp_idx] = [ tmp_value for _ in range(len(tmp_idx))] 
                    input_data[col] = tmp_data
                    # self.save_info(f'{groupkey}의 결측 컬럼 {col}을 {tmp_value}로 채웁니다.')
                    self.save_info(f'The missing column {col} of {groupkey} is filled with {tmp_value}.')

            input_data = pd.concat([input_data[input_data[groupkey_column] == groupkey] for groupkey in groupkey_list])

        # self.save_info('input data 필터링 후 shape: ' + str(input_data.shape))
        self.save_info('shape of input_data after filtering: ' + str(input_data.shape))
        input_data.columns = input_data.columns.astype('string') 
        return input_data

    def setting_default_configs(self, config, args, workflow_type):
        config['preprocess'] = {}
        ### config default settings
        if workflow_type == 'train_pipeline':
            config['preprocess'] = self.default_args

            ### config['preprocess']에 user arguments 값으로 업데이트
            for arg_name, user_value in args.items():
                config['preprocess'][arg_name] = user_value
        else:
            # train에서 사용한 config를 가져와서 덮어 씌움
            with open(file=self.model_path + '/' + self.config_file_name, mode='rb') as f:
                train_config = pickle.load(f)
            config['preprocess'] = train_config['preprocess']

        return config 

    def change_column_types(self, config, input_data, workflow_type):
        def change_to_num(x): # 강제 형변환 함수(형변환 안되면 np.nan으로 변경)
            try: return np.float64(x)
            except: return np.nan

        if (len(config['readiness']['groupkey_columns']) > 0) and (workflow_type == 'inference_pipeline'):
            save_original_columns = config['preprocess'][config['readiness']['groupkey_list'][0]]['save_original_columns']
        else:
            save_original_columns = config['preprocess']['save_original_columns']
        x_columns, categorical_columns, numeric_columns = config['readiness']['x_columns'], config['readiness']['categorical_columns'], config['readiness']['numeric_columns']
        
        ### x_columns 형변환(numeric columns 형변환)
        tmp_x_df = input_data[x_columns].copy()
        for num_col in numeric_columns: 
            tmp_x_df[num_col] = tmp_x_df[num_col].apply(change_to_num)

        tmp_x_df.rename(columns={col:'prep_'+str(col) for col in x_columns}, inplace = True)
        config['readiness']['x_columns']            = tmp_x_df.columns.tolist() 
        config['readiness']['categorical_columns']  = ['prep_'+str(col) for col in categorical_columns] 
        config['readiness']['numeric_columns']      = ['prep_'+str(col) for col in numeric_columns]

        ### y_column 형변환(regression일 때 형변환)
        if workflow_type == 'train_pipeline':
            y_column = config['readiness']['y_column']
            # config['preprocess']['original_y_column'] = config['readiness']['y_column']

            if config['readiness']['task_type'] == 'regression':
                tmp_y = input_data[[y_column]].copy().apply(change_to_num)
            else: 
                tmp_y = input_data[[y_column]].copy()

            tmp_y.rename(columns={y_column:'prep_'+str(y_column)},inplace = True)
            config['readiness']['y_column'] = tmp_y.columns[0]
            config['preprocess']['train_y_column'] = tmp_y.columns[0]

            tmp_x_y_df = [tmp_x_df,tmp_y]
        else: tmp_x_y_df = [tmp_x_df]

        if save_original_columns == True:
            total_df = pd.concat([input_data] + tmp_x_y_df,axis=1)
        else:
            total_df = pd.concat([input_data.drop(columns=x_columns)] + tmp_x_y_df,axis=1)
        return config, total_df
    
    def setting_configs(self, config, input_data):
        def revise_config(config, input_data, df_name, result, function_params):
            config = self.revise_categorical_encoding(config)
            config = self.revise_numeric_preprocess(config,'numeric_scaler')
            config = self.revise_numeric_preprocess(config,'numeric_outlier')
            config = self.revise_handle_missing(config, input_data, df_name)
            if len(config['readiness']['groupkey_columns']) > 0:
                result[df_name] = (config['preprocess'], input_data)
            return df_name, config

        if len(config['readiness']['groupkey_columns']) > 0:
            groupkey_list = config['readiness']['groupkey_list']
            result = self.groupkey_multiprocess_func(copy.deepcopy(config), input_data, revise_config, {})
            tmp_dict = {groupkey: result[groupkey][0] for groupkey in groupkey_list}  
            config['preprocess'] = tmp_dict
            # input_data = pd.concat([result[groupkey][1] for groupkey in groupkey_list])
        else:
            _, config = revise_config(config, input_data, 'non_groupkey', {}, {})
        return config

    def check_method_dict(self, method_dict, compare_columns, chk_str):
        tmp_col_list = []
        for method, col_list in method_dict.items():
            if col_list != chk_str: 
                method_dict[method] = ['prep_'+str(c) for c in col_list]
                tmp_col_list.extend(method_dict[method])
                
        if chk_str in method_dict.values(): 
            chk_str_method = [method for method, col_list in method_dict.items() if col_list == chk_str][0] # all인 것 한개만 가져오기
            rest_columns = list(set(compare_columns) - set(tmp_col_list))
            method_dict[chk_str_method] = rest_columns
        
        return method_dict

    def revise_categorical_encoding(self, config):
        default_method = config['preprocess']['default_encoding']
        categorical_encoding, categorical_columns = config['preprocess']['categorical_encoding'], config['readiness']['categorical_columns']
        if categorical_encoding == {}:
            if len(categorical_columns) > 0:
                config['preprocess']['categorical_encoding'][default_method] = categorical_columns 
            return config

        # 유저 입력 컬럼 확인
        # categorical_encoding = check_method_dict(categorical_encoding, categorical_columns, 'all', 'categorical_encoding', 'categorical')
        categorical_encoding = self.check_method_dict(categorical_encoding, categorical_columns, 'all')

        # 유저가 입력하지 않은 컬럼 default에 배정
        non_chosen_columns = set(categorical_columns) - set(np.concatenate(list(categorical_encoding.values())))   
        if default_method in categorical_encoding:
            categorical_encoding[default_method].extend(list(non_chosen_columns))
        else: 
            categorical_encoding[default_method] = list(non_chosen_columns)

        config['preprocess']['categorical_encoding'] = categorical_encoding
        return config

    def revise_numeric_preprocess(self, config, numeric_preps):
        # numeric_preps: 'numeric_scaler', 'numeric_outlier' 
        numeric_preps_dict, numeric_columns = config['preprocess'][numeric_preps], config['readiness']['numeric_columns']
        if numeric_preps_dict == {}: # 방법론 적용 X
            return config
        numeric_preps_dict = self.check_method_dict(numeric_preps_dict, numeric_columns, 'all')
        config['preprocess'][numeric_preps] = numeric_preps_dict
        return config

    def revise_handle_missing(self, config, input_data, df_name):
        def default_handle_missing(config, num_col, cat_col, input_data, df_name): # handling missing의 default
            handle_missing, missing_split_rate = config['preprocess']['handle_missing'], config['preprocess']['missing_split_rate']
            total_col = num_col + cat_col
            missing_rate = 1 - (len(input_data[config['readiness']['categorical_columns'] + config['readiness']['numeric_columns']].dropna())/len(input_data))
            self.save_info(f'{df_name} dataframe missing rate: {round(missing_rate,5)}')

            if missing_rate >= missing_split_rate: # 결측치 0.0 이상일 때 default
                if len(cat_col) > 0:
                    if 'frequent' not in handle_missing:
                        handle_missing['frequent'] = cat_col
                    else: 
                        handle_missing['frequent'].extend(cat_col)
                if len(num_col) > 0:
                    if 'median' not in handle_missing:
                        handle_missing['median'] = num_col
                    else:
                        handle_missing['median'].extend(num_col)
            else: 
                if len(total_col) > 0:
                    if 'drop' not in handle_missing:
                        handle_missing['drop'] = total_col
                    else:
                        handle_missing['drop'].extend(total_col)
            config['preprocess']['handle_missing'] = handle_missing
            return config
        
        handle_missing = config['preprocess']['handle_missing']
        numeric_columns, categorical_columns = config['readiness']['numeric_columns'], config['readiness']['categorical_columns']
        # default dict 꾸미기
        if handle_missing == {}:
            config = default_handle_missing(config, numeric_columns, categorical_columns, input_data, df_name) # 전체 컬럼 넣기
            return config

        _tmp_new_all_dict = {}
        # handle_missing에서 categorical 방법론 체크
        tmp_categorical_dict = {method: col_list for method, col_list in handle_missing.items() if method in self.handle_missing_categorical_method}
        _tmp_new_all_dict.update(self.check_method_dict(tmp_categorical_dict, categorical_columns, 'categorical_all'))

        if len(tmp_categorical_dict.values()):
            categorical_columns   = list(set(categorical_columns) - set(np.concatenate(list(tmp_categorical_dict.values()))))  
            
        # handle_missing에서 numeric 방법론 체크
        tmp_numeric_dict = {method: col_list for method, col_list in handle_missing.items() if method in self.handle_missing_numeric_method}

        _tmp_new_all_dict.update(self.check_method_dict(tmp_numeric_dict, numeric_columns, 'numeric_all'))
        if len(tmp_numeric_dict.values()):
            numeric_columns   = list(set(numeric_columns) - set(np.concatenate(list(tmp_numeric_dict.values())))) 
        
        # handle_missing에서 all_type 방법론 체크
        tmp_drop_dict = {method: col_list for method, col_list in handle_missing.items() if method == 'drop'}
        tmp_fill_dict = {method: col_list for method, col_list in handle_missing.items() if method[:4] == 'fill'}

        if (len(tmp_drop_dict) > 0) or (len(tmp_fill_dict) > 0):   
            _tmp_drop_not_all = ['prep_'+str(col) for col in [c for c in tmp_drop_dict.values() if c not in ['categorical_all','numeric_all','all']]]
            _tmp_fill_not_all = ['prep_'+str(col) for col in [c for c in tmp_fill_dict.values() if c not in ['categorical_all','numeric_all','all']]]
            tmp_total_columns = _tmp_drop_not_all + _tmp_fill_not_all
        else: 
            tmp_total_columns = []
        _left_categorical_columns = list(set(categorical_columns) - set(tmp_total_columns))
        _left_numeric_columns = list(set(numeric_columns) - set(tmp_total_columns))

        # drop    
        if len(tmp_drop_dict) > 0:
            if tmp_drop_dict['drop'] == 'categorical_all':
                _tmp_new_all_dict['drop'] = _left_categorical_columns
            elif tmp_drop_dict['drop'] == 'numeric_all': 
                _tmp_new_all_dict['drop'] = _left_numeric_columns
            elif tmp_drop_dict['drop'] == 'all':
                _tmp_new_all_dict['drop'] = _left_categorical_columns + _left_numeric_columns
            else:
                _tmp_new_all_dict['drop'] = ['prep_'+str(c) for c in tmp_drop_dict['drop']]                            
        # fill
        if len(tmp_fill_dict) > 0:
            for method, col_list in tmp_fill_dict.items():
                fill_value = method[5:]
                if col_list == 'categorical_all':
                    _tmp_new_all_dict['fill_category_'+fill_value] = _left_categorical_columns
                elif col_list == 'numeric_all':
                    _tmp_new_all_dict['fill_numeric_'+fill_value] = _left_numeric_columns
                elif col_list == 'all':
                    _tmp_new_all_dict['fill_category_'+fill_value] = _left_categorical_columns
                    _tmp_new_all_dict['fill_numeric_'+fill_value] = _left_numeric_columns
                else:
                    _cat_col_list = ['prep_'+str(c) for c in col_list if 'prep_'+str(c) in categorical_columns] 
                    _num_col_list = ['prep_'+str(c) for c in col_list if 'prep_'+str(c) in numeric_columns]
                    if len(_cat_col_list) > 0:
                        _tmp_new_all_dict['fill_category_'+fill_value] = _cat_col_list
                    if len(_num_col_list) > 0:
                        _tmp_new_all_dict['fill_numeric_'+fill_value] = _num_col_list

        config['preprocess']['handle_missing'] = _tmp_new_all_dict
        tmp_num_cols = list(set(numeric_columns) - set(np.concatenate(list(_tmp_new_all_dict.values()))))
        tmp_cat_cols = list(set(categorical_columns) - set(np.concatenate(list(_tmp_new_all_dict.values())))) 
        config = default_handle_missing(config, tmp_num_cols, tmp_cat_cols, input_data, df_name)
        return config 

    def tcr_run_preprocess(self, config, input_data, workflow_type, model_path):
        function_params = {'workflow_type': workflow_type} # run_preprocess_detaul 함수로 넘겨줄 parameters
        if len(config['readiness']['groupkey_columns']) > 0: # groupkey 있을 때
            groupkey_list = config['readiness']['groupkey_list']
            function_params.update({'model_path':{groupkey: model_path+str(groupkey)+'/' for groupkey in groupkey_list}})
            result = self.groupkey_multiprocess_func(config, input_data, self.tcr_run_preprocess_detail, function_params)

            groupkey_df = pd.concat([result[groupkey][0] for groupkey in groupkey_list], axis=0) # groupkey_df가 행끼리 붙음
            output_data = groupkey_df.fillna(0) # categorical_encoding 결과 df concat시 발생하는 결측치 nan값으로 채우기
            for groupkey in groupkey_list:
                config['preprocess'][groupkey]['y_mapping_table'] = result[groupkey][1]     
            y_mapping_dict = result[groupkey][1] # groupkey 별 다 동일함. 마지막 groupkey의 mapping table 넣음

        else: # groupkey 없을 때
            function_params.update({'model_path':{'non_groupkey': model_path}})
            output_data, y_mapping_dict = self.tcr_run_preprocess_detail(copy.deepcopy(config), input_data, 'non_groupkey', {}, function_params)
            
            config['preprocess']['y_mapping_table'] = y_mapping_dict
        
        new_target_label = []
        for enc_name, original_name in y_mapping_dict.items():
            if original_name in config['readiness']['target_label']:
                new_target_label.append(enc_name)
        config['readiness']['target_label'] = new_target_label
        return config, output_data

    def tcr_run_preprocess_detail(self, config, input_data, df_name, result, param_dict):
        def data_prep_run(config_detail, input_data, workflow_type, model_path): # config_detail: config['preprocess'] or config['preprocess']['groupkey']
            tp = TabularPreprocess(self.logger_method_dict, input_data, config_detail, workflow_type)
            if workflow_type == 'train_pipeline':
                tmp_df, _ = tp.run(workflow_type)
                tp.save_transformer(model_path)
            else:
                tp.load_transformer(model_path)
                tmp_df, _ = tp.run(workflow_type)
            return tmp_df

        if df_name != 'non_groupkey': # 그룹키 있을 때
            config_detail = config['preprocess'][df_name]
        else: 
            config_detail = config['preprocess']

        categorical_columns, numeric_columns = config['readiness']['categorical_columns'], config['readiness']['numeric_columns']
        workflow_type = param_dict['workflow_type']
        # preprocess 준비
        # x_columns preprocess 준비
        config_detail.update({'categorical_columns': categorical_columns, 
                        'numeric_columns': numeric_columns
                        })
        try:
            if not os.path.exists(param_dict['model_path'][df_name]):
                os.makedirs(param_dict['model_path'][df_name])
        except:
            # self.save_error(f'폴더 경로 생성에 문제가 있습니다. :{param_dict["model_path"][df_name]}')
            self.save_error(f'There seems to be an issue with creating the folder path: {param_dict["model_path"][df_name]}. Please check if the path is correct and accessible.')
        x_model_path = param_dict['model_path'][df_name] + 'train_pipeline_x.pkl'

        if 'catboost' in config_detail['categorical_encoding'].keys():
            if workflow_type == 'train_pipeline':
                # CatBoost 인코딩의 경우 타깃 칼럼에 대한 정보가 반드시 필요함
                y_column = [config['readiness']['y_column']]
                prep_x_df = data_prep_run(config_detail, input_data[categorical_columns + numeric_columns + y_column], workflow_type, x_model_path)
            else:
                prep_x_df = data_prep_run(config_detail, input_data[categorical_columns + numeric_columns], workflow_type, x_model_path)
        else:
            prep_x_df = data_prep_run(config_detail, input_data[categorical_columns + numeric_columns], workflow_type, x_model_path)

        if workflow_type == 'train_pipeline':
            # y_column preprocess 준비
            y_column = config['readiness']['y_column']

            if config['readiness']['task_type'] == 'classification':
                tmp_config_y = {'categorical_columns': [y_column],
                                'numeric_columns': [], 
                                'handle_missing':{'drop': [y_column]}, 
                                'categorical_encoding':{'label': [y_column]}, 
                                'numeric_scaler':{}, 
                                'numeric_outlier':{}}
            else:
                tmp_config_y = {'categorical_columns': [],
                                'numeric_columns': [y_column], 
                                'handle_missing':{'drop': [y_column]}, 
                                'categorical_encoding':{}, 
                                'numeric_scaler':{}, 
                                'numeric_outlier':{}}
            y_model_path = param_dict['model_path'][df_name] + 'train_pipeline_y.pkl'
        
            prep_y_df = data_prep_run(tmp_config_y, input_data[[y_column]], workflow_type, y_model_path)
        
            # y_mapping_dict = {prep_y_df[y_column].iloc[i]: input_data[config_detail['original_y_column']].iloc[i] for i in range(len(input_data)) }
            if config['readiness']['task_type'] == 'classification':
                original_y_column = config['readiness']['original_y_column']
                # y_mapping_dict = {int(prep_y_df[y_column].iloc[i]): input_data[config_detail['original_y_column']].iloc[i] for i in range(len(input_data))}
                # y_mapping_dict = {int(prep_y_df[y_column].iloc[i]): input_data[original_y_column].iloc[i] for i in range(len(input_data))}

                get_label_idx = [prep_y_df[prep_y_df[y_column]==label].index[0] for label in prep_y_df[y_column].unique()]
                y_mapping_dict = {int(prep_y_df[y_column].loc[i]): input_data[original_y_column].loc[i] for i in get_label_idx}
            else:
                y_mapping_dict = {}
            output_data = pd.concat([prep_x_df,prep_y_df],axis=1).dropna() # index 안맞는 경우 발생하는 결측치 삭제
            output_data[prep_y_df.columns] = output_data[prep_y_df.columns].astype(int)
        else: 
            y_mapping_dict = config_detail['y_mapping_table']
            output_data = prep_x_df

        result[df_name] = output_data, y_mapping_dict
        return output_data, y_mapping_dict

    def groupkey_multiprocess_func(self, config, input_data, function, function_params):
        groupkey_column, groupkey_list = config['readiness']['groupkey_columns'][0], config['readiness']['groupkey_list']
        df_dict = {groupkey: input_data[input_data[groupkey_column] == groupkey] for groupkey in groupkey_list}
        ### multiprocessing
        manager = Manager()
        result  = manager.dict()
        procs   = []

        for groupkey, groupkey_df in df_dict.items():
            proc = Process(target=function, args=(copy.deepcopy(config), groupkey_df, groupkey, result, function_params))
            procs.append(proc)
            proc.start()
        try:
            for proc in tqdm(procs):
                proc.join()
            if proc.exception:
                raise proc.exception()
        except:
            # raise Exception('groupkey 데이터의 preprocess multiprocessing 중 에러가 발생했습니다. process를 종료합니다.')
            raise Exception('An error occurred during the preprocessing multiprocessing of the groupkey data. The process will be terminated.')

        # result = {groupkey: {config['preprocess'], groupkey_df}}
        return result 

    def make_output(self,config, input_data, prep_output, workflow_type):
        categorical_columns, numeric_columns, x_columns = config['readiness']['categorical_columns'], config['readiness']['numeric_columns'], config['readiness']['x_columns']
        if workflow_type == 'train_pipeline':
            y_column = config['readiness']['y_column']
            prep_col_list = x_columns + [y_column]
            non_categorical_columns = numeric_columns + [y_column]
        else:
            prep_col_list = x_columns
            non_categorical_columns = numeric_columns 
        
        non_used_df = input_data.drop(columns = prep_col_list)
        output_df = pd.concat([non_used_df.loc[prep_output.index], prep_output], axis=1)
        output_df.sort_index(inplace=True)

        categorical_encoded_cols = [col for col in prep_output.columns if col not in non_categorical_columns]  
        config['readiness']['categorical_columns'] = categorical_encoded_cols
        config['readiness']['x_columns'] = categorical_encoded_cols + numeric_columns
        
        return config, output_df

    def save_info(self, msg):
        # self.logger_method_dict['info'](msg)
        self.logger_method_dict.info(msg)
     
    def save_warning(self, msg):
        # self.logger_method_dict['warning'](msg)
        self.logger_method_dict.warning(msg)
 
    def save_error(self, msg):
        # self.logger_method_dict['error'](msg)
        self.logger_method_dict.error(msg)

class Process(multiprocessing.Process): # multiprocess 예외처리를 위함
    def __init__(self, *args, **kwargs):
        multiprocessing.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception = None

    def run(self):
        try:
            multiprocessing.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            #raise e  # You can still rise this exception if you need to

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception
