import numpy as np
import pandas as pd
from collections import Counter


class readiness_check(object):
    def __init__(self, logger_method_dict):
        self.logger_method_dict = logger_method_dict
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        print(self.logger_method_dict )
    
    def train(self, config, input_data):
        config, input_data = self.convert_column_names(config, input_data)
        self.check_dataframe(config, input_data, 'train')
        config, input_data = self.check_groupkey(config, input_data, 'train')

        ### x 컬럼 결측치 검사 
        config = self.check_x_NA(config, input_data)

        ### x 컬럼 type 검사 
        x_total_cols, x_num_cols, x_cat_cols, config = self.check_column_types(config['readiness']['x_columns'], config, input_data)

        # info 
        if len(x_cat_cols) > 0:   
            # self.save_info(f'학습 컬럼 중 {x_cat_cols} 컬럼이 categorical 컬럼으로 분류되었습니다.')
            self.save_info(f'The column {x_cat_cols} from the x_columns has been classified as a categorical column.')
        if len(x_num_cols) > 0:   
            # self.save_info(f'학습 컬럼 중 {x_num_cols} 컬럼이 numeric 컬럼으로 분류되었습니다.')
            self.save_info(f'The column {x_num_cols} from the x_columns has been classified as a numeric column.')
        
        config['readiness']['x_columns'] = x_total_cols
        config['readiness']['categorical_columns'] = x_cat_cols
        config['readiness']['numeric_columns'] = x_num_cols

        ### classification일 때, y 컬럼 unique 1 검사
        y_column = config['readiness']['y_column']
        if config['readiness']['task_type'] == 'classification':
            _, _, config = self.categorical_update([], [], y_column, input_data, config) 

            if len(config['readiness']['groupkey_list']) > 0:  # groupkey O
                config = self.check_y_group(config, input_data)

            config = self.check_y_unique(config, input_data)
            config = self.set_target_label(config, input_data)

        ### train시 학습에 필요한 최소 데이터 수 검사
        config = self.check_min_rows(config, input_data)

        return config, input_data


    def inference(self, config, input_data):
        config, input_data = self.convert_column_names(config, input_data)
        self.check_dataframe(config, input_data, 'inference')
        config, input_data = self.check_groupkey(config, input_data, 'inference')
        groupkey_columns = config['readiness']['groupkey_columns']

        ### inference시 new category 등장 여부 검사
        # x_columns 검사
        config = self.check_inference_new_category(config, input_data, config['readiness']['ignore_new_category'] ) 
        
        return config, input_data

    
    def check_dataframe(self, config, input_data, mode):
        '''
        input_data에 아래 컬럼이 있는지 해서 없으면 error 발생
        - x_columns, y_column
        - 값이 있을 때) groupkey_columns, categorical_columns, numeric_columns
        '''
        def check_columns(selected_columns, all_columns, mode):
            '''config에 지정된 칼럼들이(selected_columns) input_data의 칼럼들(all_columns)에
               모두 있는 지 확인하는 함수'''
            if type(selected_columns) != list:
                selected_columns = list(selected_columns)
            if set(selected_columns).issubset(all_columns):
                # self.save_info(f"{mode} - 모든 칼럼이 데이터프레임에 존재합니다.")
                self.save_info(f"{mode} - All columns exist in the dataframe.")
                return True
            else:
                existing_columns = [col for col in selected_columns if col in all_columns ]
                missing_columns = [col for col in selected_columns if col not in all_columns ]

                # self.save_error(f"{mode} - 일부 칼럼이({missing_columns}) 데이터프레임에 존재하지 않습니다.")
                self.save_error(f"{mode} - some columns ({missing_columns}) do not exist in the dataframe.")
                return False

        check_x = check_columns(config['readiness']['x_columns'], input_data.columns, 'x_columns')
        if mode == 'train':
            check_y = check_columns([config['readiness']['y_column']], input_data.columns, 'y_column')
            if check_y:
                config['readiness']['original_y_column'] = config['readiness']['y_column']
        elif mode == 'inference':
            check_y = True
    
        check_groupkey = True ; check_categorical = True ; check_numeric = True
    
        if config['readiness']['groupkey_columns']:
            check_groupkey = check_columns(config['readiness']['groupkey_columns'], input_data.columns, 'groupkey_columns')
    
        if config['readiness']['categorical_columns']:
            check_categorical = check_columns(config['readiness']['groupkey_columns'], input_data.columns, 'categorical_columns')
        
        if config['readiness']['numeric_columns']:
            check_numeric = check_columns(config['readiness']['numeric_columns'], input_data.columns, 'numeric_columns')
        
        check_lists = [check_x, check_y, check_groupkey, check_categorical, check_numeric]

        if not all(check_lists):
            # self.save_error('experimental_plan.yaml에서 칼럼과 관련된 옵션들을 다시 검토하세요')
            self.save_error('Please review the column-related options (x_columns / y_columns / groupkey_columns / categorical_columns / numeric_columns) in experimental_plan.yaml again.')

        pass


    def convert_column_names(self, config, input_data):
        '''24/03/28: ui_args에 리스트 형식으로 넣은 데이터는 list int가 아닌 list str로 들어감
           1. input_data의 모든 칼럼 명을 문자열로 변환 (90 -> '90')
           2. x_columns, y_column, groupkey_columns, drop_x_columns에 지정된 정수형 칼럼명도 문자열로 변환'''
        
        input_data.columns = input_data.columns.astype(str)

        if config['readiness']['x_columns']:
            config['readiness']['x_columns'] = [str(col) for col in config['readiness']['x_columns']]

        if config['readiness']['y_column']:
            config['readiness']['y_column'] = str(config['readiness']['y_column'])

        if config['readiness']['groupkey_columns']:
            config['readiness']['groupkey_columns'] = [str(col) for col in config['readiness']['groupkey_columns']]

        if config['readiness']['drop_x_columns']:
            config['readiness']['drop_x_columns'] = [str(col) for col in config['readiness']['drop_x_columns']]

        return config, input_data


    def check_groupkey(self, config, input_data, pipeline):
    # config, input_data = self.check_groupkey(config, input_data, 'train')
        '''
        groupkey 컬럼을 조합하여 groupkey list를 생성(groupkey 컬럼 생성)
        - groupkey가 1개일때는 groupkey 컬럼 생성x, groupkey가 2개 이상일 때는 groupkey를 조합한 groupkey 컬럼을 생성
        - config['readiness']['groupkey_columns']에 새로 생성한 groupkey 컬럼으로 업데이트
        - config['readiness']['groupkey_list']에 모든 groupkey 조합 list를 입력
        '''

        def get_unique_groupkey_list(config, input_data):
            '''groupkey_columns = groupkey_columns_split 
               주어진 그룹 키 칼럼(들)로부터 조합 가능한 모든 경우의 수 계산 & config에 저장
               ex1) unique(Sex)=[F, M] & unique(Pclass)=[1, 2, 3] -> [F_1, F_2, ..., M_3]    
               ex2) unique(Sex)=[F, M] -> [F, M]         
            '''
            groupkey_columns = config['readiness']['original_groupkey_columns']

            # groupkey 여러 개
            if len(groupkey_columns) > 1:  
                # 데이터프레임의 각 인스턴스 마다 groupkey 칼럼 조합을 계산 & new_col에 저장
                new_col = config['readiness']['groupkey_columns'][0]    
                for idx, group in enumerate(groupkey_columns):
                    df_group = input_data[group].astype(str)
                    if idx == 0:
                        input_data[new_col] = df_group
                    else:
                        input_data[new_col] =  input_data[new_col] + '_' + df_group

                unique_groupkey_list = list(np.unique(input_data[new_col]))
                config['readiness']['groupkey_list'] = unique_groupkey_list
            
            # groupkey 하나 (새로운 칼럼 만들 필요 X)
            elif len(groupkey_columns):
                unique_groupkey_list = list(np.unique(input_data[groupkey_columns[0]]))

                # 범주형 변수 역할을 수행하는 데 실질적인 값들은 정수/실수형일때
                # ex) titanic의 Pclass는 범주형 변수이지만(좌석 등급) 실질적인 값들은 1,2,3 정수형
                if not all(isinstance(item, str) for item in unique_groupkey_list):
                    unique_groupkey_list = [str(value) for value in unique_groupkey_list]
                    input_data[groupkey_columns] = input_data[groupkey_columns].astype(str)

                config['readiness']['groupkey_list'] = unique_groupkey_list

            return config, input_data


        if pipeline == 'train':
            groupkey_columns_split = config['readiness']['groupkey_columns']  # ['Pclass', 'Gender]  
            groupkey_columns_split = sorted(groupkey_columns_split)  # ['Gender', 'Pclass']
            config['readiness']['original_groupkey_columns'] = groupkey_columns_split  # ['Gender', 'Pclass']  

            groupkey_columns_integrated = '_'.join(groupkey_columns_split)  # ['Gender_Pclass']  
            config['readiness']['groupkey_columns'] = [groupkey_columns_integrated]  # ['Gender_Pclass']   
            if len(groupkey_columns_split) == 0:
                config['readiness']['groupkey_columns'] = []

            # config에 groupkey_list 업데이트 & 데이터프레임 업데이트(groupkey가 여러 개 일때만)
            config, input_data = get_unique_groupkey_list(config=config, input_data=input_data)
            config['readiness']['categorical_columns_unique'] = {groupkey:{} for groupkey in config['readiness']['groupkey_list']}

            
        elif pipeline == 'inference':
            groupkey_list_train = config['readiness']['groupkey_list']  # train시 저장되어 있는 groupkey_list 불러옴
            
            # config에 groupkey_list 업데이트 & 데이터프레임 업데이트(groupkey가 여러 개 일때만)
            config, input_data = get_unique_groupkey_list(config=config, input_data=input_data)
            groupkey_list_inference = config['readiness']['groupkey_list']

            # train에는 등장하지 않았던 그룹 키 조합이지만, inference 시에는 등장한 그룹 키 조합
            only_inference = list(set(groupkey_list_inference) - set(groupkey_list_train))
            
            # train에는 등장하지 않았던 그룹 키 조합이지만, inference 시에는 등장한 그룹 키 조합을 배제시킴
            groupkey_list_inference = [element for element in groupkey_list_inference if element not in only_inference]
            config['readiness']['groupkey_list'] = groupkey_list_inference
        
        return config, input_data


    def check_min_rows(self, config, input_data):

        '''
        학습에 필요한 최소 데이터 수를 계산
        - task_type보고 default 값으로 config['readiness']['min_rows'] 업데이트
        - classification) y_column 라벨 별 데이터 수 20개 안되면 warning(error 고민)
            - y_column 라벨은 config['readiness']['categorical_columns_unique'][y_column 명]안에 있음
        - regression) y_column 데이터 수 50개 안되면 waring(error 고민)
        '''

        if config['readiness']['min_rows'] == -1:
            if config['readiness']['task_type'] == 'classification':
                min_ths = 30 ; config['readiness']['min_rows'] = min_ths
            elif config['readiness']['task_type'] == 'regression':
                min_ths = 100 ; config['readiness']['min_rows'] = min_ths
        else:
            min_ths = config['readiness']['min_rows']
        
        y_column = config['readiness']['y_column']

        # groupkey O
        if len(config['readiness']['original_groupkey_columns']) > 0:
            new_groupkey_list = []
            groupkey_column= config['readiness']['groupkey_columns'][0]
            groupkey_list = config['readiness']['groupkey_list']

            for group in groupkey_list:
                df_group = input_data[input_data[groupkey_column] == group]

                # 각 그룹 별 데이터의 타깃 칼럼의 예측 클래스 마다 min_ths 이상의 데이터가 있어야 함
                if config['readiness']['task_type'] == 'classification':  
                    y_values = list(df_group[y_column].value_counts())
                    check_y =  all(y_val > min_ths for y_val in y_values)   

                    if check_y:
                        # self.save_info(f'The number of data per target class belonging to {group} exceeds {min_ths}. (It is included in the groupkey_list)')
                        new_groupkey_list.append(group)
                    else:
                        self.save_info(f'The number of data per target class belonging to {groupkey_column}-{group} is less than {min_ths}. (It is excluded from the groupkey_list)')

                # 전체 데이터가 min_ths 이상 있어야 함
                elif config['readiness']['task_type'] == 'regression':
                    check_y = (len(df_group) > min_ths)

                    if check_y:
                        # self.save_info(f'{group}에 속하는 데이터의 수가 {min_ths}를 초과합니다.')
                        # self.save_info(f'The number of data belonging to {group} exceeds {min_ths}. (It is included in the groupkey_list)')
                        new_groupkey_list.append(group)
                    else:
                        # self.save_info(f'{group}에 속하는 데이터의 수가 {min_ths}보다 적습니다.')
                        self.save_info(f'The number of data belonging to {groupkey_column}-{group} is less than {min_ths}. (It is excluded from the groupkey_list)')
            
            config['readiness']['groupkey_list'] = new_groupkey_list

            if len(new_groupkey_list) == 0 and config['readiness']['task_type'] == 'regression':
                # self.save_error(f'모든 group별 데이터의 수가 min_ths-{min_ths} 미만입니다. groupkey 설정을 다시 고려해주세요')
                self.save_error(f'The number of data for each group is less than min_rows={min_ths}. There are no groups available for training, so the process is terminated.')

            elif len(new_groupkey_list) == 0 and config['readiness']['task_type'] == 'classification':
                # self.save_error(f'모든 group별 데이터의 수가 min_ths-{min_ths} 미만입니다. groupkey 설정을 다시 고려해주세요')
                self.save_error(f'The number of data per target class for each group is less than min_rows={min_ths}. There are no groups available for training, so the process is terminated.')

            return config

        # groupkey X
        else:
            if config['readiness']['task_type'] == 'classification':
                y_values = list(input_data[y_column].value_counts())
                check_y =  all(y_val > min_ths for y_val in y_values)     

                if check_y:
                    # self.save_info(f'입력 데이터프레임의 데이터의 수가 {min_ths}를 초과합니다.')
                    # self.save_info(f'The number of data per class exceeds {min_ths}.')
                    return config
                else:
                    # self.save_info(f'입력 데이터프레임의 데이터의 수가 {min_ths}보다 적습니다.')
                    self.save_error(f'The number of data per class is less than {min_ths}. Please modify the min_rows option or secure more data.')
                    return config

            elif config['readiness']['task_type'] == 'regression':
                check_y = (len(input_data) > min_ths)

                if check_y:
                    # self.save_info(f'입력 데이터프레임의 데이터의 수가 {min_ths}를 초과합니다.')
                    # self.save_info(f'The number of data in the input dataframe exceeds {min_ths}.')
                    return config
                else:
                    # self.save_info(f'입력 데이터프레임의 데이터의 수가 {min_ths}보다 적습니다.')
                    self.save_error(f'The number of data in the input dataframe is less than {min_ths}. Please modify the min_rows option or secure more data.')
                    return config
                
            
    def check_column_types(self, check_columns, config, input_data):
        '''
            check_columns 컬럼의 categorical/numeric 여부를 검사합니다.
            - 컬럼 dtype이 numeric -> numeric
            - 컬럼 dtype이 object & top N numeric -> numeric
            - 컬럼 dtype이 object & top N object & cardinality 조건 만족 -> categorical
            - 컬럼 dtype이 object & top N object & cardinality 조건 만족X -> 학습에서 제외 
        '''
        tmp_total, tmp_numeric_columns, tmp_categorical_columns = [] ,[], []
        for col in check_columns:
            if col in config['readiness']['categorical_columns']:
                tmp_categorical_columns, tmp_total, config = self.categorical_update(tmp_categorical_columns, tmp_total, col, input_data, config)
                continue
            elif col in config['readiness']['numeric_columns']:
                tmp_numeric_columns, tmp_total, config = self.numeric_update(tmp_numeric_columns,tmp_total, col, config)
                continue

            if input_data[col].dtype == object:
                try: # input_data[col]의 top10을 numeric으로 변환
                    object_col_unique = input_data[col].value_counts().index
                    object_col_unique[:config['readiness']['num_cat_split']].astype(np.float64)
                    # 에러 나지 않으면 numeric 컬럼으로 분류
                    tmp_numeric_columns, tmp_total, config = self.numeric_update(tmp_numeric_columns, tmp_total, col, config)
                except Exception:
                    # cardinality 조건을 만족하면 categorical 컬럼으로 분류
                    if len(object_col_unique) <= config['readiness']['cardinality']:
                        tmp_categorical_columns, tmp_total, config = self.categorical_update(tmp_categorical_columns, tmp_total, col, input_data, config)
                    else:
                        # self.save_warning(f'컬럼 {col}의 unique 데이터 수가 {config["readiness"]["cardinality"]}를 넘어 x_columns에서 제외합니다.')
                        self.save_warning(f'The number of unique data in column {col} exceeds {config["readiness"]["cardinality"]}, so it is excluded from x_columns.')
            elif input_data[col].dtype == bool:
                tmp_categorical_columns, tmp_total, config = self.categorical_update(tmp_categorical_columns, tmp_total, col, input_data, config)
            else: 
                # numeric 컬럼으로 분류
                tmp_numeric_columns, tmp_total, config = self.numeric_update(tmp_numeric_columns, tmp_total, col, config)

        return tmp_total, tmp_numeric_columns, tmp_categorical_columns, config



    def categorical_update(self, tmp_cat_cols, tmp_tot, col, input_data, config):
        '''
            total_columns(tmp_tot)와 categorical_columns(tmp_cat_cols)에 col 추가
            - categorical 컬럼의 경우 config['readiness']['categorical_columns_unique']에 컬럼 별 unique list를 추가해야함
            - groupkey 없을 때: config['readiness']['categorical_columns_unique'] = {colA:[colA unique list], colB: [colB unique list],...}
            - groupkey 있을 때: config['readiness']['categorical_columns_unique'] = {groupkey1: {colA:[colA unique list], colB: [colB unique list],...}, groupkey2: {...}, ...}
        '''
        tmp_cat_cols.append(col), tmp_tot.append(col)
        # groupkey_column, groupkey_list = config['readiness']['groupkey_columns'][0], config['readiness']['groupkey_list']  
        if config['readiness']['groupkey_columns']:
            groupkey_column =  config['readiness']['groupkey_columns'][0]
        else: 
            groupkey_column = []
        groupkey_list = config['readiness']['groupkey_list']  

        if len(groupkey_column) > 0: # {groupkey1: {colA:[], colB:[],...}, gruopkey2: {}}
            for groupkey in groupkey_list:
                # config['readiness']['categorical_columns_unique'][groupkey][col] = input_data[input_data[groupkey_column[0]] == groupkey][col].dropna().unique().tolist() 
                config['readiness']['categorical_columns_unique'][groupkey][col] = input_data[input_data[groupkey_column] == groupkey][col].dropna().unique().tolist() 
        else: 
            config['readiness']['categorical_columns_unique'][col] = input_data[col].dropna().unique().tolist()
        # self.save_info(f'{col} 컬럼이 categorical 컬럼으로 분류됩니다.')
        self.save_info(f'The {col} column is classified as a categorical column.')

        return tmp_cat_cols, tmp_tot, config
    
    def numeric_update(self, tmp_num_cols, tmp_tot, col, config):
        '''
            total_columns(tmp_tot)와 numeric_columns(tmp_num_cols)에 col 추가
        '''
        tmp_num_cols.append(col), tmp_tot.append(col)
        # self.save_info(f'{col} 컬럼이 numeric 컬럼으로 분류됩니다.')
        self.save_info(f'The {col} column is classified as a numeric column.')
        return tmp_num_cols, tmp_tot, config

    def check_x_NA(self, config, input_data):
        '''
            독립 변수로 지정된 칼럼들 중 모든 값이 결측치인 칼럼이 있으면 학습 대상에서 제외
            사용자가 입력한 numeric_columns의 경우(추후 자동 float으로 변환) change_to_num()시 null인지 체크 필요!  
        '''
        def change_to_num(x):
            try: return np.float64(x)
            except: return np.nan
        def get_all_NA_cols(input_data):
            all_NA_cols = []
            for x_col in config['readiness']['x_columns']:        
                if x_col in config['readiness']['numeric_columns']:
                    check_all_NA = input_data[x_col].apply(change_to_num).isnull().all()            
                else:
                    check_all_NA = input_data[x_col].isnull().all()
                if check_all_NA:
                    all_NA_cols.append(x_col)
            return all_NA_cols

        # dataframe 전체에 대해 NA있는지 검사
        all_NA_cols = get_all_NA_cols(input_data)
        if len(all_NA_cols) > 0:
            # self.save_warning(f'{all_NA_cols} 칼럼이 모두 결측치로 구성되어 있습니다. x_columns에서 제외됩니다.')
            self.save_warning(f'The {all_NA_cols} column is entirely composed of missing values. It will be excluded from x_columns.')
            config['readiness']['x_columns'] = [col for col in config['readiness']['x_columns'] if col not in all_NA_cols]
            config['readiness']['categorical_columns'] = [col for col in config['readiness']['categorical_columns'] if col not in all_NA_cols]
            config['readiness']['numeric_columns'] = [col for col in config['readiness']['numeric_columns'] if col not in all_NA_cols]

        # groupkey에 대해서 검사
        if len(config['readiness']['groupkey_list']) > 0:  
            config['readiness']['groupkey_fill_columns'] = {}    
            groupkey_columns, groupkey_list = config['readiness']['groupkey_columns'][0], config['readiness']['groupkey_list']
            for group in groupkey_list:
                df_group = input_data[input_data[groupkey_columns] == group]
                all_NA_cols = get_all_NA_cols(df_group)
                if len(all_NA_cols) > 0:
                    self.save_warning(f'In group {group}, the {all_NA_cols} column(s) is composed of all missing values. It will be filled with dummy values.')
                    config['readiness']['groupkey_fill_columns'][group] = all_NA_cols
        return config

         
    def check_y_group(self, config, input_data):
        y_column, groupkey_columns, groupkey_list = config['readiness']['y_column'], config['readiness']['groupkey_columns'][0], config['readiness']['groupkey_list']

        new_groupkey_list = []
        total_y_unique = list(input_data[y_column].unique())
        for group in groupkey_list:
            data_group = input_data[input_data[groupkey_columns] == group]
            group_y_unique = list(data_group[y_column].unique())
    
            if total_y_unique.sort() == group_y_unique.sort():
                new_groupkey_list.append(group)
            else:
                # self.save_error(f'{group}에 해당하는 데이터의 y칼럼의 unique값이 {group_y_unique}로 전체 데이터의 y칼럼의 unique값인 {total_y_unique}와 다릅니다')
                self.save_error(f'The unique values of the y_column for the data corresponding to {group} are {group_y_unique}, which is different from the unique values of the y_column for the entire data, which are {total_y_unique}.')

        config['readiness']['groupkey_list'] = new_groupkey_list
        return config 

    def check_y_unique(self, config, input_data):
        '''
            학습 시 y_column의 unique 값이 1인지 검사합니다.
            - groupkey가 없을 때는 y_column의 unique 종류가 1이면 error
            - groupkey가 있을 때는 y_coulmn의 unique 종류가 1이 아닌 groupkey만 학습시킴. 모든 groupkey에서 y_column의 unique 수가 1이면 error 발생
        '''
        y_column, groupkey_columns, groupkey_list = config['readiness']['y_column'], config['readiness']['groupkey_columns'], config['readiness']['groupkey_list']
        config['readiness']['num_classes'] = len(input_data[y_column].unique())
        if len(groupkey_columns) > 0: # y_column_1_groupkey = y_column의 unique 수가 1인 groupkey list
            y_column_1_groupkey = [groupkey for groupkey in groupkey_list if len(config['readiness']['categorical_columns_unique'][groupkey][y_column]) == 1]
            if len(y_column_1_groupkey) > 0: 
                if len(y_column_1_groupkey) == len(groupkey_list):
                    # self.save_error('모든 그룹키의 y_column 값이 1개로 구성되어 있습니다. 학습을 위해서는 y_column 값의 종류가 2개 이상이어야 합니다. process를 종료합니다.')
                    self.save_error('All group keys are composed of a single y_column value. For learning, there must be at least two types of y_column values. The process will be terminated.')
                else:
                    # prt = f'train 데이터 그룹키: {y_column_1_groupkey}의 y_column 값이 1개 입니다. 해당 groupkey는 학습 대상에서 제외합니다.'
                    prt = f'The y_column value of the train data group key: {y_column_1_groupkey} is 1. This group key is excluded from the learning target.'
                    self.save_warning(prt)
                    config['readiness']['groupkey_list'] = [groupkey for groupkey in config['readiness']['groupkey_list'] if groupkey not in y_column_1_groupkey]
        else:
            if len(config['readiness']['categorical_columns_unique'][y_column]) == 1:
                # self.save_error('y_column 값이 1개로 구성되어 있습니다. 학습을 위해서는 y_column 값의 종류가 2개 이상이어야 합니다. process를 종료합니다.')
                self.save_error('The y_column value is composed of a single value. For training the classification model, there must be at least two types of y_column values. The process will be terminated.')
        return config

    def set_target_label(self, config, input_data):
        def most_common_element(lst):
            counter = Counter(lst)
            most_common = counter.most_common(1)
            return most_common[0][0]if most_common else None
      
        def least_common_element(lst):
            counter = Counter(lst)
            least_common = counter.most_common()[:-2:-1]
            return least_common[0][0]if least_common else None

        target_label = config['readiness']['target_label']

        ## 태스크 정의 (binary/multi)
        y_col = config['readiness']['y_column']
        y_unique = list(input_data[y_col].unique())
        y_values = list(input_data[y_col])
        if len(y_unique) == 2:
            config['readiness']['classification_type'] = 'binary'
        elif len(y_unique) >= 3:
            config['readiness']['classification_type'] = 'multi'
        classification_type = config['readiness']['classification_type'] 

        ## pos_label(binary) / target_label(multi) 지정
        # 1. _major (default 옵션) : bianry/multi 둘 다 가능
        if target_label == '_major':
            config['readiness']['target_label'] =  [most_common_element(y_values)]
        
        # 2. _minor : bianry/multi 둘 다 가능
        elif target_label == '_minor':
            config['readiness']['target_label'] =  [least_common_element(y_values)]

        # 3. _all : multi만 가능
        elif target_label == '_all':
            if classification_type == 'binary':
                # self.save_error('Binary classification에서는 사용할 수 없는 target_label 옵션입니다')
                self.save_error('This is a target_label option that cannot be used in binary classification.')
            config['readiness']['target_label'] =  y_unique
        
        # 6. 유저 세팅 (단일 클래스 명칭 - classA & 정수타입) : bianry/multi 둘 다 가능
        elif type(target_label) == int:
            config['readiness']['target_label'] =  [target_label]

        # 4. 유저 세팅 (단일 클래스 리스트 - [classA]) : bianry/multi 둘 다 가능
        elif len(target_label) == 1 and type(target_label) == list:
            config['readiness']['target_label'] =  target_label
        
        # 5. 유저 세팅 (다중 클래스 리스트 - [classA, classB, ....]) : multi만 가능
        elif len(target_label) > 1 and type(target_label) == list:
            if classification_type == 'binary':
                # self.save_error('Binary classification에서는 사용할 수 없는 target_label 옵션입니다')
                self.save_error("This is a target_label option that cannot be used in binary classification.")
            config['readiness']['target_label'] =  target_label
        
        # 6. 유저 세팅 (단일 클래스 명칭 - classA & 문자열) : bianry/multi 둘 다 가능
        else:
            config['readiness']['target_label'] =  [target_label]

        return config

    def check_inference_new_category(self, config, input_data, rule):
        '''
            inference시 categorical 컬럼(target_columns)에 학습에 쓰이지 않은 category 값이 들어올 경우 처리합니다. 
            - rule(ignore_new_category)이 False 일 때: 
                - groupkey가 없는 경우: error 처리
                - groupkey가 있는 경우: groupkey list에서 제거
            - rule이 True일 때:
                - 삭제할 row index 기록
            - rule이 float type일 때:
                - 전체 컬럼에서 새 category 데이터가 차지하는 비율 > rule 일 때:
                    - groupkey가 없는 경우: error 처리
                    - groupkey가 있는 경우: groupkey list에서 제거
                - 전체 컬럼에서 새 category 데이터가 차지하는 비율 <= rule 일 때:
                    - 삭제할 row index 기록
        '''

        # def store_indices(config, input_data):
        #     '''rule=True용, train에는 없었던 category 값이 등장했을 때
        #        해당 데이터의 인덱스를 칼럼 별로 (딕셔너리 형태로) 저장'''

        #     target_columns = config['readiness']['categorical_columns']  # 학습에 사용된 범주형 변수
        #     if len(config['readiness']['groupkey_columns']) == 0:
        #         known_unique_values = config['readiness']['categorical_columns_unique']  # 범주형 칼럼 별 unique값 (dictionary) 
        #     else:
        #         group = list(config['readiness'].keys())[0]  # group마다 동일한 encoding 방식을 적용한다고 가정
        #         categorical_encoding = config['readiness'][group]['categorical_columns_unique']
        #     indices_dict = {column: [] for column in target_columns}  # target_columns에 지정된 칼럼 별로 새로운 범주형 값이 등장하는 데이터의 index 저장

        #     for column in target_columns:
        #         if column in input_data.columns:
        #             for idx, cat_value in input_data[[column]].iterrows():
        #                 if cat_value[column] not in known_unique_values[column]:
        #                     # df.at[idx, column] = np.nan
        #                     indices_dict[column].append(idx)

        #     config['readiness']['new_category_row_index'] = indices_dict

        #     return config


        def get_new_category_indexes(config, input_data): 
            '''rule=True인 case용 (groupkey 있는 경우와 없는 경우 포함)
               train에는 없었던 category 값이 등장한 데이터 인스턴스의 index를 리스트로 저장'''

            target_columns = config['readiness']['categorical_columns']
            
            # groupkey O
            if len(config['readiness']['groupkey_columns']) > 0:
                groupkey_column, groupkey_list = config['readiness']['groupkey_columns'][0], config['readiness']['groupkey_list']
                # new_cat_idx_list: input_data에서 제거할 (train에는 등장하지 않지만 inference에서 새로 등장한 인스턴스의) index저장
                #               groupkey_list의 groupkey로 분할된 input_data는 서로 중복되지 않으므로 (groupkey구별 없이) index는 한꺼번에 저장
                # new_cat_idx_list = []  
                config['readiness']['new_category_row_index'] = {}
                for cat_col in target_columns:
                    config['readiness']['new_category_row_index'][cat_col] = []

                for groupkey in groupkey_list:  # 각 그룹 별 데이터마다
                    new_cat_col = []  # 새로운 범주형 변수값이 등장하는 범주형 칼럼명 저장 (그룹별로)
                    input_group = input_data[input_data[groupkey_column] == groupkey]  
                    for cat_col in target_columns:  # category 칼럼들을 순회 
                        # train data에 사용된 특정 cateogry 칼럼의 unique 값들
                        cat_unique_train = config['readiness']['categorical_columns_unique'][groupkey][cat_col]  
                        # inference data에 사용된 특정 cateogry 칼럼의 unique 값들
                        cat_unique_inference = input_group[cat_col].dropna().unique()
                        # inference data의 cat_col 칼럼에 새롭게 등장한 unique 값들
                        new_cat_values = list(set(cat_unique_inference) - set(cat_unique_train)) 

                        if len(new_cat_values) > 0:
                            for new_cat in new_cat_values:
                                input_new_cat = (input_group[[cat_col]] == new_cat)      
                                input_new_cat_idx = list(input_new_cat[input_new_cat[cat_col] == True].index)  # train데이터에는 사용되지 않은 범주형 변수 값이 나와 삭제해야 하는 행의 index
                                config['readiness']['new_category_row_index'][cat_col] += input_new_cat_idx
                                self.save_warning(f'In {groupkey} Data : New categorical value(s) that did not used in training step appear in column: {cat_col}.')

            # groupkey X
            else:
                config['readiness']['new_category_row_index'] = {}  # input_data에서 제거할 (train에는 등장하지 않지만 inference에서 새로 등장한 인스턴스의) index저장
                for cat_col in target_columns:  # category 칼럼들을 순회 
                    cat_unique_train = config['readiness']['categorical_columns_unique'][cat_col]
                    cat_unique_inference = input_data[cat_col].dropna().unique()
                    new_cat_list = list(set(cat_unique_inference) - set(cat_unique_train)) 

                    if len(new_cat_list) > 0:
                        config['readiness']['new_category_row_index'][cat_col] = []
                        self.save_warning(f'New categorical value(s) that did not used in training step appear in column: {cat_col}.')
                        for new_cat in new_cat_list:
                            input_new_cat = (input_data[[cat_col]] == new_cat)       
                            # train데이터에는 사용되지 않은 범주형 변수 값이 나와 삭제해야 하는 행의 index
                            input_new_cat_idx = list(input_new_cat[input_new_cat[cat_col] == True].index)  
                            config['readiness']['new_category_row_index'][cat_col] += input_new_cat_idx

            return config


        def remove_element_or_return_error(config, input_data):
            '''rule=False인 case용, train에는 없었던 category 값이 등장했을 때
               groupkey가 있으면 해당 groupkey 요소를 groupkey_list에서 제거
               gorupkeyrk 없으면 error 메세지를 출력'''

            target_columns = config['readiness']['categorical_columns']

            # groupkey O
            if len(config['readiness']['groupkey_columns']) > 0:
            # if is_groupkey:
                groupkey_column, groupkey_list = config['readiness']['groupkey_columns'][0], config['readiness']['groupkey_list']
                del_groupkey_list = []  

                for groupkey in groupkey_list:  # 각 그룹 별 데이터마다
                    new_cat_col = []
                    input_group = input_data[input_data[groupkey_column] == groupkey]
                    for cat_col in target_columns:  # category 칼럼들을 순회 
                        cat_unique_train = config['readiness']['categorical_columns_unique'][groupkey][cat_col]
                        cat_unique_inference = input_group[cat_col].dropna().unique()
                        new_cat_list = list(set(cat_unique_inference) - set(cat_unique_train)) 
        
                        if len(new_cat_list) > 0:
                            new_cat_col.append(cat_col)
                            del_groupkey_list.append(groupkey)

                    if len(new_cat_col) > 0:
                        self.save_warning(f'New categorical values that did not appear in the training data are appearing in {new_cat_col} columns of the inference data belonging to group {groupkey}. Please check the categorical variables in the data.')

                del_groupkey_list = list(set(del_groupkey_list))
                if len(del_groupkey_list) > 0:
                    # self.save_info(f'groupkey_list에서 {del_groupkey_list}가 삭제되어야 함 (학습 데이터에는 등장하지 않은 새로운 category 값 등장)')
                    self.save_warning(f'The {del_groupkey_list} groups should be removed from the groupkey_list as new category values that did not appear in the training data have emerged.')
                
                groupkey_list = config['readiness']['groupkey_list']
                config['readiness']['groupkey_list'] = list(set(groupkey_list) - set(del_groupkey_list))
                if len(config['readiness']['groupkey_list']) == 0:
                    self.save_error(f'The inference process is terminated due to the absence of available group keys. Categorical values that were not used in the train data appear in the categorical variable column of all group key data. Please try changing the rule option.')

                return config
            
            # groupkey X
            else:
                error_col_list = []
                for cat_col in target_columns:
                    cat_unique_train = config['readiness']['categorical_columns_unique'][cat_col]
                    cat_unique_inference = input_data[cat_col].dropna().unique()
                    new_cat_list = list(set(cat_unique_inference) - set(cat_unique_train)) 
    
                    if len(new_cat_list) > 0:
                        error_col_list.append(cat_col)

                if len(error_col_list) != 0:
                    # self.save_error(f'테스트용 데이터의 {error_col_list} 칼럼에  학습 데이터에는 존재하지 않는 범주형 값들이 등장합니다. 다음 방법 중 하나를 고려해주세요. 1. 해당 칼럼들을 readiness의 x_columns/categorical_columns옵션에서 제외 2. 새로 등장한 값을 포함하여 재 학습 진행 3. ignore_new_category: True를 experimental_plan.yaml파일의 readiness asset에 추가하여 해당 이슈를 무시')
                    self.save_error(f'The {error_col_list} column in the test data contains categorical values that do not exist in the training data. Please consider one of the following methods: 1. Exclude these columns from the x_columns/categorical_columns option of readiness. 2. Re-train including the newly emerged values. 3. Add ignore_new_category: True to the readiness asset in the experimental_plan.yaml file to ignore this issue.')
                return config


        def consider_ratio(config, input_data, ths_ratio):
            '''rule=float_type용'''

            target_columns = config['readiness']['categorical_columns']

            # groupkey O
            if len(config['readiness']['groupkey_columns']) > 0:
                groupkey_column, groupkey_list = config['readiness']['groupkey_columns'][0], config['readiness']['groupkey_list']
                
                del_groupkey_list = []
                del_idx_list = []

                for groupkey in groupkey_list: 
                    new_cat_col = []
                    input_group = input_data[input_data[groupkey_column] == groupkey]
                    for cat_col in target_columns:  # category 칼럼들을 순회
                        # train data에 사용된 특정 cateogry 칼럼의 unique 값들
                        cat_unique_train = config['readiness']['categorical_columns_unique'][groupkey][cat_col]  
                        # inference data에 사용된 특정 cateogry 칼럼의 unique 값들
                        cat_unique_inference = input_group[cat_col].dropna().unique()
                        # inference data의 cat_col 칼럼에 새롭게 등장한 unique 값들
                        new_cat_list = list(set(cat_unique_inference) - set(cat_unique_train))
                        
                        if len(new_cat_list) > 0:
                            new_cat_col.append(cat_col)
                            for new_cat in new_cat_list:
                                input_new_cat = (input_group[[cat_col]] == new_cat)   
                                ratio = input_new_cat[cat_col].sum() / len(input_new_cat)

                                if ratio > ths_ratio:
                                    del_groupkey_list = []
                                    del_groupkey_list.append(groupkey)
                                else:
                                    input_new_cat_idx = list(input_new_cat[input_new_cat[cat_col] == True].index)
                                    del_idx_list += input_new_cat_idx

                    if len(new_cat_col) > 0:
                        # self.save_info(f'{groupkey} groupkey를 groupkey_list에서 삭제')
                        self.save_warning(f'New categorical values that did not appear in the training data are appearing in {new_cat_col} columns of the inference data belonging to group {groupkey}. Please check the categorical variables in the data.')
                        self.save_warning(f'The {groupkey} groupkey has been removed from the groupkey_list. It will be exclueded from the inference process')

                if len(del_idx_list) > 0:
                    config['readiness']['del_idx_list'] += del_idx_list
                    config['readiness']['del_idx_list'] = list(set(config['readiness']['del_idx_list']))
                    # self.save_info(f'{groupkey}데이터에서 총 {len(del_idx_list)}개의 데이터가 삭제됨')
                    self.save_warning(f'A total of {len(del_idx_list)} data has been deleted from the total data.')

                groupkey_list = config['readiness']['groupkey_list']
                config['readiness']['groupkey_list'] = list(set(groupkey_list) - set(del_groupkey_list))
                if len(config['readiness']['groupkey_list']) == 0:
                    self.save_error(f'The inference process is terminated due to the absence of available group keys. Categorical values that were not used in the train data appear in the categorical variable column of all group key data. Please try changing the rule option.')
                    
                if len(error_col_list) != 0:
                    # self.save_error(f'테스트용 데이터의 {error_col_list} 칼럼에  학습 데이터에는 존재하지 않는 범주형 값들이 등장합니다. 다음 방법 중 하나를 고려해주세요. 1. 해당 칼럼들을 readiness의 x_columns/categorical_columns옵션에서 제외 2. 새로 등장한 값을 포함하여 재 학습 진행 3. ignore_new_category: True를 experimental_plan.yaml파일의 readiness asset에 추가하여 해당 이슈를 무시')
                    self.save_error(f'The {error_col_list} column in the test data contains categorical values that do not exist in the training data. Please consider one of the following methods: 1. Exclude these columns from the x_columns/categorical_columns option of readiness. 2. Re-train including the newly emerged values. 3. Add ignore_new_category: True to the readiness asset in the experimental_plan.yaml file to ignore this issue.')

            # groupkey X
            else:
                del_idx_list = []
                for cat_col in target_columns:  # category 칼럼들을 순회 
                    cat_unique_train = config['readiness']['categorical_columns_unique'][cat_col]
                    cat_unique_inference = input_data[cat_col].dropna().unique()
                    new_cat_list = list(set(cat_unique_inference) - set(cat_unique_train)) 
    
                    for new_cat in new_cat_list:
                        input_new_cat = (input_data[[cat_col]] == new_cat)   
                        ratio = input_new_cat[cat_col].sum() / len(input_new_cat)

                        if ratio > ths_ratio:
                            self.save_error(f'The ratio of new category values appearing in the categorical variable {cat_col} that did not appear in the training data exceeds the rule={ths_ratio} (it is {round(ratio, 3)}). The process will be terminated')
                        else:
                            input_new_cat_idx = list(input_new_cat[input_new_cat[cat_col] == True].index)
                            del_idx_list += input_new_cat_idx

                if len(del_idx_list) > 0:
                    config['readiness']['del_idx_list'] += del_idx_list
                    config['readiness']['del_idx_list'] = list(set(config['readiness']['del_idx_list']))
                    self.save_warning(f'A total of {len(del_idx_list)} data points have been deleted from the input data.')

            if len(config['readiness']['del_idx_list']) == input_data.shape[0]:
                self.save_error('There is no data available to proceed with the inference. The process will be terminated.')

            return config


        if rule == True:  
            config = get_new_category_indexes(config, input_data)
            # config = store_indices(config, input_data)
        
        elif rule == False:
            config = remove_element_or_return_error(config, input_data)

        else:
            config = consider_ratio(config, input_data, rule)

        return config


    def save_info(self, msg):
        # self.logger_method_dict['info'](msg)
        self.logger_method_dict.info(msg)
     
    def save_warning(self, msg):
        # self.logger_method_dict['warning'](msg)
        self.logger_method_dict.warning(msg)
 
    def save_error(self, msg):
        # self.logger_method_dict['error'](msg)
        self.logger_method_dict.error(msg)


# asset.readiness.py에서 그대로 복사
def check_configs(config, logger):
    '''
    config의 조합을 검사하는 함수로 def setting_configs에서 호출해서 사용합니다.
    - x_columns와 groupkey 컬럼이 겹치면 error
    - x_columns에 y_column이 포함되지 않도록
    - x_columns와 drop_x_columns이 동시에 사용되지 않도록 함
    '''
    x_columns = config['readiness']['x_columns']
    if len(set(x_columns).intersection(set(config['readiness']['groupkey_columns']))) > 0:
        logger.debug('x_columns에 groupkey_columns의 컬럼이 포함되었습니다.')
    if config['readiness']['y_column'] in x_columns:
        logger.debug('x_columns에 y_column이 포함되었습니다.')
    if (len(x_columns)>0) and (len(config['readiness']['drop_x_columns'])>0):
        logger.debug('x_columns 와 drop_x_columns 중 하나만 사용해야 합니다.')

# asset.readiness.py에서 그대로 복사
def revise_config(config, input_col_list):
    '''
        config를 가공하는 함수로 def setting_configs에서 호출해서 사용합니다.
        - drop_x_columns 사용시, x_columns 지정
        - column_types가 auto가 아닐 때 categorical/numeric_columns 지정
    '''
    drop_x_columns, groupkey_columns = config['readiness']['drop_x_columns'], config['readiness']['groupkey_columns']
    if not config['readiness']['x_columns']:
        new_columns = []
        for col in input_col_list:
            if col == config['readiness']['y_column']: continue
            if len(groupkey_columns) > 0:
                if col in groupkey_columns: continue
            if col in drop_x_columns: continue
            new_columns.append(col)
        config['readiness']['x_columns'] = new_columns
        config['readiness']['drop_x_columns'] = []

    column_types = config['readiness']['column_types']
    if column_types != 'auto':
        if 'categorical_columns' in column_types:
            config['readiness']['categorical_columns'] = column_types['categorical_columns']
        if 'numeric_columns' in column_types:
            config['readiness']['numeric_columns'] = column_types['numeric_columns']

    return config


# asset.readiness.py에서 약간 수정
def setting_configs(config, args, input_data, pipeline, logger, default_args):
    '''
        user arguments를 받아 config를 세팅합니다.
        - config default 값 지정
        - user arguments 값 업데이트
        - config error 조건(def check_config) 확인
        - config 가공(def revise_config)
    '''
    ### config default settings
    if pipeline['name'] == 'train':
        # config['readiness']에 모든 가능한 arguments에 대해서 default 값 채워 넣기
        config['readiness'] = default_args

        for arg_name, user_value in args.items():
            if (arg_name in ['x_columns', 'drop_x_columns', 'groupkey_columns']) and (type(user_value) == str):
                # string -> [string]으로 변경
                if user_value == '':
                    config['readiness'][arg_name] = []
                else:
                    config['readiness'][arg_name] = [user_value]
            else:
                config['readiness'][arg_name] = user_value

        config = revise_config(config, input_data.columns)
        ### config 조합 체크
        check_configs(config, logger)

    else:
        # train에서 사용한 config를 가져와서 덮어 씌움
        # with open(asset.get_model_path() + CONFIG_FILE_NAME, mode='rb') as f:
        train_config = pipeline['model']['train_config']
        config['readiness'] = train_config
        config['readiness']['groupkey_columns'] = config['readiness']['original_groupkey_columns']


    return config