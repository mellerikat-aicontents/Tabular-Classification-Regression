import pandas as pd
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

#from alolib.asset import Asset
import json
import time

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold


class DataManager:
    def __init__(self, config, input_data):
 
        self.input_data = input_data  
        self.config = config 

        self.x = config['readiness']['x_columns']  # 독립 변수 
        self.y = config['readiness']['y_column']  # 타깃 변수

    
    def run(self):
        # group_key 옵션을 고려햐여 데이터 반환 (있으면 dict형식, 없으면 dataframe형식)
        data = self.make_data(input_data=self.input_data, config=self.config)
        
        # split 옵션을 고려하여 데이터 반환
        if 'data_split' in self.config['sampling'].keys():
            # split (kfold / train_test_split)
            if 'random_state' in self.config['sampling'].keys():
                random_state = self.config['sampling']['random_state']
            else:
                random_state = None
            data = self.split_data(data=data, 
                                   task=self.config['readiness']['task_type'],
                                   method=self.config['sampling']['data_split']['method'],
                                   option=self.config['sampling']['data_split']['options'],
                                   random_state=random_state)

        return data

    
    def make_grouped_df(self, df, config): 
        '''df -> {groupA: df_A, groupB: df_B, ...}'''
        group_keys = config['readiness']['original_groupkey_columns']
        groupkey_list = config['readiness']['groupkey_list']
        groupkey_column = config['readiness']['groupkey_columns'][0]

        df_dict = {}
        for groupkey in groupkey_list:
            partial_df = df[df[groupkey_column] == groupkey]
            df_dict[groupkey] = partial_df

        return df_dict


    def train_test_split(self, data, task, test_size=0.2, random_state=None):
        # CASE 1. group_keys O
        if type(data) == dict:  
            groups = data.keys()
            data_res = {}

            for group in groups:
                df_group = data[group]

                if task == 'classification':  # classification 이면 stratify옵션 사용
                    X_train, X_test, y_train, y_test = train_test_split(df_group[self.x], df_group[self.y], random_state=random_state, test_size=test_size, stratify=df_group[self.y])
                elif task == 'regression':  # stratify옵션 사용 X
                     X_train, X_test, y_train, y_test = train_test_split(df_group[self.x], df_group[self.y], random_state=random_state, test_size=test_size)
    
                df_group_train = pd.concat([X_train, y_train], axis=1) 
                df_group_test = pd.concat([X_test, y_test], axis=1) 

                data_res[group] = [df_group_train, df_group_test]

        # CASE 2. group_keys X
        else:  
            if task == 'classification':  # classification 이면 stratify옵션 사용
                X_train, X_test, y_train, y_test = train_test_split(data[self.x], data[self.y], random_state=random_state, test_size=test_size, stratify=data[self.y])
            elif task == 'regression':  # stratify옵션 사용 X
                X_train, X_test, y_train, y_test = train_test_split(data[self.x], data[self.y], random_state=random_state, test_size=test_size)

            df_train = pd.concat([X_train, y_train], axis=1) 
            df_test = pd.concat([X_test, y_test], axis=1) 

            data_res = [df_train, df_test]

        return data_res


    def kfold_split(self, data, task, n_splits=5, random_state=None):
        '''df -> [[df_train1, df_val1], [df_train2, df_val2, ...]]
           groupkey가 있으면 group별 데이터마다 위의 방식으로 split구성'''

        if task == 'classification':
                kfold = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
                print('We use StratifiedKFold, which is suitable for Classification.')
        else:
            kfold = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
            print('We use KFold, which is suitable for Regression.')

        # CASE 1. group_keys O
        if type(data) == dict:  
            groups = data.keys()
            data_res = {}

            for group in groups:
                df_group = data[group]  # n_splits 대상
                df_group_cv = []  # split된 데이터를 저장 - len(df_group_cv)=n_splits

                for train_idx, val_idx in kfold.split(df_group[self.x], df_group[self.y]):
                    df_group_train = df_group.iloc[train_idx]
                    df_group_val = df_group.iloc[val_idx]
                    df_group_cv.append([df_group_train, df_group_val])         

                data_res[group] = df_group_cv

        # CASE 2. group_keys X
        else:  
            data_res = []  # len(data)=n_splits
            for train_idx, val_idx in kfold.split(data[self.x], data[self.y]):
                data_train = data.iloc[train_idx]
                data_val = data.iloc[val_idx]
                data_res.append([data_train, data_val])

        return data_res

    
    def split_data(self, data, task='classification', method='cross_validation', option=5, random_state=None):   
        # classification의 경우 어떤 방식으로 split해도 입력 데이터인 data에서의 y 라벨 분포가 split된 데이터에서도 유지됨
        if method == 'cross_validation':
            data_split = self.kfold_split(data, task, n_splits=option, random_state=random_state)  

        elif method == 'train_test':
            data_split = self.train_test_split(data, task, test_size=option, random_state=random_state)  

        return data_split

    
    def make_data(self, input_data, config):
        # CASE 1. groupkey O
        if len(config['readiness']['original_groupkey_columns']) > 0:  
            # {A: df_A, B: df_B, ...}
            data = self.make_grouped_df(input_data, config) 

            if 'inference' in list(config.keys()):
                data = {key: value for key, value in data.items()}
        
        # CASE 2. groupkey X
        else:  
            data = input_data

        return data