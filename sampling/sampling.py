import sys
import random
import itertools
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from sampling.src_spl.sampler_manager import sampler_manager
from sampling.src_spl.data_manager import DataManager


class Sampling():
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def under_run(self, target_df, target_group_key=[], group_key = None, param_config = None):
        target_df = target_df.copy()
        if len(target_group_key)==0:
            # under, no_group_key
            sampling_method = list(param_config.keys())[0] # method
            
            dict_method = {}
            dict_method['sampling_strategy'] = param_config[sampling_method]
            dict_method['random_state'] = param_config['random_state']
            self.sampler_manager.set_param('under', sampling_method, dict_method)

            changed_idx = self.under_sampling(target_df, sampling_method) # sampling run
            origin_index = target_df.index
            temp_list = [origin_index[changed_idx[x]] for x in range(len(changed_idx))]
            
            target_df['tcr_sampled'] = 'none'
            target_df.loc[temp_list, 'tcr_sampled'] = 'sampled'

        else:
            #under, with group_key
            target_df['tcr_sampled'] = 'none'
            target_df['tcr_sampled'] = np.where(target_df[group_key].isin(target_group_key)==False, 'sampled', 'none')

            under_idx_list = []
            for i in range(len(target_group_key)):
                temp_df = target_df[target_df[group_key] == target_group_key[i]].copy()
                if target_group_key[i] not in param_config.keys():
                    target_df.loc[target_df[target_df[group_key] == target_group_key[i]].index, 'tcr_sampled'] = 'sampled'
                    continue
                sampling_method = list(param_config[target_group_key[i]].keys())[0] # method
            
                dict_method = {} # groupkey 별 set_param
                dict_method['sampling_strategy'] = param_config[target_group_key[i]][sampling_method]
                dict_method['random_state'] = param_config[target_group_key[i]]['random_state']
                self.sampler_manager.set_param('under', sampling_method, dict_method)

                changed_idx = list(self.under_sampling(temp_df, sampling_method)) # sampling run
                origin_index = temp_df.index
                
                temp_list = [origin_index[changed_idx[x]] for x in range(len(changed_idx))]

                under_idx_list = under_idx_list + temp_list

            target_df.loc[under_idx_list, 'tcr_sampled'] = 'sampled'
        
        
        return target_df
    
    def under_sampling(self, target_df, sampling_method):
        target_df = target_df.copy()
        x_data = target_df[self.config['readiness']['x_columns']]
        y_data = target_df[self.config['readiness']['y_column']]
        x_res, y_res, changed_idx = self.sampler_manager.under_fit_resampling(sampling_method, x_data, y_data)
        return changed_idx
    
    def over_run(self, target_df, start_idx, target_group_key=[], group_key = None, param_config = None):
        target_df = target_df.copy()
        if len(target_group_key) == 0:
            # over, no_group_key
            sampling_method = list(param_config.keys())[0] # method
            
            dict_method = {}
            dict_method['sampling_strategy'] = param_config[sampling_method]
            dict_method['random_state'] = param_config['random_state']
            self.sampler_manager.set_param('over', sampling_method, dict_method)

            over_df = self.over_sampling(target_df, sampling_method) # sampling run
            over_df['tcr_sampled'] = 'over'
            target_df['tcr_sampled'] = 'sampled'
            
            over_df = over_df.set_index(pd.Index([x + start_idx for x in range(len(over_df))]))
            
            target_df = pd.concat([target_df, over_df])
        else:
            over_df_list = []
            for groupkey in list(param_config.keys()):
                temp_df = target_df[target_df[group_key] == groupkey].copy()
                sampling_method = list(param_config[groupkey].keys())[0] # method

                dict_method = {}
                dict_method['sampling_strategy'] = param_config[groupkey][sampling_method]
                dict_method['random_state'] = param_config[groupkey]['random_state']
                self.sampler_manager.set_param('over', sampling_method, dict_method)

                over_df = self.over_sampling(temp_df, sampling_method) # sampling run
                over_df[group_key] = groupkey
                over_df_list.append(over_df)

            over_df_total = pd.concat(over_df_list)
            target_df['tcr_sampled'] = 'sampled'
            over_df_total['tcr_sampled'] = 'over'
            
            over_df_total = over_df_total.set_index(pd.Index([x + start_idx for x in range(len(over_df_total))]))
            target_df = pd.concat([target_df, over_df_total])
            
        
        return target_df
            
    def over_sampling(self, target_df, sampling_method):
        target_df = target_df.copy()
        x_data = target_df[self.config['readiness']['x_columns']]
        y_data = target_df[self.config['readiness']['y_column']]
        over_df = self.sampler_manager.over_fit_resampling(sampling_method, x_data, y_data)
        return over_df
    
    def run(self, target_df, param_config = None, target_group_key=[], group_column_name = None, start_idx = 0):
        task_type = (lambda : 'over_sampling' if ('over_sampling' in self.config['sampling'].keys()) else \
                        ('under_sampling' if 'under_sampling' in self.config['sampling'].keys() else False))()

        self.config['sampling']['task_type'] = task_type
        self.sampler_manager = sampler_manager(self.config, self.logger)

        if param_config is None:
            param_config = defaultdict(dict)
            if group_column_name != None: # groupkey O
                for groupkey in target_group_key:
                    # self.save_info(f"groupkey '{groupkey}'에 대해서 sampling을 진행합니다.")
                    self.save_info(f"We are proceeding with sampling for the groupkey '{groupkey}'.")
                    parser_config = Sampling_Config_Parser(target_df[target_df[group_column_name] == groupkey], self.config, task_type, self.logger).run()
                    if parser_config != None: param_config.update({groupkey: parser_config})
            else: # groupkey X
                parser_config = Sampling_Config_Parser(target_df, self.config, task_type, self.logger).run()
                if parser_config != None: param_config = parser_config

        if param_config == {}:
            target_df['tcr_sampled'] = 'sampled'
            return target_df
        
        if 'under_sampling' == task_type:
            return self.under_run(target_df, target_group_key, group_column_name, param_config)

        else:
            return self.over_run(
                target_df, 
                start_idx = start_idx, 
                target_group_key=target_group_key,
                group_key = group_column_name,
                param_config = param_config
            )
    
    def save_info(self, msg):
        self.logger['info'](msg)
     
    def save_warning(self, msg):
        self.logger['warning'](msg)
 
    def save_error(self, msg):
        self.logger['error'](msg)


class Sampling_Config_Parser:
    def __init__(self, input_data, config, sampling_key, logger):
        self.input_data = input_data
        self.config = config
        self.y_mapping_table = self.get_y_mapping_table()
        self.label2num = {v:k for k, v in self.y_mapping_table.items()}
        self.label2count = self.get_label2count()
        self.sampling_key = sampling_key
        self.logger = logger
        
    def run(self):
        sampler_parameter = {}

        config_tmp = self.config['sampling'][self.sampling_key]
        label = config_tmp['label'] 
        label = label if isinstance(label, list) else [label]
        method = config_tmp['method']

        if 'random_state' in self.config['sampling'].keys():
            config_tmp['random_state'] = self.config['sampling']['random_state']
 
        if 'ratio' in config_tmp.keys():
            ratio = config_tmp['ratio'] 
            ratio = ratio if isinstance(ratio, list) else [ratio]
            sampler_parameter = {method:{l:self.get_size(l, r) for l, r in zip(label, itertools.cycle(ratio))}}

        elif 'compare' in config_tmp.keys():
            compare = config_tmp['compare']
            target = compare['target']
            multiply = compare['multiply']
            multiply = multiply if isinstance(multiply, list) else [multiply]
            sampler_parameter = {method:{l:self.get_size(target, m) for l, m in zip(label, itertools.cycle(multiply))}}
            
        # break # TODO: 추후 under/over 동시에 진행할 때를 위해 둘 다 입력했을 경우 첫번째 입력된 샘플링만 파싱하도록 설정
        random_state = None
        if 'random_state' in self.config['sampling'].keys():
            random_state = self.config['sampling']['random_state']
        return self.check_validity_and_modify(sampler_parameter, self.sampling_key, method, random_state=random_state)
    
    def get_y_mapping_table(self):
        config_tmp = self.config['preprocess']
        try:
            y_mapping_table = config_tmp['y_mapping_table']
        except:
            groupkey_tmp = self.config['readiness']['groupkey_list'][0]
            y_mapping_table = config_tmp[groupkey_tmp]['y_mapping_table']
        
        return y_mapping_table
    
    def get_label2count(self):
        return {label:self.get_labeling_count(label) for label in list(self.y_mapping_table.values())}
            
    def get_labeling_count(self, label):
        label = self.label2num[label]

        return len(self.input_data[self.input_data[self.config['readiness']['y_column']]==label])
    
    def get_size(self, label, value):    
        return int(self.label2count[label] * value)
        
    def check_validity_and_modify(self, sampler_parameter, sampling, method, random_state=None):
        params = sampler_parameter[method]
        
        is_undersampling = sampling == 'under_sampling'
        word = 'larger' if is_undersampling else 'smaller' #TODO: warning error 수정

        labels = list(params.keys())
        before_values = np.array([self.label2count[k] for k in labels])
        current_values = np.array(list(params.values()))

        diff = before_values - current_values
        not_valid_idxs = np.where(diff <= 0 if is_undersampling else diff >= 0)[0]

        for idx in not_valid_idxs: #TODO: warning error 수정
            # self.save_warning(f'입력한 샘플링된 {labels[idx]} 데이터 수 {current_values[idx]}가 {before_values[idx]}보다 같거나 {word}기 때문에 {sampling}을 진행하지 않습니다.')
            self.save_warning(f'We will not proceed with {sampling} because the number of sampled {labels[idx]} data {current_values[idx]} is the same or {word} than {before_values[idx]}.')

        labels, values = np.delete(labels, not_valid_idxs), np.delete(current_values, not_valid_idxs)

        if labels.size != 0:
            sampler_parameter[method] = {self.label2num[l]:v for l, v in zip(labels, values)}
            sampler_parameter['random_state'] = random_state

        else:
            sampler_parameter = None

        return sampler_parameter           
    
    def save_info(self, msg):
        self.logger['info'](msg)
     
    def save_warning(self, msg):
        self.logger['warning'](msg)
 
    def save_error(self, msg):
        self.logger['error'](msg)


def apply_sampling(input_data, data_collection, config, sampling, sampling_type, data_split_method, is_groupkey):
    if sampling_type[0] == 'over_sampling':
        start_idx = len(input_data)
    else:
        start_idx = 0  # default in samplng.py

    if is_groupkey:
        groupkey_list = config['readiness']['groupkey_list']
        group_col = config['readiness']['groupkey_columns'][0]

    data_res = data_collection.copy()

    def sample(data_train, start_idx):
        data_train_sampled = sampling.run(target_df=data_train, start_idx=start_idx)
        if start_idx > 0:
            start_idx += len(data_train_sampled)
        return data_train_sampled, start_idx

    # CASE 1.1 groupkey X & cross-val
    if data_split_method == 'cross_validation' and not is_groupkey:
        for cv_idx in range(len(data_collection)):
            data = data_collection[cv_idx]  # [df_train_i, df_val_i]
            data_train = data[0]

            data_train_sampled, start_idx = sample(data_train=data_train, start_idx=start_idx)
            data_res[cv_idx][0] = data_train_sampled


    # CASE 1.2 groupkey X & train-test split
    elif data_split_method == 'train_test' and not is_groupkey:
        # data_collection = [df_train, df_val]
        data_train = data_collection[0]

        data_train_sampled, start_idx = sample(data_train=data_train, start_idx=start_idx)
        data_res[0] = data_train_sampled


    # CASE 2.1 groupkey O & cross-val
    elif data_split_method == 'cross_validation' and is_groupkey:
        for group in groupkey_list:
            data_group = data_collection[group]  # [[df_train1, df_val1], [df_train2, df_val2], ...]
            for cv_idx in range(len(data_group)):
                data = data_group[cv_idx]  # [df_train_i, df_val_i]
                data_train = data[0]

                data_train_sampled, start_idx = sample(data_train=data_train, start_idx=start_idx)
                data_res[group][cv_idx][0] = data_train_sampled


    # CASE 2.2 groupkey O & train-test split
    elif data_split_method == 'train_test' and is_groupkey:
        for group in groupkey_list:
            data_group = data_collection[group]  # [df_train, df_val]
            data_train = data_group[0]

            data_train_sampled, start_idx = sample(data_train=data_train, start_idx=start_idx)
            data_res[group][0] = data_train_sampled


    # CASE 3.1 groupkey X & data_split X
    elif data_split_method == 'False' and not is_groupkey:
        data_res = sample(data_collection, start_idx=start_idx)
        data_res = data_res[0]


    # CASE 3.2 groupkey O & data_split X
    elif data_split_method == 'False' and is_groupkey:
        data_res = sampling.run(target_df=data_collection, target_group_key=groupkey_list, group_column_name=group_col, start_idx=len(data_collection))

    return data_res

def get_output(config, input_data, logger):
    # groupkey check
    is_groupkey = len(config['readiness']['groupkey_list']) > 0

    # sampling check
    is_sampling = len([key for key in config['sampling'].keys() if '_sampling' in key]) > 0

    df = input_data

    if is_sampling and (config['readiness']['task_type'] == 'classification'):
        sampling = Sampling(config=config, logger=logger)  # sampling을 수행하는 class객체 호출
        sampling_type = [key for key in config['sampling'] if '_sampling' in key]  # ['over_sampling'] / ['under_sampling']
        df = apply_sampling(data_collection=input_data, sampling=sampling,
                                            sampling_type=sampling_type, data_split_method='False', is_groupkey=is_groupkey)

    # data split check
    is_data_split = 'data_split' in config['sampling'].keys()
    data_collection = None

    if is_data_split:
        data_split_method = config['sampling']['data_split']['method']  # cross_val / train_test
        data_manager = DataManager(config=config, input_data=input_data)
        data_collection = data_manager.run()
        if is_sampling and (config['readiness']['task_type'] == 'classification'):
            data_collection = apply_sampling(data_collection=data_collection, sampling=sampling, config=config,
                                                sampling_type=sampling_type, data_split_method=data_split_method, is_groupkey=is_groupkey)

    return df, data_collection