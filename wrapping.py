import argparse
import time
import os
os.chdir(os.path.abspath(os.path.join('./alo')))
from src.alo import ALO
from src.alo import AssetStructure
from src.external import external_load_data, external_save_artifacts
import pickle
from glob import glob
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno

EVAL_PATH = './evaluation_data'
PIPELINE_DICT = {'train_pipeline':'train', 'inference_pipeline':'inference'}
CARDINALITY_LIMIT = 20

class Wrapper(ALO):
    def __init__(self, nth_pipeline, eval_report=True):
        super().__init__()
        self.preset()
        pipelines = list(self.asset_source.keys())
        self.pipeline = pipelines[nth_pipeline]
        self.eval_report = eval_report
        
        external_load_data(pipelines[nth_pipeline], self.external_path, self.external_path_permission, self.control['get_external_data'])
        self.install_steps(self.pipeline, self.control["get_asset_source"])
        envs, args, data, config = {}, {}, {}, {}
        self.asset_structure = AssetStructure(envs, args, data, config)
        self.set_proc_logger()
        self.step = 0
        self.args_checker = 0
        
        if eval_report:
            os.makedirs(EVAL_PATH, exist_ok=True)
        
        
    def get_args(self, step=None):
        if step is None:
            step = self.step
        self.args = super().get_args(self.pipeline, step)
        self.asset_structure.args = self.args
        self.args_checker = 1

        return self.args
    
    def run(self, step=None, args=None, data=None):
        if self.args_checker==0:
            self.get_args()
        
        if step is None:
            step = self.step
        if args is not None:
            self.asset_structure.args = args
        if data is not None:
            self.asset_structure.data = data
            
        self.asset_structure = self.process_asset_step(self.asset_source[self.pipeline][step], step, self.pipeline, self.asset_structure)
        
        self.data = self.asset_structure.data
        self.args = self.asset_structure.args
        self.config = self.asset_structure.config
        
        if self.eval_report:
            self.save_pkl(self)
        
        self.step += 1
        self.args_checker = 0
        
    def save_pkl(self, obj):
        path = '{eval_path}/{pipeline}_{step}.pkl'.format(
            eval_path=EVAL_PATH, 
            pipeline=PIPELINE_DICT[self.pipeline],
            step=self.step)
        with open(path, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
            
    def load_pkl(self, pipeline, step):
        path = '{eval_path}/{pipeline}_{step}.pkl'.format(
            eval_path=EVAL_PATH, 
            pipeline=pipeline,
            step=step)
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj
    
    
class EvaluationReport:
    def __init__(self):
        self.data = {}
        self.args = {}
        self.config = {}
        self.train_path_list = sorted(list(glob('{eval_path}/{pipeline}_*.pkl'.format(
                eval_path=EVAL_PATH, 
                pipeline='train'))))
        self.inf_path_list = sorted(list(glob('{eval_path}/{pipeline}_*.pkl'.format(
                eval_path=EVAL_PATH, 
                pipeline='inference'))))
        self.params = {}
        
        self_train_describe = None
        
    def viz_class_distribution(self, cat_col=None, df=None):
        if df is None:
            df = self._df
        if cat_col is None:
            cat_col = self._y_column
        
        labels, data = df[cat_col].value_counts().index, df[cat_col].value_counts().values
        cardinality = df[cat_col].nunique()
        if cardinality>CARDINALITY_LIMIT:
            print('This method supports cardinality under 11, currently cardinality is {}'.format(cardinality))
        else:
            colors = sns.color_palette('pastel')[:len(labels)]
            plt.figure(figsize=(10, 4))
            plt.subplot(121)
            plt.title('Category ratio:{}'.format(cat_col))
            plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
            plt.subplot(122)
            plt.title('Category count:{}'.format(cat_col))
            plt.bar(labels, data, color=colors)
            for i in range(len(labels)):
                plt.text(i,data[i],data[i])
            plt.show()

    def create_dataframe_viz(self, nx_cols=None, cat_cols=None, df=None):
        if df is None:
            df = self._df
        if nx_cols is None:
            nx_cols = self._nx_columns
        if cat_cols is None:
            cat_cols = [self._y_column]
        df_viz = df[nx_cols].stack().reset_index()

        df_viz.columns = ['idx', 'nx_columns', 'value']
        cat_cols = [col for col in cat_cols if col in df.columns]
        for cat_col in cat_cols:
            df_viz[cat_col] = [i for i in df[cat_col] for j in range(len(nx_cols))]
        self.df_viz = df_viz
        
        return df_viz

    def viz_pairplot(self, nx_cols=None, cat_col=None, df=None):
        if df is None:
            df = self._df
        if nx_cols is None:
            nx_cols = self._nx_columns
        if cat_col is None:
            cat_col = self._y_column
        
        if cat_col in df.columns:
            cardinality = df[cat_col].nunique()
        else:
            cardinality = 0
        
        if cardinality>CARDINALITY_LIMIT:
            print('This method supports cardinality under 11, currently cardinality is {}'.format(cardinality))
        elif cat_col in df.columns:
            colors = sns.color_palette('pastel')[:cardinality]
            sns.pairplot(data=df[nx_cols+[cat_col]], hue=cat_col, palette=colors)
            plt.show()
        else:
            sns.pairplot(data=df[nx_cols])
            plt.show()
    def viz_boxplot(self, cat_col=None, df_viz=None, vline=None):
        if cat_col is None:
            cat_col = self._y_column
        if df_viz is None:
            df_viz = self.df_viz
        
        if cat_col in df_viz.columns:
            cardinality = df_viz[cat_col].nunique()
        else:
            cardinality = 0
        
        if cardinality>CARDINALITY_LIMIT:
            print('This method supports cardinality under 11, currently cardinality is {}'.format(cardinality))
        elif cat_col in df_viz.columns:
            colors = sns.color_palette('pastel')[:cardinality]
            sns.boxplot(data=df_viz, y='nx_columns', x='value', hue=cat_col, palette=colors, orient='h')
            if vline is not None:
                plt.axvline(vline, ls='--', c='lightgray')
            plt.legend(loc='center left', bbox_to_anchor=(1., .5))
            plt.show()
        else:
            sns.boxplot(data=df_viz, y='nx_columns', x='value', orient='h')
            if vline is not None:
                plt.axvline(vline, ls='--', c='lightgray')
#             plt.legend(loc='center left', bbox_to_anchor=(1., .5))
            plt.show()

    def viz_missing(self, x_columns=None, y_column=None, df=None):
        if df is None:
            df = self._df
        if x_columns is None:
            x_columns = self._x_columns
        if y_column is None:
            y_column = self._y_column
        msno.matrix(df[x_columns+[y_column]])
        plt.show()
        
    def viz_probability(self, y_pred_column=None, k_fold_split=False, df=None):
        if df is None:
            df = self._df
        if y_pred_column is None:
            y_pred_column = self._y_pred_column
        prob_cols = df.columns[df.columns.str.contains('prob_')]
        df['prob_max'] = df[prob_cols].max(axis=1)
        plt.title(self.params['evaluation_metric'])
        if self.params['data_split_method']=='cross_validate' and k_fold_split:
            colors = sns.color_palette('pastel')[:4]
            sns.boxplot(df, y='prob_max', x=y_pred_column, hue='train_test', palette=colors)
        else:
            sns.boxplot(df, y='prob_max', x=y_pred_column)
        plt.show()
        return df
        
    def describe(self, x_columns=None, y_column=None, df=None):
        if df is None:
            df = self._df
        if x_columns is None:
            x_columns = self._x_columns
        if y_column is None:
            y_column = self._y_column
        if y_column in df.columns:
            # data에 y_column이 있는 경우
            df_describe = df[x_columns+[y_column]].describe(include='all').fillna('-')
        else:
            # data에 y_column이 없는 경우
            df_describe = df[x_columns].describe(include='all').fillna('-')
        if self.p_step=='train_0':
            self._train_describe = df_describe
        if self.p_step=='inference_0' and self._train_describe is not None:
            self._inference_describe = df_describe
            return pd.concat([self._train_describe, self._inference_describe], keys=['train', 'inference'], axis=1)
        return df_describe
    
    def viz_confusion_matrix(self, y_column=None, y_pred_column=None, df=None):
        if df is None:
            df = self._df
        if y_column is None:
            y_column = self._py_column
        if y_pred_column is None:
            y_pred_column = self._y_pred_column

        indices = sorted(df[y_column].value_counts().index)

        plt.title('Confusion matrix')
        sns.heatmap(confusion_matrix(df[y_column], df[y_pred_column], labels=indices), xticklabels=indices, yticklabels=indices, annot=True)
        plt.show()
        
    def classification_report(self, y_column=None, y_pred_column=None, df=None):
        if df is None:
            df = self._df
        if y_column is None:
            y_column = self._py_column
        if y_pred_column is None:
            y_pred_column = self._y_pred_column
            
        indices = sorted(df[y_column].value_counts().index)
        print('Classification report')
        print(classification_report(df[y_column], df[y_pred_column], labels=indices))
        
    def set_data(self, p_step):
        self.p_step = p_step
        self._data = self.data[p_step]
        self._df = self.data[p_step]['dataframe']
        self._args = self.args[p_step]
        self._config = self.config[p_step]
        if p_step == 'train_0':
            self._nx_columns = self._args['x_columns'] # 추후 변경 필요
            self._x_columns = self._args['x_columns']
            self._y_column = self._args['y_column']
        if p_step == 'train_1':
            self._pnx_columns = self._config['x_columns'] # 추후 변경 필요
            self._px_columns = self._config['x_columns']
            self._py_column = self._config['y_column']
        if p_step == 'train_3':
            self._y_pred_column = 'pred_' + self._config['y_column']
        if p_step == 'inference_2':
            self._y_pred_column = 'pred_' + self._config['y_column']
        return self._data, self._args, self._config
        
    def load_data(self):
        key_list = []
        path_list = self.train_path_list + self.inf_path_list
        
        for p, path in enumerate(path_list):
            if p < len(self.train_path_list):
                pipeline = 'train'
                pth = p
            else:
                pipeline = 'inference'
                pth = p - len(self.train_path_list)
            obj = self.load_pkl(path=path)
            key = '{}_{}'.format(pipeline, pth)
            self.data[key] = obj.data
            self.args[key] = obj.args
            self.config[key] = obj.config
            key_list.append(key)
            if 'model_type' in obj.args.keys():
                self.params['model_type'] = obj.args['model_type']
            if 'evaluation_metric' in obj.args.keys():
                self.params['evaluation_metric'] = obj.args['evaluation_metric']
            if 'data_split_method' in obj.args.keys():
                self.params['data_split_method'] = obj.args['data_split_method']
                
        print("{} are loaded".format(key_list))
        
#     def load_pkl(self, pipeline, step):
    def load_pkl(self, path=None, pipeline=None, step=None):
        if path is None:
            path = '{eval_path}/{pipeline}_{step}.pkl'.format(
                eval_path=EVAL_PATH, 
                pipeline=pipeline,
                step=step)
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj