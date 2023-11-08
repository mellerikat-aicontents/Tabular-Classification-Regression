import argparse
import time
import os
os.chdir(os.path.abspath(os.path.join('./alo')))
from src.alo import ALO
from src.alo import AssetStructure
from src.external import external_load_data, external_save_artifacts

class SimpleALO(ALO):
    def __init__(self, nth_pipeline):
        super().__init__()
        self.preset()
        pipelines = list(self.asset_source.keys())
        self.pipeline = pipelines[nth_pipeline]
        
        external_load_data(pipelines[nth_pipeline], self.external_path, self.external_path_permission, self.control['get_external_data'])
        self.install_steps(self.pipeline, self.control["get_asset_source"])
        envs, args, data, config = {}, {}, {}, {}
        self.asset_structure = AssetStructure(envs, args, data, config)
        self.set_proc_logger()
        self.step = 0
        self.args_checker = 0
        
    def get_args(self, step=None):
        if step is None:
            step = self.step
        self.args = super().get_args(self.pipeline, step)
        self.asset_structure.args = self.args
        self.args_checker = 1
        
        return self.args
    
    def run(self, step=None, args=None):
        if self.args_checker==0:
            self.get_args()
        
        if step is None:
            step = self.step
        if args is not None:
            self.asset_structure.args = args
            
        self.asset_structure = self.process_asset_step(self.asset_source[self.pipeline][step], step, self.pipeline, self.asset_structure)
        
        self.data = self.asset_structure.data
        self.args = self.asset_structure.args
        self.config = self.asset_structure.config
        
        self.step += 1
        self.args_checker = 0