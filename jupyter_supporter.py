# python lib
import os
import shutil
import sys
import subprocess
import time
original_directory = os.getcwd()
# import alolib
import copy

# local import
os.chdir(os.path.abspath('alo'))
from common import Logger, print_color, find_matching_strings, asset_info, extract_requirements_txt, check_install_requirements

req_list = extract_requirements_txt("master")
master_req = {"master": req_list}
check_install_requirements(master_req)

from alolib.asset import Asset
from alolib.exception import print_color

class Jsupport:
    def __init__(self):
        os.chdir(os.path.abspath('alo'))
        self.pipe_mode = 'train_pipeline'
        req_list = extract_requirements_txt("master")
        master_req = {"master": req_list}
        
        check_install_requirements(master_req)

        self.PROJECT_HOME = os.getcwd()
        # experimental plan yaml의 위치
        EXP_PLAN = self.PROJECT_HOME + "/config/experimental_plan.yaml"
        # asset 코드들의 위치
        self.SCRIPT_HOME = self.PROJECT_HOME + "/assets/"

        envs = {}
        envs['project_home'] = self.PROJECT_HOME + "/"

        self.SUPPORT_TYPE = ['memory', 'file']

        req_list = extract_requirements_txt("master")
        master_req = {"master": req_list}
        check_install_requirements(master_req)
        
        # asset init
        self.asset = Asset(envs=envs, argv=0, version=0.1)
        # configure setting
        self.asset.get_yaml(EXP_PLAN)
        self.external_path = self.asset.get_external_path()
        self.external_path_permission = self.asset.get_external_path_permission()
        self.pipelines_list = self.asset.get_pipeline()
        self.user_parameters = self.asset.get_user_parameters()
        self.controls = self.asset.get_control()
        self.artifacts = self.asset.set_artifacts()
        
        ####################### ALO master requirements 리스트업 #######################
        self.requirements_dict = dict() 
        self.requirements_dict['master'] =  extract_requirements_txt(step_name = 'master')
        ####################### Slave Asset 설치 및 Slave requirements 리스트업 #######################
        # setup asset (asset을 git clone (or local) 및 requirements 설치)
        get_asset_source = self.controls["get_asset_source"]  # once, every
        for step, asset_config in enumerate(self.pipelines_list[self.pipe_mode]):
            # self.asset.setup_asset 기능 :
            # local or git pull 결정 및 scripts 폴더 내에 위치시킴 
            self.asset.setup_asset(asset_config, get_asset_source)
            self.requirements_dict[asset_config['step']] = asset_config['source']['requirements']
            # local 모드일 땐 이번 step(=asset)의 종속 package들이 내 환경에 깔려있는 지 항상 체크 후 없으면 설치 
            # git 모드일 땐 every이거나 once면서 첫 실행 시에만 requirements 설치 

        ####################### Master & Slave requirements 설치 #######################
        self.asset.check_install_requirements(self.requirements_dict) 
        self.check_asset_source = self.controls["get_asset_source"]
    
    def download_data(self):
        self.asset.fetch_data(self.external_path, self.external_path_permission)
    
    def run(self, step, args, pipelines, data, pipe_val):
        asset_config = self.pipelines_list[pipelines][step]
        self.asset.setup_asset(asset_config, self.check_asset_source)

        asset_info(pipelines, asset_config['step'])

        # scripts 폴더에 있는 내용을 가져와 import 한다
        _path = self.SCRIPT_HOME + asset_config['step'] + "/"
        _file = "asset_" + asset_config['step']
        user_asset = self.asset.import_asset(_path, _file)

        if self.controls['interface_mode'] in self.SUPPORT_TYPE:
            # 첫 동작시에는 초기화하여 사용 
            if step == 0:
                data = 0
                pipe_val = {}
            else:
                if self.controls['interface_mode'] == 'memory':
                    pass
                elif self.controls['interface_mode'] == 'file':
                    data, pipe_val = self.asset.get_toss(pipelines, envs) # file interface
        else:
            return ValueError("only file and memory")

        envs = {}
        envs['project_home'] = self.PROJECT_HOME
        envs['pipeline'] = pipelines
        envs['step'] = self.user_parameters[pipelines][step]['step']
        envs['artifacts'] = self.artifacts

        ua = user_asset(envs, args, data, pipe_val)
        data, pipe_val = ua.run()

        # self.asset.save_file(data)

        sys.path = [item for item in sys.path if envs['step'] not in item]

        if self.controls['interface_mode'] == 'file':
            self.asset.toss(data, pipe_val, pipelines, envs)
        else:
            return data, pipe_val
                    
    def get_arguments(self,pipelines,step):
        return self.user_parameters[pipelines][step]['args'][0].copy()
