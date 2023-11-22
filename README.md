# Welcome to TCR !

⚡ input/target data로 구성된 Tabular 형태의 데이터에 대해 분류/예측할 수 있는 AI 컨텐츠입니다. ⚡

[![Generic badge](https://img.shields.io/badge/release-v1.1.4-green.svg?style=for-the-badge)](http://링크)
[![Generic badge](https://img.shields.io/badge/last_update-2023.11.17-002E5F?style=for-the-badge)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Generic badge](https://img.shields.io/badge/python-3.10.12-purple.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Generic badge](https://img.shields.io/badge/dependencies-up_to_date-green.svg?style=for-the-badge&logo=python&logoColor=white)](requirement링크)
[![Generic badge](https://img.shields.io/badge/collab-blue.svg?style=for-the-badge)](http://collab.lge.com/main/display/AICONTENTS)
[![Generic badge](https://img.shields.io/badge/request_clm-green.svg?style=for-the-badge)](http://collab.lge.com/main/pages/viewpage.action?pageId=2157128981)


## 데이터 준비
1. Single y target(문자열 혹은 실수형 데이터)으로 구성되어야 합니다.
2. 분류의 경우 추론 데이터의 라벨의 유형이 학습 데이터에 반드시 포함되어 있어야 합니다.
3. 결손치 수가 10% 이하여야 합니다.
4. 최소 데이터 수는 100개 이상, 데이터 용량은 100GB이하로 준비해야 합니다.
5. 카테고리 변수의 경우 최소한의 학습 성능을 보장하기 위해서 변수 내 카테고리 종류가 총 데이터 수보다 작아야 합니다.
6. 데이터의 항목명은 문자열로 이루어져 있으며 숫자로만 구성되면 안됩니다.

데이터 명세서 상세한 내용은 [Documentation](http://collab.lge.com/main/pages/viewpage.action?pageId=2178808470)를 참고해주세요.

샘플 데이터 설명: [iris flower dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set)
 


## 주요 기능 소개
- 총 다섯가지 머신러닝 알고리즘에 대해 HPO를 수행하여 가장 성능이 우수한 모델을 도출합니다.
- 모델 평가 단계에서 4-fold 교차검증 방식을 채택하여 검증 데이터 선별에 의한 과최적화를 예방하여 신뢰성 있는 모델을 얻을 수 있습니다.
- 위 과정에서 병렬연산을 수행하여 빠르게 모델을 학습/도출할 수 있습니다.
- 다른 asset과 조합하거나 custom asset과 결합하여 분석가가 원하는 수준의 모델을 개발할 수 있습니다.

상세한 설명은 [Documentation](http://collab.lge.com/main/pages/viewpage.action?pageId=2178808489)을 참고해주세요. 

## Quick Install Guide


```
git clone http://mod.lge.com/hub/dxadvtech/aicontents/tcr.git 
cd tcr 

conda create -n tcr python=3.10
conda init bash
conda activate tcr 

#jupyter 사용시 ipykernel 추가 필요
#pip install ipykernel
#python -m ipykernel install --user --name tcr 

source install.sh

```

## Quick Run Guide
- 아래 코드 블럭을 실행하면 TCR이 실행되고 이때 자동으로 `alo/config/experimental_plan.yaml`을 참조합니다. 
```
cd alo
python main.py 
```
<details><summary>실행 과정 도중 permission error가 발생한 경우</summary>
오류가 발생한 파일(예: config/experimental_plan.yaml)에 대해 하단 코드를 실행해주세요.
sudo chmod 777 config/experimental_plan.yaml
</details>

- TCR 구동을 위해서는 분석 데이터에 대한 정보 및 사용할 TCR 기능이 기록된 yaml파일이 필요합니다.  
- TCR default yaml파일인 `alo/config/experimental_plan.yaml`의 argument를 변경하여 분석하고 싶은 데이터에 TCR을 적용할 수 있습니다.
- 필수적으로 수정해야하는 ***arguments***는 아래와 같습니다. 
***
external_path:  
&emsp;- *load_train_data_path*: ***~/example/train_data_folder/***  # 학습 데이터가 들어있는 폴더 경로 입력(csv 입력 X)  
&emsp;- *load_inference_data_path*: ***~/example/inference_data_folder/***  # 추론 데이터가 들어있는 폴더 경로 입력(csv 입력 X)  
user_parameters:  
&emsp;- train_pipeline:  
&emsp;&emsp;- step: input  
&emsp;&emsp;&emsp;args:  
&emsp;&emsp;&emsp;- *input_path*: ***train_data_folder***  # 학습 데이터가 들어있는 폴더  
&emsp;&emsp;&emsp;&emsp;*x_columns*: ***[column1,column2]***  # 분석 데이터의 X컬럼 명  
&emsp;&emsp;&emsp;&emsp;*y_column*: ***label***  # 분석 데이터의 Y컬럼 명  
&emsp;&emsp;&emsp;&emsp;...  
&emsp;&emsp;- step: preprocess   
&emsp;&emsp;&emsp;args:   
&emsp;&emsp;&emsp;- *handling_encoding_y*: ***label***     # 분석 데이터의 Y컬럼 명(input과 동일하게 입력)  
&emsp;&emsp;&emsp;&emsp;...   
&emsp;- inference_pipeline:  
&emsp;&emsp;- step: input  
&emsp;&emsp;&emsp;args:   
&emsp;&emsp;&emsp;- *input_path*: ***inference_data_folder***  # 추론 데이터가 들어있는 폴더  
&emsp;&emsp;&emsp;&emsp;*x_columns*: ***[column1,column2]***  #분석 데이터의 X컬럼 명  
&emsp;&emsp;&emsp;&emsp;*y_column*: ***label***  # 분석 데이터의 Y컬럼 명  
&emsp;&emsp;&emsp;&emsp;...  
***
- preprocess, sampling 및 TCR의 다양한 기능을 사용하고 싶으신 경우 [User Guide (TCR)](http://collab.lge.com/main/pages/viewpage.action?pageId=2184973450)를 참고하여 yaml파일을 수정하시면 됩니다. 
- 학습 결과 파일 저장 경로: `alo/.train_artifacts/models/train/`
- 추론 결과 파일 저장 경로: `alo/.inference_artifacts/output/inference/`



## Sample notebook
Jupyter 환경에서 Workflow 단계마다 asset을 실행하고 setting을 바꿔 실험할 수 있습니다. [Sample notebook 링크](http://mod.lge.com/hub/dxadvtech/aicontents/tcr/-/blob/main/TCR_user.ipynb)
각 asset에 대한 자세한 설명은 하단 guide notebook을 참고해주세요.
- [Preprocess - TBD]()
- [Sampling](http://mod.lge.com/hub/dxadvtech/aicontents/tcr/-/blob/main/TCR_guide_sampling.ipynb)
- [Train/inference](http://mod.lge.com/hub/dxadvtech/aicontents/tcr/-/blob/main/TCR_guide_modeling.ipynb)
- [Evaluation report for classification](http://mod.lge.com/hub/dxadvtech/aicontents/tcr/-/blob/main/Evaluation_report_classification.ipynb)

## 관련 Collab
[AICONTENTS](http://collab.lge.com/main/display/AICONTENTS)

## 요청 및 문의
담당자: 서윤지(yoonji.suh@lge.com)

신규 AI Contents나 추가 기능 요청을 등록하시면 검토 후 반영합니다  [Request CLM](http://clm.lge.com/issue/projects/AICONTENTS/summary)


