# Welcome to TCR !

⚡ input/target data로 구성된 Tabular 형태의 데이터에 대해 분류/예측할 수 있는 AI 컨텐츠입니다. ⚡

[![Generic badge](https://img.shields.io/badge/release-v1.0.0-green.svg?style=for-the-badge)](http://링크)
[![Generic badge](https://img.shields.io/badge/last_update-2023.10.16-002E5F?style=for-the-badge)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Generic badge](https://img.shields.io/badge/python-3.10.12-purple.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Generic badge](https://img.shields.io/badge/dependencies-up_to_date-green.svg?style=for-the-badge&logo=python&logoColor=white)](requirement링크)
[![Generic badge](https://img.shields.io/badge/collab-blue.svg?style=for-the-badge)](http://collab.lge.com/main/display/AICONTENTS)
[![Generic badge](https://img.shields.io/badge/request_clm-green.svg?style=for-the-badge)](http://collab.lge.com/main/pages/viewpage.action?pageId=2157128981)


## 데이터 준비
1. Single y target(문자열 혹은 실수형 데이터)으로 구성
2. 분류의 경우 추론 데이터의 라벨이 학습 데이터에 반드시 포함되어 있어야 한다
3. 결손치 수가 10% 이하여야 한다.	
4. 데이터 용량:	~100GB
5. 카테고리 변수의 경우 변수 내 카테고리 종류가 총 데이터 수보다 작은가(cardinality 이슈. input, target 모두 해당)	
6. 데이터의 항목명은 무조건 문자열로 이루어져 있으며 숫자로만 구성되면 안된다.	
7. 현재 분류 모델의 경우 binary만 지원합니다.(hotfix 예정)

데이터 명세서 상세한 내용은 [Documentation](http://collab.lge.com/main/pages/viewpage.action?pageId=2082913519)를 참고해주세요.

샘플 데이터 설명: [Bolt Fastening inspection](http://collab.lge.com/main/pages/viewpage.action?pageId=1965439195&src=sidebar)
 


## 주요 기능 소개
- 총 다섯가지 머신러닝 알고리즘에 대해 HPO를 수행하여 가장 성능이 우수한 모델을 도출합니다.
- 모델 평가 단계에서 4-fold 교차검증 방식을 채택하여 검증 데이터 선별에 의한 과최적화를 예방하여 신뢰성 있는 모델을 얻을 수 있습니다.
- 위 과정에서 병렬연산을 수행하여 빠르게 모델을 학습/도출할 수 있습니다.
- 다른 asset과 조합하거나 custom asset과 결합하여 분석가가 원하는 수준의 모델을 개발할 수 있습니다.

상세한 설명은 [Documentation](http://collab.lge.com/main/pages/viewpage.action?pageId=2008478373)을 참고해주세요. 

## Quick Install Guide


```console
git clone http://mod.lge.com/hub/dxadvtech/aicontents/tcr.git 
cd tcr 

conda create -n tcr python=3.10
conda activate tcr 

#jupyter 사용시 ipykernel 추가 필요
#pip install ipykernel
#python -m ipykernel install --user --name tcr 

source install.sh

cd alo
python main.py

```

## Quick Run Guide
- `{config_path}`에 원하는 설정 파일을 지정하여 실행하면 됩니다. default: `config/experimental_plan.yaml`
- 학습 결과 파일 저장 경로: `.train_artifacts/models/train/`
- 추론 결과 파일 저장 경로: `.inference_artifacts/output/inference/`

```
cd alov2
python main.py --config {config_path}
```

## Sample notebook
Jupyter 환경에서 Workflow구동과정을 확인하고 다양한 경우를 실험해볼 수 있습니다 [Sample notebook](http://mod.lge.com/hub/dxadvtech/aicontents/tcr/-/blob/main/TCR_asset_run_template.ipynb)

## 관련 Collab
[AICONTENTS](http://collab.lge.com/main/display/AICONTENTS)

## 요청 및 문의
담당자: yoonji.suh@lge.com
신규 AI Contents나 추가 기능 요청을 등록하시면 검토 후 반영합니다  [Request CLM](http:/링크)


