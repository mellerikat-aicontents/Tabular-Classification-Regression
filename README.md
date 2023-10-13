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
데이터 명세서: [Documentation](http://collab.lge.com/main/pages/viewpage.action?pageId=2082913519)

샘플 데이터 설명: [Iris flower dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set)

[데이터명세서](http://collab.lge.com/main/pages/viewpage.action?pageId=2082913519)
## 주요 기능 소개
.. 대표적인 기능 소개 ()
[Documentation](http://알고리즘 설명 링크)

## Quick Install Guide


```
git cloen http://mod.lge.com/hub/smartdata/ml-framework/alov2.git
git checkout release-1.0
```


```
cd alov2
cd config
git pull http://mod.lge.com/hub/dxadvtech/aicontents/tcr.git

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
담당자: 담당자 이메일
신규 AI Contents나 추가 기능 요청을 등록하시면 검토 후 반영합니다  [Request CLM](http:/링크)


