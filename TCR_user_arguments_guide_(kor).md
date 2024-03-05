# TCR User Arguments Guide (KOR) -!! 현재 수정 중
## User arguments란? 
experimental_plan.yaml에 작성하는 데이터 분석을 위한 각 asset 별 설정 항목입니다. experimental_plan.yaml에 보여지는 **필수 arguments**와 유저가 이 가이드를 보고 추가하는 **Custom arguments**로 구성됩니다. 

1. 필수 arguments
    - 필수 arguments는 experimental_plan.yaml에 보여지는 기본 arguments 입니다. 
    - 대부분의 필수 arguments는 default 값이 내장되어 있습니다. default 값이 있는 arguments의 경우에는 유저가 별도로 값을 설정하지 않아도 동작합니다.
    - 필수 arguments에 공란이 있는 경우 유저가 필수로 값을 설정해주어야 합니다. 
2. Custom arguments
    - Custom arguments는 experimental_plan.yaml에 보여지지 않지만, asset에서 제공하고 있는 기능으로 사용자가 experimental_plan.yaml에 기입하여 사용할 수 있습니다. 
    - 각 asset 별 필수 arguments 밑에 추가하여 사용합니다.    
  
TCR의 pipeline은 **Input-Readiness-Preprocess-Train-Output** 순으로 구성되어 있으며 각 asset 별로 지원하는 기능이 다르며, user arguments도 다르게 구성되어 있습니다. 

다음은 asset 별 user arguments 사용가이드 입니다. 아래 가이드를 보고 asset 별 필요한 기능에 해당하는 user arguments를 experimental_plan.yaml에 넣어 사용해주세요

아래는 train pipeline의 user arguments입니다. inference시에는 train에서 사용한 arguments값을 가져옵니다.

&nbsp;
***
## Asset 별 User Arguments 설명
### Asset - user arguments
목차 만드는 중
<!-- - [Input asset](##input-asset)
- [Readiness asset](##readiness-asset)
- [Input asset](##input-asset) -->

***
## Input asset
### Default arguments
#### 1. Argument 명: **file_type** (필수)
- input data의 파일 확장자를 입력합니다. 
- default 값: csv
- 사용 가능한 값:
    - csv: csv 파일
    - parquet: parquet 파일
- 사용 방법(아래 copy하여 experimental_plan.yaml에 추가합니다.)
    - file_type: csv

#### 2. Argument 명: **encoding** (필수)
- input data의 encoding type을 입력합니다.. 
- default 값: utf-8
- 사용 가능한 값: utf-8, cp949 등, python에서 지원하는 encoding 값을 넣어주세요
https://docs.python.org/3/library/codecs.html#standard-encodings
- 사용 방법(아래 copy하여 experimental_plan.yaml에 추가합니다.)
    - encoding: utf-8
***
## Readiness asset - user arguments
### Default arguments
#### 1. Argument 명: **x_columns** (필수)
- 학습에 필요한 x 컬럼 명을 입력합니다. 
- default 값: -
- 사용 가능한 값:
    - list에 컬럼 명을 담아서 사용합니다.
- 사용 방법(아래 copy하여 experimental_plan.yaml에 추가합니다.)
    - file_type: [col1, col2]

#### 2. Argument 명: **y_column** (필수)
- 학습에 필요한 y 컬럼 명을 입력합니다.
- default 값: -
- 사용 가능한 값:
    - string 컬럼 명을 입력합니다.
- 사용 방법(아래 copy하여 experimental_plan.yaml에 추가합니다.)
    - y_column: target

#### 3. Argument 명: **column_types** (필수)
- Default 값인 auto를 사용하면 자동으로 readiness asset에서 컬럼의 유형이 categorical인지 numeric인지 구분해줍니다.
- readiness에서 자동 분류 한 categorical/numeric 컬럼 리스트를 수정하고 싶을 때 사용합니다. 
- 사용자가 입력한 categorical_columns는 무조건 categorical 컬럼으로, numeric_columns는 numeric 컬럼으로 분류됩니다. 입력하지 않은 컬럼은 자동으로 readiness가 categorical/numeric 분류를 진행합니다.
- default 값: auto
- 사용 가능한 값:
    - dictionary에 아래와 같이 작성합니다.
    - {categorical_columns: [categorical 컬럼 리스트], numeric_columns: [numeric 컬럼 리스트]}
- 사용 방법(아래 copy하여 experimental_plan.yaml에 추가합니다.)
    - {categorical_columns: [col1, col2], numeric_columns: [col3, col4]}

#### 4. Argument 명: **task_type** (필수)
- Solution 과제의 유형(classification/regression)을 입력합니다.
- default 값: classification
- 사용 가능한 값:
    - classification
    - regression
- 사용 방법(아래 copy하여 experimental_plan.yaml에 추가합니다.)
    - task_type: classification

#### 5. Argument 명: **drop_x_columns** (Custom)
- drop_x_columns를 사용하면 전체 input data의 컬럼에서 drop_x_column, groupkey_columns, y_column에 있는 컬럼을 제외한 나머지 컬럼을 학습 데이터로 사용합니다.
- x_columns와 동시에 사용할 수 없습니다.(둘 중 하나만 yaml에 기입해야함) 
- default 값: -
- 사용 가능한 값:
    - list에 drop할 컬럼 명을 담아서 사용합니다.
- 사용 방법(아래 copy하여 experimental_plan.yaml에 추가합니다.)
    - drop_x_columns: [co1, col2]

#### ~~6. Argument 명: **groupkey_columns** (Custom)~~ !!현재 에러 있음 사용x
- groupkey 컬럼을 입력하여 해당 컬럼의 value 값을 기준으로 dataframe을 grouping 합니다.
- default 값: -
- 사용 가능한 값:
    - list에 grouping에 쓸 컬럼 명을 입력합니다.
- 사용 방법(아래 copy하여 experimental_plan.yaml에 추가합니다.)
    - groupkey_columns: [col1, col2]

#### 7. Argument 명: **min_rows** (Custom)
- 학습 시 필요한 최소 row 수를 지정합니다. 
- default 값: 
    - classification) label 별 30개
    - regression) 100개
- 사용 가능한 값:
    - 정수(int)
- 사용 방법(아래 copy하여 experimental_plan.yaml에 추가합니다.)
    - min_rows: 50

#### 8. Argument 명: **cardinality** (Custom)
- categorical/numeric 컬럼 자동 분류 기능을 사용할 때 categorical_columns가 갖추어야 하는 cardinality 조건입니다.   
- cardinality가 입력 값보다 같거나 작아야 categorical_columns로 판단합니다.
- default 값: 50
- 사용 가능한 값:
    - 정수(int)
- 사용 방법(아래 copy하여 experimental_plan.yaml에 추가합니다.)
    - cardinality: 100

#### 9. Argument 명: **num_cat_split** (Custom)
- categorical/numeric 컬럼 자동 분류 기능을 사용할 때, 자주 등장하는 top N 데이터의 numeric/object 여부를 검사하여 컬럼을 분류합니다.
- num_cat_split은 N값을 지정합니다.
- default 값: 10
- 사용 가능한 값:
    - 정수(int)
- 사용 방법(아래 copy하여 experimental_plan.yaml에 추가합니다.)
    - num_cat_split: 20

#### 10. Argument 명: **ignore_new_category** (Custom)
- inference시 categorical x 컬럼에 학습에 사용되지 않은 카테고리 값이 들어왔을 때 처리합니다.
- default 값: False
- 사용 가능한 값:
    - False: error 발생
    - True: warning만 발생
    - float값 n입력
        - train에서 사용하지 않은 값 비율이 전체 데이터의 n 미만일 경우에는 warning
        - train에서 사용하지 않은 값 비율이 전체 데이터의 n 미만일 경우에는 error 발생
- 사용 방법(아래 copy하여 experimental_plan.yaml에 추가합니다.)
    - ignore_new_category: 0.3
***
## Preprocess asset - user arguments
### Default arguments
#### 1. Argument 명: **categorical_encoding** (Custom)
- categorical_encoding은 category 컬럼에 적용할 인코딩 방법론을 지정합니다. {방법론: 값}의 dictionary형태로 input을 받습니다.
- 현재 onehot과 label을 지원합니다.
- onehot: onehot encoding
- label: label encoding 
- default 값: {onehot: all}	
- 사용 가능한 값:
    - {방법론: 값}의 dictionary 형태로 작성합니다. 값에는 방법론을 적용할 [컬럼 리스트] 또는 all을 입력할 수 있습니다. 이 때, all은 categorical 컬럼 전체입니다.  
    - {onehot: [onehot 방법론을 적용할 컬럼 명], label: [label 방법론을 적용할 컬럼 명]}
        - onehot: onehot encoding
        - label: label encoding
- 사용 방법(아래 copy하여 experimental_plan.yaml에 추가합니다.)
    - categorical_encoding: {onehot: [col1,col2], label: [col1]}

#### 2. Argument 명: **handle_missing** (Custom)
- handle_missing은 category, numeric 컬럼에 적용할 결측치 처리 방식을 지정합니다. {방법론: 값}의 dictionary형태로 input을 받습니다.
- default 값: 
    - 결측치 10% 이하: {drop: all}
    - 결측치 10% 초과: {frequent: category_all, median: numeric_all}
- 사용 가능한 값:
    - {방법론: [방법론을 적용할 컬럼 명]}
    - categorical 컬럼에 적용가능한 방법론: frequent
        - [컬럼명 list] or all 입력. all은 categorical 컬럼 전체 입니다.
    - numeric 컬럼에 적용가능한 방법론: mean, median, interpolation
        - [컬럼명 list] or all 입력. all은 numeric 컬럼 전체 입니다.
    - 전체 컬럼에 적용가능한 방법론: drop, fill_숫자
        - [컬럼명 list] or all, numeric_all, categorical_all
        - all은 컬럼 전체, numeric_all은 numeric 컬럼 전체, categorical_all은 categorical 컬럼 전체입니다.
- 사용 방법(아래 copy하여 experimental_plan.yaml에 추가합니다.)
    - handle_missing: {frequent: [cat_col1, cat_col2], median: [num_col1, num_col2], drop: all}

#### 3. Argument 명: **numeric_outlier** (Custom)
- numeric 컬럼에 적용할 outlier 제거 방법을 선택합니다.
- default 값: -
- 사용 가능한 값:
    - {방법론: 값}의 dictionary 형태로 작성합니다. 값에는 방법론을 적용할 [컬럼 리스트] 또는 all을 입력할 수 있습니다. 이 때, all은 numeric 컬럼 전체입니다.  
    - normal: 현재 데이터 분포에서 3sigma 넘는 이상치를 삭제
- 사용 방법(아래 copy하여 experimental_plan.yaml에 추가합니다.)
    - numeric_outlier: {normal: [num_col1, num_col2]}

#### 4. Argument 명: **numeric_scaler** (Custom)
- numeric 컬럼에 적용할 scaling방법을 선택합니다.
- default 값: -
- 사용 가능한 값:
    - list에 컬럼 명을 담아서 사용합니다.
- 사용 방법(아래 copy하여 experimental_plan.yaml에 추가합니다.)
    - numeric_scaler: [col1, col2]
***
## Train asset
### Default arguments
#### 1. Argument 명: **pos_label** (필수)
- 특정 y라벨 값을 지정하여 해당 value 값을 기준으로 HPO를 진행하고 score를 구합니다. 
- default 값: _major
- 사용 가능한 값:
    - _major: y label majority class를 지정합니다.
    - _minor: y label minor class를 지정합니다.
    - y컬럼의 라벨 값
- 사용 방법(아래 copy하여 experimental_plan.yaml에 추가합니다.)
    - pos_label: _major

#### 2. Argument 명: **evaluation_metric** (필수)
- 평가 metric을 선택합니다.
- default 값: auto
- 사용 가능한 값:
    - auto: classification일 때는 accuracy, regression일 때는 mse
    - accuracy, f1, recall, precision (classification에 적용 가능)
    - mse, r2, mae, rmse (regression에 적용 가능)
- 사용 방법(아래 copy하여 experimental_plan.yaml에 추가합니다.)
    - evaluation_metric: accuracy

#### 3. Argument 명: **shapley_value** (필수)
- shapley value를 계산하여 출력할지 결정합니다.
- default 값: False
- 사용 가능한 값:
    - True: shapley value를 출력합니다.
    - False: shapley value를 출력하지 않습니다.
- 사용 방법(아래 copy하여 experimental_plan.yaml에 추가합니다.)
    - shapley_value: True

#### 4. Argument 명: **model_list** (Custom)
- HPO로 비교할 모델을 선택합니다. 유저가 새롭게 추가한 모델에 대해서도 HPO 리스트에 추가하기 위해서는, 파일의 요약어를 model_list에 추가합니다.
- hpo_setting에 값을 입력하더라도, model_list에 해당 모델 명이 없으면 HPO에 추가되지 않습니다.
- default 값: [rf, gbm, lgbm, cb] (현재 ngb x)
- 사용 가능한 값:
    - rf: random forest
    - gbm: gradient boosting machine
    - lgbm: light gradient boosting machine
    - cb: catboost
- 사용 방법(아래 copy하여 experimental_plan.yaml에 추가합니다.)
    - model_list: [rf, gbm]

#### 5. Argument 명: **data_split** (Custom)
- HPO시 train/validation set 구성 방법론을 선택합니다. 
- default 값: {method: 'cross_validation', options: 5}
- 사용 가능한 값: 
    - {method: 방법론 명, options: 필요한 값}의 dictionary 형태로 입력합니다.
    - cross_validation: cross validation 방법론을 적용합니다.
        - options: kfold 값을 입력합니다.  
    - train_test: 입력한 options값을 기준으로 train/test를 split 합니다. 
        - options: test 데이터의 비율을 입력합니다. 
- 사용 방법(아래 copy하여 experimental_plan.yaml에 추가합니다.)
    - data_split: 0.3

#### 6. Argument 명: **hpo_settings** (Custom)
- HPO에 포함 된 모델에 대해 parameter를 변경합니다.
- default 값: -
- 사용 가능한 값:
    - {모델1 : {parameter: value}, 모델2 : {parameter: value}}와 같이 작성합니다
    - ex) {rf: {max_depth: [100, 500], n_estimators: [300, 500], min_sample_leaf: 3, tcr_param_mix: ‘one_to_one’}}
    - 각 parameter 마다 search range를 list로 입력합니다.
    - tcr_param_mix:
        - one_to_one: 각 element끼리 1:1 대응하여 HPO를 진행합니다. parameter value가 list인 경우에는 elements 수 가 동일해야 합니다.
        - all: 입력한 parameter list의 모든 경우의 수 case를 계산하여 HPO를 진행합니다.
- 사용 방법(아래 copy하여 experimental_plan.yaml에 추가합니다.)
    - hpo_settings: {rf: {max_depth: [100, 500], n_estimators: [300, 500], min_sample_leaf: 3, tcr_param_mix: ‘one_to_one’}}

#### 7. Argument 명: **shapley_sampling** (Custom)
- shapley_value 값이 True일 때, 모든 데이터에 대해 sampling하지 않고 일부 만 sampilng 하여 shapley value를 출력할 수 있습니다. 데이터가 많을 때 모든 값에 대해 shapley value 값을 구하면 학습 시간이 오래걸립니다. 
- default 값: 10000
- 사용 가능한 값:
    - 0< arg < 1 소수 값(float): 해당 비율 만큼 sampling 합니다. ex) 0.8 – 80% 샘플링
    - arg = 1: 모든 값에 대해 sampling 합니다.
    - arg > 1: 입력 값 만큼 sampling 합니다. ex) 10000 – 만개 sampling
- 사용 방법(아래 copy하여 experimental_plan.yaml에 추가합니다.)
    - shapley_sampling: 10000