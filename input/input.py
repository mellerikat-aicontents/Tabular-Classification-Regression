import pandas as pd
from tqdm import tqdm

def check_columns_equal(df1, df2):
        '''
            Description:
                - 두 데이터프레임을 합치기 전에 colmun별로 name과 dtype이 같은지 확인

            Args:
                - df1 (DataFrame)               : concat할 데이터프레임 1
                - df2 (DataFrame)               : concat할 데이터프레임 2

            Return:
                - Boolean (두 데이터프레임의 colmun이 서로 같은지 여부)
        '''

        try:
            return (df1.columns == df2.columns).all() and (df1.dtypes == df2.dtypes).all()
        except:
            return False



def concat_dataframe(logger, df_list, file_path_list):
        '''
            Description:
                - 모든 dataframe의 column이 같은 경우 하나의 dataframe 생성

            Args:
                - df_list (list)                    : 데이터프레임 리스트
                - file_path_list (list)             : 각 데이터프레임 파일 경로 리스트

            Return:
                - output_df (DataFrame or None)     : (모든 dataframe의 column이 같은 경우) 하나로 통합된 dataframe
                                                    (모든 dataframe의 column이 같지 않은 경우) None #TODO 같지 않은 두 데이터프레임 파일 경로 리스트 리턴하여 구체적 에러 발생
        '''
        logger.debug('Concat DataFrames: START')

        output_df = df_list[0]
        output_path = file_path_list[0]

        for tmp_df, tmp_path in zip(tqdm(df_list[1:]), file_path_list[1:]):
            if check_columns_equal(output_df, tmp_df):
                output_df = pd.concat([output_df, tmp_df])
            else: # 두 데이터프레임을 concat할 수 없는 경우
                logger.debug(f"The columns of the two files are different: \n file1: {output_path} \n file2:{tmp_path}")

        logger.debug('Concat DataFrames: FINISH')
        return output_df

def _read_csv(file_path, encoding):
    return pd.read_csv(file_path, encoding=encoding)
    
def _read_parquet(file_path, PARQUET_ENGINE):
    return pd.read_parquet(file_path, engine=PARQUET_ENGINE)

def read_file(ext2func, file_path, file_type, encoding, logger):
    try:
        df = ext2func[file_type](file_path, encoding)
    except pd.errors.ParserError: # 파일이 데이터프레임 형식을 갖추지 못하는 경우
        logger.debug(f'The file cannot be loaded: {file_path}')
    except Exception as e:
        logger.debug(str(e))
    return df

def filter_files(file_list, ext2format):
    filtered_files = []
    for file_path in file_list:
        for extensions in ext2format.values():
            if any(file_path.endswith(ext) for ext in extensions):
                filtered_files.append(file_path)
                break
    return filtered_files