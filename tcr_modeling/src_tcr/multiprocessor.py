import traceback
import ctypes

import multiprocessing
from multiprocessing import Manager, Process, Semaphore
from multiprocessing import Value, Lock

from tqdm import tqdm

class Multiprocessor:
    '''
    Description:
        - 특정 함수를 multiprocessing 하기위한 custom class
        - 첫 하나의 subprocess에서 error가 발생할 시 더 이상 process을 할당하지 않고 raise Exception

    Parameter:
        - num_processes (int)           : 사용할 Process 개수

    Attribute:
        - target_function (function)    : Multiprocessing을 적용할 function
        - num_processes (int)           : 사용할 Process 개수
        - manager (class)               : Multiprocessing Manager
        - error_occured (bool)          : Error 발생 여부 Flag
        - lock (class)                  : Multiprocessing Lock (Ensure 1 process at a time)
        - progress (int)                : Main process에서 progress 체크할 수 있는 인자
    
    Method:
        - set_target_function           : Child process에 할당할 function을 load
        - worker                        : target function의 wrapper, error 발생 시 traceboack msg를 return
        - run                           : Multiprocessing 수행
        
    '''

    def __init__(self, num_processes=None):
        self.target_function = None
        self.num_processes = num_processes or 3
        self.manager = Manager()
        self.error_occurred = self.manager.Value(ctypes.c_bool, False)
        self.lock = Lock()
        self.progress = self.manager.Value('i', 0)  # shared progress

    def set_target_function(self, target_function):
        '''
        Description:
            - target function을 설정
        
        Args:
            - target_function (function)    : Multiprocessing을 돌릴 function

        '''

        self.target_function = target_function

    def worker(self, kwargs, error_occurred, results, se, progress):
        '''
        Description:
            - target function의 wrapper
            - run에서 call되어 multiprocessing
        
        Args:
            - args (tuple)              : target_function의 arg
            - error_occurred (bool)     : Child process의 error 발생 여부 Flag
            - results (list)            : Child process의 결과 저장 list

        '''

        if not error_occurred.value:
            try:
                result = self.target_function(**kwargs)

                with self.lock:
                    results.append(result)
                    progress.value += 1  # increment progress
                    se.release()

            except Exception:
                traceback_msg = traceback.format_exc()
                print(f"An error occurred in the subprocess: {traceback_msg}")
                with self.lock:
                    error_occurred.value = True
                se.release()

    def run(self, list_of_kwargs):
        '''
        Description:
            - Multiprocessing 수행
        
        Args:
            - list_of_args (list)   : target function의 args list

        Return:
            - results (list)        : Multiprocessing 결과 list

        '''

        se = Semaphore(self.num_processes)
        results = self.manager.list()
        processes = []

        with tqdm(total=len(list_of_kwargs)) as pbar:
            for kwargs in list_of_kwargs:
                if self.error_occurred.value:
                    break  

                se.acquire()
                process = Process(target=self.worker, args=(kwargs, self.error_occurred, results, se, self.progress))
                processes.append(process)
                process.start()

                pbar.n = self.progress.value  # update progress bar
                pbar.refresh()

            # Collect results
            for process in processes:
                process.join()

                pbar.n = self.progress.value  # update progress bar
                pbar.refresh()

                if self.error_occurred.value:
                    for p in processes:
                        p.terminate()
                    raise Exception

            if self.error_occurred.value:
                raise Exception(f'An error occurred in one subprocess, causing the remaining processes to terminate.')

        return list(results)
