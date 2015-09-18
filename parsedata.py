import os
import random

import numpy as np
import pandas as pd

data_cache = '/home/scip/Desktop/accel data'
file_list = os.listdir(data_cache)
file_list = [fn for fn in file_list if 'accelerometer' in fn]


def get_vector_norm_chunks(chunk_size=1024):

    for file_name in file_list:
        df = pd.read_csv(os.path.join(data_cache, file_name))
        df.dropna(inplace=True)
        del df['TimeStamp']
        vn = np.sqrt(np.square((df+2**15)/2**16-1).sum(axis=1))
        for k, g in vn.groupby(np.arange(len(vn))//chunk_size):
            yield g


def get_random_vector_norm_chunks(chunk_size=1024):
    # df = None
    # for i in range(5):
    #     file_name = random.sample(file_list, 1)[0]
    #     try:
    #         df = pd.read_csv(os.path.join(data_cache, file_name))
    #         break
    #     except:
    #         pass
    #
    # if df is None:
    #     raise
    npy_file_list = os.listdir('/home/scip/Desktop/stdchunks')
    file_name = random.sample(npy_file_list, 1)[0]
    df = pd.read_csv(os.path.join(data_cache, file_name))
    df.dropna(inplace=True)
    del df['TimeStamp']
    vn = np.sqrt(np.square((df+2**15)/2**16-1).sum(axis=1))
    random_start = random.randint(0, len(vn) - chunk_size)
    return vn[random_start:random_start+chunk_size]


def get_random_cached_chunks():
    # df = None
    # for i in range(5):
    #     file_name = random.sample(file_list, 1)[0]
    #     try:
    #         df = pd.read_csv(os.path.join(data_cache, file_name))
    #         break
    #     except:
    #         pass
    #
    # if df is None:
    #     raise
    std_cache = '/home/scip/Desktop/stdchunks'
    npy_file_list = os.listdir(std_cache)
    file_name = random.sample(npy_file_list, 1)[0]
    return np.load(os.path.join(std_cache, file_name))


if __name__ == '__main__':
    get_vector_norm_chunks()