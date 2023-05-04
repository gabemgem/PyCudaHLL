import pycudahll.StandardHLL
from pycudahll.ExactCardinality import getExactCardinality
from pycudahll.CudaHLL import CudaHLL, hashDataGPUHLL, addDataGPUHLL, getGPUHLLCardinality

from time import perf_counter_ns

import cupy as cp
from cupyx.profiler import benchmark
import numpy as np
import json

import hyperloglog


# data = [i+512 for i in range(512)]
# hll = CudaHLL(p=9,totalThreads=8, roundThreads=False)
# hll.add(data)
# print('ghll: ', len(hll))

# data2 = ([1]*512)
# hll2 = hyperloglog.HyperLogLog(error_rate=0.06)
# hll2.M = data2
# print('chll: ', len(hll2))

#######################################################################

def get_gpu_avg(results):
    # get running times in seconds
    arr = np.array(results.gpu_times)
    # convert to microseconds
    arr = arr * 1000000

    avg = np.average(arr)
    sd = np.std(arr)
    return f'{avg},{sd}'

def get_cpu_avg(results):
    # get running times in seconds
    arr = np.array(results.cpu_times)
    # convert to microseconds
    arr = arr * 1000000

    avg = np.average(arr)
    sd = np.std(arr)
    return f'{avg},{sd}'


def testGPUHLL(data, p=8, threads=4):
    ghll = CudaHLL(p=p,totalThreads=threads,roundThreads=False)
    getGPUHLLCardinality(ghll, data)

def testGPUHLLInit(p=8, threads=4):
    ghll = CudaHLL(p=p,totalThreads=threads,roundThreads=False)

def testGPUHLLHash(data, p=8, threads=4):
    ghll = CudaHLL(p=p,totalThreads=threads,roundThreads=False)
    hashDataGPUHLL(data)

def testGPUHLLAdd(data, p=8, threads=4):
    ghll = CudaHLL(p=p,totalThreads=threads,roundThreads=False)
    addDataGPUHLL(ghll, data)

def testGPUHLLCard(data, p=8, threads=4):
    ghll = CudaHLL(p=p,totalThreads=threads,roundThreads=False)
    getGPUHLLCardinality(ghll, data)


def testCPUHLL(data, error_rate):
    chll = hyperloglog.HyperLogLog(error_rate=error_rate)
    print('chll: ', StandardHLL.getCPUHLLCardinality(chll, data))


with open('test_output.csv', 'w') as output:
    files = ['shakespeare.csv',
             #'shakespeare_x_2.csv',
             #'shakespeare_x_5.csv',
             #'shakespeare_x_10.csv',
             #'shakespeare_x_50.csv',
             #'shakespeare_x_100.csv',
             #'shakespeare_x_2_new_card.csv',
             #'shakespeare_x_5_new_card.csv',
             #'shakespeare_x_10_new_card.csv',
             #'shakespeare_x_50_new_card.csv',
             #'shakespeare_x_100_new_card.csv'
            ]
    for f in files:
        with open(f, 'r') as file:
            data = file.read().split(',')

            threads = 64
            p = 14
            cpu_precision = 0.01

            # results = benchmark(testGPUHLLHash, (data,p,threads,), n_repeat=10)
            # output.write(f'{f},{get_cpu_avg(results)}\n')

            results = benchmark(testGPUHLL, (data, p, threads,), n_repeat=1)
            # output.write(f'{f},gpu,{get_gpu_avg(results)}\n')

            # results = benchmark(testCPUHLL, (data, cpu_precision,), n_repeat=10)
            # output.write(f'{f},cpu,{get_cpu_avg(results)}\n')

# with open('shakespeare.csv', 'r') as file:
#     data = file.read().split(',')

#     threads = 64
#     p = 14

#     results = benchmark(testGPUHLL, (data, p,threads,), n_repeat=2)
#     print(get_gpu_avg(results))

    ###################################################################
    # test algorithm components
    # print(benchmark(testGPUHLLInit, (p,threads,), n_repeat=40))
    # print(benchmark(testGPUHLLHash, (data, p, threads,), n_repeat=40))
    # print(benchmark(testGPUHLLAdd, (data,p,threads,),n_repeat=40))
    # print(benchmark(testGPUHLLCard, (data,p,threads,),n_repeat=40))


    #################################################################
    # do one test
    # testGPUHLL(data, p=9)
    # testCPUHLL(data, 0.06)
    

    ##################################################################
    # test p-values
    # for i in range(10, 15):
    #     print(f'p-value={i}')
    #     print(benchmark(testGPUHLL, (data,i,), n_repeat=20))


    # cpu_precision = [0.3,0.2,0.15,0.1,0.08,0.06,0.04,0.03,0.02,0.015,0.01,0.007,0.005]
    # for i in range(13):
    #     print(f'p-value={i+4}')
    #     print(benchmark(testCPUHLL, (data,cpu_precision[i],), n_repeat=20))


    ##################################################################    

    # print(benchmark(testGPUHLL1, (data,), n_repeat=40))
    # print(benchmark(testGPUHLL2, (data,), n_repeat=20))

    # ghll = CudaHLL(p=8,totalThreads=4,roundThreads=False)
    # addDataGPUHLL(ghll, data)
    # print(benchmark(testGPUHLL3, (ghll,), n_repeat=20))

    # chll = hyperloglog.HyperLogLog(error_rate=0.08)
    # StandardHLL.addDataCPUHLL(chll, data)
    # print(benchmark(testCPUHLL2, (chll,), n_repeat=20))


# device = cp.cuda.Device()
# print(json.dumps(device.attributes, indent=2))

# with open('shakespeare.csv', 'r') as file:
#     data = file.read().split(',')

#     e_start = perf_counter_ns()

#     exact = getExactCardinality(data)

#     e_end = perf_counter_ns()


#     cpu_start = perf_counter_ns()

#     cpu = getCPUHLLCardinality(data)

#     cpu_end = perf_counter_ns()


#     print('Exact count: ', exact)
#     print('Nanoseconds elapsed for exact: ', e_end-e_start)

#     print('CPU count: ', cpu)
#     print('Nanoseconds elapsed for cpu: ', cpu_end-cpu_start)