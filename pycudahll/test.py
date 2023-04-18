from StandardHLL import getCPUHLLCardinality
from ExactCardinality import getExactCardinality

from time import perf_counter_ns

with open('shakespeare.csv', 'r') as file:
    data = file.read().split(',')

    e_start = perf_counter_ns()

    exact = getExactCardinality(data)

    e_end = perf_counter_ns()


    cpu_start = perf_counter_ns()

    cpu = getCPUHLLCardinality(data)

    cpu_end = perf_counter_ns()


    print('Exact count: ', exact)
    print('Nanoseconds elapsed for exact: ', e_end-e_start)

    print('CPU count: ', cpu)
    print('Nanoseconds elapsed for cpu: ', cpu_end-cpu_start)