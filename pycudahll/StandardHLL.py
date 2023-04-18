import hyperloglog

hll = hyperloglog.HyperLogLog(0.01) # accept 1% counting error
print(hll.m, hll.p)

# with open('shakespeare.csv', 'r') as file:
#     data = file.read().split(',')
#     for val in data:
#         hll.add(val)

#     print(len(hll))

def getCPUHLLCardinality(data):
    for val in data:
        hll.add(val)
    return len(hll)