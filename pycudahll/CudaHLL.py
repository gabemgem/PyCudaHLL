import cupy as cp
from hashlib import sha1
import struct
import numpy as np
import scipy as sp
import math

def hash_string(s):
    s = s.encode('utf-8')
    return struct.unpack('!Q', sha1(s).digest()[:8])[0]

# val_per_thread -> should be number of elements / number of threads
#   in cuda terms, number of elements / (blockSize * gridSize)
find_max_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void find_max(
        const unsigned long long* buckets, // array -> a bucket index for each val
        const unsigned long long* vals, // array -> a count of leading zeros+1 for each element
        const unsigned long val_per_thread, // number of elements each thread should process
        const unsigned long num_buckets, // total number of buckets
        unsigned long long* max // output array -> size = (num threads) x (num buckets)
    ) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int val_i = tid * val_per_thread;
        int max_i = tid * num_buckets;
        for(int count = 0; count < val_per_thread; count++) {
            int element = val_i+count;
            max[max_i+buckets[element]] = (max[max_i+buckets[element]] > vals[element]) ?
                max[max_i+buckets[element]] : vals[element];
        }
    }
''', 'find_max')


kernel = cp.ElementwiseKernel(
    'uint64 x, uint64 m, uint64 p', 'uint64 bucket, uint64 leading_zeros',
    '''
    bucket = x & m;
    unsigned int w = x >> p;

    for(unsigned int count = 0; count < 64; count++) {
        if((1 << count) & w) {
            leading_zeros = count + 1;
            return;
        }
    }
    leading_zeros = 64;
    '''
)


############################################################################
# x = cp.array([4,18,9,35], dtype='uint64') # input array of hashed data
# num_elements = len(x) # total number of elements to process

# m = 3 # bit mask for bucket bits
p = 2 # number of bucket bits
num_buckets = 1 << p # total number of buckets = 2^p

# threads_per_block = 2 # should generally be a multiple of 32
# num_blocks = 1
# total_num_threads = num_blocks * threads_per_block
# vals_per_thread = math.ceil(num_elements / total_num_threads) # number of values each thread should process

# # buckets is an array of bucket indices for each element
# # leading zeros is an array of number of leading zeros+1 for each element
# buckets, leading_zeros = kernel(x, m, p)

# print(buckets)
# print(leading_zeros)

# # initial outputs per thread
# maxes = cp.zeros((total_num_threads,num_buckets), dtype=cp.uint64)
# find_max_kernel((num_blocks,), (threads_per_block,), (buckets, leading_zeros, vals_per_thread, num_buckets, maxes))

# print(maxes)

# total_max = cp.amax(maxes, axis=0)

# print(total_max)
################################################################################

def calcAlpha(p, numBuckets):
    if p < 4 or p > 18:
        raise ValueError(f'p={p} should be in the range [4 : 18]')
    
    if p == 4:
        return 0.673
    if p == 5:
        return 0.697
    if p == 6:
        return 0.709
    return 0.7213 / (1 + (1.079/numBuckets))

def estimateBias(E, p):
    pass

def linearCounting(numBuckets, zeroElements):
    # returns the linear counting cardinality estimate
    return numBuckets * math.log2(numBuckets/zeroElements)

def threshold(p):
    thresh = {
        4: 10,
        5: 20,
        6: 40,
        7: 80,
        8: 220,
        9: 400,
        10: 900,
        11: 1800,
        12: 3100,
        13: 6500,
        14: 11500,
        15: 20000,
        16: 50000,
        17: 120000,
        18: 350000
    }
    if p in thresh:
        return thresh[p]
    return 0

temp = cp.array([1,1,1,1], dtype='uint64')
M = cp.asnumpy(temp)
temp3 = np.power(2, M)
temp4 = sp.stats.hmean(temp3)

E = alpha * temp4
E_prime = E - estimateBias(E, p) if E <= 5*num_buckets else E

zero_elements = np.count_nonzero(M==0)
if zero_elements > 0:
    H = linearCounting(num_buckets, zero_elements)
else:
    H = E_prime

if H <= threshold(p):
    output = H
else:
    output = E_prime


print(temp4)

# x = hash_string('hello')
# print(x)
# print(type(x))
# print(hex(x))

# temp = np.array(['hello', 'hello', 'how', 'are'])
# vhash = np.vectorize(hash_string)

# print(temp)

# hashed = vhash(temp)
# gpuarr = cp.array(hashed)
# # vhash = cp.vectorize(hash_string)

# # hashed = vhash(temp)

# print(hashed)
# print(hashed.dtype)

