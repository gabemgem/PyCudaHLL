import cupy as cp
from hashlib import sha1
import struct
import numpy as np

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


x = cp.array([4,18,9,35], dtype='uint64') # input array of hashed data
num_elements = len(x) # total number of elements to process

m = 3 # bit mask for bucket bits
p = 2 # number of bucket bits
num_buckets = 1 << p # total number of buckets = 2^p

threads_per_block = 2 # should generally be a multiple of 32
num_blocks = 2
total_num_threads = num_blocks * threads_per_block
vals_per_thread = num_elements / total_num_threads # number of values each thread should process

# buckets is an array of bucket indices for each element
# leading zeros is an array of number of leading zeros+1 for each element
buckets, leading_zeros = kernel(x, m, p)

print(f'{7:b}')
print(f'{8:b}')
print(buckets)
print(leading_zeros)

# initial outputs per thread
maxes = cp.zeros((total_num_threads,num_buckets), dtype=cp.uint64)
find_max_kernel((num_blocks,), (threads_per_block,), (buckets, leading_zeros, vals_per_thread, num_buckets, maxes))

print(maxes)

total_max = cp.amax(maxes, ax)



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

