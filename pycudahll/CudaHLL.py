import cupy as cp
from hashlib import sha1
import struct
import numpy as np

def hash_string(s):
    s = s.encode('utf-8')
    return struct.unpack('!Q', sha1(s).digest()[:8])[0]

find_max_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void find_max(
        const unsigned long long* indices, 
        const unsigned long long* vals, 
        const unsigned long val_per_thread,
        const unsigned long indices_range,
        unsigned long long* max) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int val_i = tid * val_per_thread;
        int max_i = tid * indices_range;
        for(int count = 0; count < val_per_thread; count++) {
            int element = val_i+count;
            max[max_i+indices[element]] = (max[max_i+indices[element]] > vals[element]) ?
                max[max_i+indices[element]] : vals[element];
            //max[max_i+count] = vals[element];
        }
        //max[max_i] = max_i;
    }
''', 'find_max')


kernel = cp.ElementwiseKernel(
    'uint64 x, uint64 m, uint64 p', 'uint64 index, uint64 leading_zeros',
    '''
    index = x & m;
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

x = cp.array([4,18,9,35], dtype='uint64')
index, leading_zeros = kernel(x, 3, 2)

print(f'{7:b}')
print(f'{8:b}')
print(index)
print(leading_zeros)

maxes = cp.zeros((4,4), dtype=cp.uint64)
vals_per_thread = 1
indices_range = 4
find_max_kernel((2,), (2,), (index, leading_zeros, vals_per_thread, indices_range, maxes))

print(maxes)



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

