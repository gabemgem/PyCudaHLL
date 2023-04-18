import cupy as cp
from hashlib import sha1
import struct
import numpy as np

def hash_string(s):
    s = s.encode('utf-8')
    return struct.unpack('!Q', sha1(s).digest()[:8])[0]



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

x = cp.array([7, 8], dtype='uint64')
index, leading_zeros = kernel(x, 1, 1)

print(f'{7:b}')
print(f'{8:b}')
print(index)
print(leading_zeros)



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

