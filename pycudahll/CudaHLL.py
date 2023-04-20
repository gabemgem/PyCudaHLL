import cupy as cp
from hashlib import sha1
import struct
import numpy as np
import scipy as sp
import math
from helpers import estimate_bias

// TODO: Needs testing
def round_up_to_nearest_32(number) -> int:
    return int(32 * math.ceil(number / 32))


class CudaHLL:
    def __init__(self, p: int = 14, totalThreads: int = 1, cudaDevice: int = 0) -> None:
        self.p = p
        self.num_buckets = 1 << p
        self.bucket_bit_mask = self.num_buckets - 1

        self.alpha = calcAlpha(p, self.num_buckets)

        self.cudaDevice = cudaDevice
        device = cp.cuda.Device(cudaDevice)
        device.use()
        deviceAttributes = device.attributes
        maxThreadsPerBlock = int(deviceAttributes['MaxThreadsPerBlock'])

        self.init_total_threads = totalThreads
        self.num_blocks = math.ceil(float(totalThreads) / maxThreadsPerBlock)
        self.threads_per_block = round_up_to_nearest_32(totalThreads/self.num_blocks)
        self.total_threads = self.num_blocks*self.threads_per_block

        self.registers = cp.zeros((self.total_threads,self.num_buckets), dtype=cp.int8)

    def add(self, hashed_data) -> None:
        """ Adds an array of data to this CudaHLL. 
            hashed_data should be array-like with values already hashed to 64-bit numbers.
            You can use the included 'hash_string()' method to hash values
        """
        vals_per_thread = math.ceil(float(len(hashed_data))/self.total_threads)

        buckets, leading_zeros = find_bucket_and_leading_zeros_kernel(
            hashed_data, 
            self.bucket_bit_mask, 
            self.p
            )
        
        find_max_kernel(
            (self.num_blocks,),
            (self.threads_per_block,),
            (buckets, 
             leading_zeros, 
             vals_per_thread, 
             self.num_buckets, 
             self.registers
            ))

    def merge(self, other):
        """ Merges two CudaHLL objects. Both objects must have the same p-value.
            The result CudaHLL will have the size of the larger initial object, 
            and will use the CUDA device of this object.
        """
        if self.p != other.p:
            raise Exception('CudaHLL Merge: cannot merge CudaHLLs with different p values')
        
        maxInitThreads = self.init_total_threads if self.init_total_threads > other.init_total_threads else other.init_total_threads
        maxThreads = self.total_threads if self.total_threads > other.total_threads else other.total_threads

        newHLL = CudaHLL(self.p, totalThreads=maxInitThreads, cudaDevice=self.cudaDevice)
        if self.total_threads != maxThreads:
            self.registers.resize((maxThreads,self.num_buckets))
        elif other.total_threads != maxThreads:
            other.registers.resize((maxThreads,self.num_buckets))

        newHLL.registers = cp.maximum(self.registers, other.registers)

        return newHLL
    
    def card(self) -> float:
        # get max values for each bucket column and bring data back to host machine
        bucket_maxes = cp.asnumpy(cp.amax(self.registers, axis=0))

        # take zero-count values to the power of 2
        powers_of_two = np.power(2, bucket_maxes)
        
        # take the harmonic mean
        hmean = sp.stats.hmean(powers_of_two)

        # raw estimate
        E = self.alpha * hmean

        # adjusted estimate
        E_prime = E - estimate_bias(E, self.p) if E <= 5*self.num_buckets else E

        # count zero elements
        zero_elements = np.count_nonzero(bucket_maxes==0)

        # check linear counting estimate if there are registers equal to zero
        if zero_elements > 0:
            H = linearCounting(self.num_buckets, zero_elements)
        else:
            H = E_prime

        # use H value if less than a constant threshold
        if H <= threshold(self.p):
            return H
        return E_prime
        


    



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


find_bucket_and_leading_zeros_kernel = cp.ElementwiseKernel(
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
x = cp.array([4,18,9,35], dtype='uint64') # input array of hashed data
num_elements = len(x) # total number of elements to process

p = 2 # number of bucket bits
num_buckets = 1 << p # total number of buckets = 2^p
m = num_buckets-1 # bit mask for bucket bits

threads_per_block = 2 # should generally be a multiple of 32
num_blocks = 1
total_num_threads = num_blocks * threads_per_block
vals_per_thread = math.ceil(num_elements / total_num_threads) # number of values each thread should process

# buckets is an array of bucket indices for each element
# leading zeros is an array of number of leading zeros+1 for each element
buckets, leading_zeros = find_bucket_and_leading_zeros_kernel(x, m, p)

print(buckets)
print(leading_zeros)

# initial outputs per thread
maxes = cp.zeros((total_num_threads,num_buckets), dtype=cp.uint64)
find_max_kernel((num_blocks,), (threads_per_block,), (buckets, leading_zeros, vals_per_thread, num_buckets, maxes))

print(maxes)

total_max = cp.amax(maxes, axis=0)

print(total_max)
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

E = calcAlpha(p, num_buckets) * temp4
E_prime = E - estimate_bias(E, p) if E <= 5*num_buckets else E

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

