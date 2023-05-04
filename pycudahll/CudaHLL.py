import cupy as cp
import numpy as np
import math
from pycudahll.helpers import estimate_bias, calcAlpha, linearCounting, threshold, hash_array

def hashDataGPUHLL(data):
    return hash_array(data)

def addDataGPUHLL(ghll, data):
    hashed = hashDataGPUHLL(data)
    ghll.add(hashed)

def getCardGPUHLL(ghll):
    return len(ghll)

def getGPUHLLCardinality(ghll, data):
    addDataGPUHLL(ghll, data)
    return getCardGPUHLL(ghll)



def round_up_to_nearest_32(number) -> int:
    return int(32 * math.ceil(number / 32))

# val_per_thread -> should be number of elements / number of threads
#   in cuda terms, number of elements / (blockSize * gridSize)
find_max_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void find_max(
        const unsigned short* buckets, // array -> a bucket index for each val
        const unsigned char* vals, // array -> a count of leading zeros+1 for each element
        const unsigned int val_per_thread, // number of elements each thread should process
        const unsigned int num_buckets, // total number of buckets
        const unsigned int data_length, // total size of the input array
        unsigned char* max // output array -> size = (num threads) x (num buckets)
    ) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int val_i = tid * val_per_thread;
        int max_i = tid * num_buckets;
        for(int count = 0; count < val_per_thread; count++) {
            int element = val_i+count;
            if (val_i < data_length) {
                max[max_i+buckets[element]] = (max[max_i+buckets[element]] > vals[element]) ?
                    max[max_i+buckets[element]] : vals[element];
            }
        }
    }
''', 'find_max')


find_bucket_and_leading_zeros_kernel = cp.ElementwiseKernel(
    'uint64 x, int64 m, int64 p', 'uint16 bucket, uint8 leading_zeros',
    '''
    bucket = x & m;
    unsigned int w = x >> p;

    int max_width = 64-p;
    //int count = 0;
    for(unsigned int count = 0; count < max_width ; count++) {
        if((1 << count) & w) {
            leading_zeros = count + 1;
            return;
        }
        // count++;
    }
    leading_zeros = 64;
    '''
)


class CudaHLL:
    def __init__(self, p: int = 14, totalThreads: int = 1, cudaDevice: int = 0, roundThreads: bool = True) -> None:
        """Initialize an HLL data structure.
            Args:
                p = 'precision' or the number of bits used for bucket indices - can be 4 to 16
                totalThreads = number of threads to use for aggregation (default = 1, try 64)
                cudaDevice = which GPU to use (default = 0 which is the main GPU on the system)
                roundThreads = boolean value to round totalThreads up to the nearest 32
                    Look up 'thread warps' for more info
        """
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
        self.round_threads = roundThreads
        if roundThreads:
            self.threads_per_block = round_up_to_nearest_32(totalThreads/self.num_blocks)
        else:
            self.threads_per_block = math.ceil(totalThreads/self.num_blocks)
        self.total_threads = self.num_blocks*self.threads_per_block

        self.registers = cp.zeros((self.total_threads,self.num_buckets), dtype=cp.uint8)

    def merge(self, other):
        """ Merges two CudaHLL objects. Both objects must have the same p-value.
            The result CudaHLL will have the size of the larger initial object, 
            and will use the CUDA device of this object.
        """
        if self.p != other.p:
            raise Exception('CudaHLL Merge: cannot merge CudaHLLs with different p values')
        
        maxInitThreads = self.init_total_threads if self.init_total_threads > other.init_total_threads else other.init_total_threads
        maxThreads = self.total_threads if self.total_threads > other.total_threads else other.total_threads

        newHLL = CudaHLL(self.p, totalThreads=maxInitThreads, cudaDevice=self.cudaDevice, roundThreads=self.round_threads)
        if self.total_threads != maxThreads:
            self.registers.resize((maxThreads,self.num_buckets))
        elif other.total_threads != maxThreads:
            other.registers.resize((maxThreads,self.num_buckets))

        newHLL.registers = cp.maximum(self.registers, other.registers)

        return newHLL
    

    def add(self, hashed_data) -> None:
        """ Adds an array of data to this CudaHLL. 
            hashed_data should be array-like with values already hashed to 64-bit numbers.
            You can use the included 'hash_string()' method to hash values.
            The data should still be on the host (i.e. not a cupy array)
        """
        data_length = len(hashed_data)
        vals_per_thread = math.ceil(float(data_length)/self.total_threads)
        # print(vals_per_thread)

        gpu_data = cp.array(hashed_data, dtype=cp.uint64)

        buckets, leading_zeros = find_bucket_and_leading_zeros_kernel(
            gpu_data, 
            self.bucket_bit_mask, 
            self.p
            )
        
        # print(buckets)
        
        find_max_kernel(
            (self.num_blocks,),
            (self.threads_per_block,),
            (buckets, 
             leading_zeros, 
             vals_per_thread, 
             self.num_buckets, 
             data_length,
             self.registers
            ))

    
    def card(self) -> float:
        # Return the current cardinality estimate

        # get max values for each bucket column and bring data back to host machine
        bucket_maxes = cp.asnumpy(cp.amax(self.registers, axis=0))

        # take zero-count values to the power of 2
        powers_of_two = np.power(2.0, (bucket_maxes*-1))
        
        # take the harmonic mean
        hmean = (self.num_buckets**2) / np.sum(powers_of_two)

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
    
    def __len__(self):
        # Return the current cardinality estimate rounded to the nearest integer
        return round(self.card())
