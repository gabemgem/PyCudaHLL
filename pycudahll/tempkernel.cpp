extern "C" __global__;
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