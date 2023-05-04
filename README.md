# PyCudaHLL

This is a GPU accelerated implementation of HyperLogLog using the CuPy library. It was created for the class "Algorithmic Techniques for Taming Big Data" at the department of Computing and Data Science at Boston University.

## Using the Code

To use this code, you can either get the library from PyPI or build it from source.

### Get from PyPI (Recommended)

- Install using pip: `pip install pycudahll`
- In your code, import the library: `from pycudahll.CudaHLL import CudaHLL`

### Building from Source

- Clone the repository
- Install dependencies: `poetry install`
- In your code, import the library: `from pycudahll.CudaHLL import CudaHLL`
- See `test.py` for examples. (Note: `test.py` is most likely in a broken state, but should give you an idea of how to use the library.)

## API

The main class of the library is CudaHLL. It can be imported in your code with:
```python
from pycudahll.CudaHLL import CudaHLL
```

CudaHLL also includes a helper function to hash data to use with the main class:
```python
from pycudahll.CudaHLL import hashDataGPUHLL
```

A short example of how to use the library is as follows:
```python
from pycudahll.CudaHLL import CudaHLL, hashDataGPUHLL

with open('data.csv', 'r') as file:
    data = file.read().split(',')
    hashedData = hashDataGPUHLL(data)

    threads = 64
    p = 14
    cudaDevice = 0 # optional
    roundThreads = True # optional
    hll = CudaHLL(p, threads, cudaDevice, roundThreads)

    hll.add(hashedData)
    print(hll.card()) # print unrounded cardinality estimate
    print(len(hll)) # print rounded cardinality estimate
```

## Future Work

- Parallelize Hashing
  - Will need to switch to cuDF or only allow numerical data

## Test Data

Text of Shakespeare plays obtained from https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt. Original text can be found in t8.shakespeare.txt and the modified text can be found in shakespeare.csv.

Total number of items = 899300
Exact cardinality = 34065
