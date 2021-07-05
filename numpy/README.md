### Importing library and basic array


```python
import numpy as np

a = np.array([[1,2],[3,4]], dtype="int16")
print(a)
```

    [[1 2]
     [3 4]]


### Printing characteristcs of an Array


```python
# number of dimentions
a.ndim
```




    2




```python
# shape of the array - this is a tuple with dimention's size
a.shape
```




    (2, 2)




```python
# basic data type
a.dtype
```




    dtype('int8')




```python
# number of elements in the array
a.size
```




    4




```python
# number of bytes used in the array
a.nbytes
```




    8




```python
# number of bytes of data type
a.itemsize
```




    2



### Accessing and updating elements in array


```python
# accessing one specific element
a[1,1]
```




    4




```python
# accessing first row
a[0, :]
```




    array([1, 2], dtype=int16)




```python
# Fetching first column
a[: , 0]
```




    array([1, 3], dtype=int16)




```python
# Updating a specific column
a[: , 1] = (99, 99)
print(a)
```

    [[ 1 99]
     [ 3 99]]


### Generating specific matrixes


```python
# creates arrays of 1s
np.ones((3, 3), dtype="int8")
```




    array([[1, 1, 1],
           [1, 1, 1],
           [1, 1, 1]], dtype=int8)




```python
# creates arrays of 0s
np.zeros((3,3), dtype="int8")
```




    array([[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]], dtype=int8)




```python
# creates arrays fileld with specific value
np.full((2,2), 99)
```




    array([[99, 99],
           [99, 99]])




```python
# Generate random float values arrays
np.random.rand(2,2)
```




    array([[0.04652306, 0.46851432],
           [0.081302  , 0.39715904]])




```python
# Generate random int values arrays 
np.random.randint(7, size=(2,2))
```




    array([[1, 6],
           [6, 2]])




```python
# Generate identity matrixes
np.identity(5)
```




    array([[1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 1.]])



### Transformations and Mathematical operations


```python
a * 2
```




    array([[2, 4],
           [6, 8]], dtype=int16)




```python
a / 2
```




    array([[0.5, 1. ],
           [1.5, 2. ]])




```python
a ** 2
```




    array([[ 1,  4],
           [ 9, 16]], dtype=int16)




```python
a + 5
```




    array([[6, 7],
           [8, 9]], dtype=int16)




```python
# reshaping an array
a.reshape((4))
```




    array([1, 2, 3, 4], dtype=int16)




```python
# repeat the content of an specific array
b = np.array([[1,2,3]])
np.repeat(b, 3, axis=0)
```




    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]])




```python
b = np.array([1,2,3])
c = np.array([4,5,6])
np.vstack([b,c])
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
np.hstack([b,c])
```




    array([1, 2, 3, 4, 5, 6])



### Statistics


```python
# or a.max() - returns max value inside Array
np.max(a)
```




    4




```python
# or a.min() - returns min value inside Array
np.min(a)
```




    1




```python
# returns an array with max value of each internal Array
np.max(a, axis=1)
```




    array([2, 4], dtype=int16)




```python
# returns a summation of all elements in an array
np.sum(a)
```




    10




```python
# returns summation of rows
np.sum(a, axis=1)
```




    array([3, 7])




```python
# returns average
np.average(a)
```




    2.5



### Linear Algebra


```python
# Matrix multiplication
a = np.random.randint(7, size=(2,3))
b = np.random.randint(7, size=(3,2))
print(a)
print(b)
np.matmul(a,b)
```

    [[6 2 0]
     [2 6 1]]
    [[2 3]
     [4 6]
     [1 2]]





    array([[20, 30],
           [29, 44]])




```python
# Generating identity matrix
np.identity(3)
```




    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])




```python
# Determinant of a identity matrix is 1
c = np.identity(3)
np.linalg.det(c)
```




    1.0




```python
# inverted matrix (A * A**-1 == Determinant)
c = np.random.randint(7, size=(3,3))
print(c)
np.linalg.inv(c)
```

    [[2 0 2]
     [2 3 3]
     [2 6 6]]





    array([[ 0.        ,  1.        , -0.5       ],
           [-0.5       ,  0.66666667, -0.16666667],
           [ 0.5       , -1.        ,  0.5       ]])



### File IO - Reading and Writing


```python
# Reading data from file
a = np.genfromtxt("data.txt", delimiter=",", dtype="int")
a
```




    array([[ 1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10]])




```python
a[: , -1] = (99, 99)
a
```




    array([[ 1,  2,  3,  4, 99],
           [ 6,  7,  8,  9, 99]])




```python
# Writing array to file
np.savetxt("data_002.txt", a, fmt="%d", delimiter=",")
```


```python
b = np.genfromtxt("data_002.txt", delimiter=",", dtype="int")
b
```




    array([[ 1,  2,  3,  4, 99],
           [ 6,  7,  8,  9, 99]])



### Advanced Indexing


```python
# Where in b is greater than 50
b > 50
```




    array([[False, False, False, False,  True],
           [False, False, False, False,  True]])




```python
# filter only values greater than 50
b[b > 50]
```




    array([99, 99])




```python
# multiple indexes at once
a = np.array([1,2,3,4,5,6,7,8,9])
a[[0,4,8]]
```




    array([1, 5, 9])




```python
# columns containing at least one value greater than 8
np.any(b > 8, axis=0)
```




    array([False, False, False,  True,  True])




```python
# columns containing all elements greater than 8
np.all(b > 8, axis=0)
```




    array([False, False, False, False,  True])




```python
# bolean logic and multiple conditions
((b > 3) & (b <8))
```




    array([[False, False, False,  True, False],
           [ True,  True, False, False, False]])




```python
example = np.arange(1,31).reshape(6,5)
example
```




    array([[ 1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10],
           [11, 12, 13, 14, 15],
           [16, 17, 18, 19, 20],
           [21, 22, 23, 24, 25],
           [26, 27, 28, 29, 30]])




```python
example[2:4, 0:2]
```




    array([[11, 12],
           [16, 17]])




```python
a = example[0:-2, 1:]
a[np.identity(4) == 1]
```




    array([ 2,  8, 14, 20])




```python
example[[0,1,2,3],[1,2,3,4]] 
```




    array([ 2,  8, 14, 20])




```python
example[[0, 4, 5],3:]
```




    array([[ 4,  5],
           [24, 25],
           [29, 30]])


