## Importing library and basic array


```python
import numpy as np

a = np.array([[1,2],[3,4]], dtype="int16")
print(a)
```

    [[1 2]
     [3 4]]


## Printing characteristcs of an Array


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




    dtype('int16')




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



## Accessing and updating elements in array


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



```python
# "Fancy indexing" == specifing a list with indexes 
l1 = np.array([1,2,3,4,5,6], dtype="int")
l1[[0,3,4]]
```




    array([1, 4, 5])



## Generating specific matrixes


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




    array([[0.28195721, 0.78903445],
           [0.51407183, 0.09611987]])




```python
# Generate random int values arrays 
np.random.randint(7, size=(2,2))
```




    array([[0, 4],
           [3, 5]])




```python
# Generate identity matrixes
np.identity(5)
```




    array([[1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 1.]])




```python
# Pics up elements from a list to generate array
np.random.choice([1,2,11], size=(3,3))
```




    array([[ 2,  1, 11],
           [ 1, 11, 11],
           [ 1,  2,  1]])



## Transformations and Mathematical operations


```python
a * 2
```




    array([[  2, 198],
           [  6, 198]], dtype=int16)




```python
a / 2
```




    array([[ 0.5, 49.5],
           [ 1.5, 49.5]])




```python
a ** 2
```




    array([[   1, 9801],
           [   9, 9801]], dtype=int16)




```python
a + 5
```




    array([[  6, 104],
           [  8, 104]], dtype=int16)




```python
# DOT multiplication
b1 = np.array([1,2,3], dtype="int")
b2 = np.array([1,2,3], dtype="int")
np.dot(b1,b2) # multiplies correspondent elements and do a summation
b1 @ b2 # another syntax to do DOT multiplications
```




    14




```python
# reshaping an array
a.reshape((4))
```




    array([ 1, 99,  3, 99], dtype=int16)




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




```python
# Array Flattening
l1 = np.array([[1,2], [3,4]], dtype="int")
l1.flatten()
```




    array([1, 2, 3, 4])




```python
# Array flattening but by reference == ravel (efficient!)
print(l1)
l2 = l1.ravel()
print(l2)
l2[3] = 99
print(l1)
```

    [[1 2]
     [3 4]]
    [1 2 3 4]
    [[ 1  2]
     [ 3 99]]



```python
# Adding new axis == adding new dimention
l1 = np.arange(1,10)
l2 = l1[np.newaxis, :]
print(l2)
l2 = l1[: , np.newaxis]
print(l2)
```

    [[1 2 3 4 5 6 7 8 9]]
    [[1]
     [2]
     [3]
     [4]
     [5]
     [6]
     [7]
     [8]
     [9]]



```python
# Concatenating arrays
l1 = np.array([[1,2], [3,4]], dtype="int")
l2 = np.array([[5,6]], dtype="int")
np.concatenate((l1,l2))
```




    array([[1, 2],
           [3, 4],
           [5, 6]])




```python
# Broadcasting
```

Broadcasting is operations using arrays of different dimensionalities. All those basic arithmetic operations at the beginning of this section are broadcats `a1 * 2` or `a1 + 2`. Where all elements in a1 are impacted by the operation with a single element.


```python
l1 = np.array([[1,2,3], [4,5,6], [7,8,9]], dtype="int")
l2 = np.array([[1,2,1]], dtype="int")
l1 * l2
```




    array([[ 1,  4,  3],
           [ 4, 10,  6],
           [ 7, 16,  9]])




```python
# Comparing two arrays
l1 = np.array([1,2,3,4], dtype="int")
l2 = np.array([1,2,3,4], dtype="int")
np.allclose(l1,l2)
```




    True



## Statistics


```python
# or a.max() - returns max value inside Array
np.max(a)
```




    99




```python
# or a.min() - returns min value inside Array
np.min(a)
```




    1




```python
# returns an array with max value of each internal Array
np.max(a, axis=1)
```




    array([99, 99], dtype=int16)




```python
# returns a summation of all elements in an array
np.sum(a)
```




    202




```python
# returns summation of rows
np.sum(a, axis=1)
```




    array([100, 102])




```python
# returns average
np.average(a)
np.mean(a) # another way to do the same
```




    50.5



## Linear Algebra


```python
# Matrix multiplication
a = np.random.randint(7, size=(2,3))
b = np.random.randint(7, size=(3,2))
print(a)
print(b)
np.matmul(a,b)
```

    [[4 1 3]
     [0 6 0]]
    [[4 0]
     [5 2]
     [4 4]]





    array([[33, 14],
           [30, 12]])




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

    [[1 4 6]
     [6 3 4]
     [2 2 5]]





    array([[-0.15555556,  0.17777778,  0.04444444],
           [ 0.48888889,  0.15555556, -0.71111111],
           [-0.13333333, -0.13333333,  0.46666667]])




```python
# Transposing a Matrix
b = np.array([[1,2], [3,4]], dtype="int")
np.transpose(b)
b.T # another syntax convention to transpose an Array
```




    array([[1, 3],
           [2, 4]])




```python
# Determinant of a Matrix
np.linalg.det(b)
```




    -2.0000000000000004




```python
# main diagonal of a Matrix
np.diag(b)
```




    array([1, 4])



## Solving Linear Equations

```
The admission fee at a small fair is $1.50 for children and $4.00 for adults. On a certain day, 2200 people enter the fair and $5050 is collected. How many children and how many adults attended?
```


```python
A = np.array([[1.0,1.0], [1.5, 4.0]])
b = np.array([2200, 5050])
x = np.linalg.inv(A).dot(b)
print(x)
x = np.linalg.solve(A,b)
print(x)
```

    [1500.  700.]
    [1500.  700.]


## File IO - Reading and Writing


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



## Advanced Indexing


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



## References

 * https://docs.scipy.org/doc/scipy/reference/linalg.html
 * https://numpy.org/doc/stable/index.html
 * https://numpy.org/doc/stable/reference/arrays.dtypes.html
 * https://www.youtube.com/watch?v=9JUAPgtkKpI **
 * https://www.youtube.com/watch?v=QUT1VHiLmmI *****
 * https://www.youtube.com/watch?v=ZB7BZMhfPgk *****
