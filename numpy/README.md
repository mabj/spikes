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
a.ndim  # number of dimentions
```




    2




```python
a.shape # shape of the array - this is a tuple with dimention's size
```




    (2, 2)




```python
a.dtype # basic data type
```




    dtype('int8')




```python
a.size # number of elements in the array
```




    4




```python
a.nbytes # number of bytes used in the array
```




    8




```python
a.itemsize # number of bytes of data type
```




    2



### Accessing and updating elements in array


```python
a[1,1] # accessing one specific element
```




    4




```python
a[0, :] # accessing first row
```




    array([1, 2], dtype=int16)




```python
a[: , 0] # Fetching first column
```




    array([1, 3], dtype=int16)




```python
a[: , 1] = (99, 99) # Updating a specific column
print(a)
```

    [[ 1 99]
     [ 3 99]]


### Generating specific matrixes


```python
np.ones((3, 3), dtype="int8") # creates arrays of 1s
```




    array([[1, 1, 1],
           [1, 1, 1],
           [1, 1, 1]], dtype=int8)




```python
np.zeros((3,3), dtype="int8") # creates arrays of 0s
```




    array([[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]], dtype=int8)




```python
np.full((2,2), 99) # creates arrays fileld with specific value
```




    array([[99, 99],
           [99, 99]])




```python
np.random.rand(2,2) # Generate random float values arrays
```




    array([[0.04652306, 0.46851432],
           [0.081302  , 0.39715904]])




```python
np.random.randint(7, size=(2,2)) # Generate random int values arrays 
```




    array([[1, 6],
           [6, 2]])




```python
np.identity(5) # Generate identity matrixes
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
a.reshape((4)) # reshaping an array
```




    array([1, 2, 3, 4], dtype=int16)




```python
b = np.array([[1,2,3]])
np.repeat(b, 3, axis=0) # repeat the content of an specific array
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

```
