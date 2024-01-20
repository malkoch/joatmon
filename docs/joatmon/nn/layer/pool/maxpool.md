#


## MaxPool
```python 
MaxPool(
   kernel_size, stride = None, padding = 0
)
```


---
Applies max pooling to the input.

# Arguments
kernel_size (int or tuple): Size of the window to take max over.
stride (int or tuple): Stride of the window. Default value is `kernel_size`.
padding (int or tuple): Implicit zero padding to be added on both sides.

---
# Attributes
    _padding (int or tuple): Implicit zero padding to be added on both sides.


**Methods:**


### .forward
```python
.forward(
   inp
)
```

---
Applies max pooling to the input.

# Arguments
inp (Tensor): The input tensor.

---
# Returns
    Tensor: The output tensor after applying max pooling.
