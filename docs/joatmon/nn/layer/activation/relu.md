#


## ReLU
```python 
ReLU(
   alpha = 0
)
```


---
Applies the ReLU (Rectified Linear Unit) activation function to the input.

# Arguments
alpha (float): Controls the slope for values less than zero. Default is 0.

---
# Attributes
    alpha (float): Controls the slope for values less than zero.


**Methods:**


### .forward
```python
.forward(
   inp
)
```

---
Applies the ReLU activation function to the input.

# Arguments
inp (Tensor): The input tensor.

---
# Returns
    Tensor: The output tensor with the same shape as the input.
