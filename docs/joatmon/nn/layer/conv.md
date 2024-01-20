#


## Conv
```python 
Conv(
   in_features, out_features, kernel_size, stride = 1, padding = 0
)
```


---
Applies a 2D convolution over an input signal composed of several input planes.

# Arguments
in_features (int): Number of channels in the input image.
out_features (int): Number of channels produced by the convolution.
kernel_size (int or tuple): Size of the convolving kernel.
stride (int or tuple, optional): Stride of the convolution. Default: 1
padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0

---
# Attributes
    bias (Tensor): The learnable bias of the module of shape (out_features).


**Methods:**


### .forward
```python
.forward(
   inp
)
```

---
Defines the computation performed at every call.

# Arguments
inp (Tensor): The input tensor.

---
# Returns
    Tensor: The output tensor after applying 2D convolution.

----


## ConvTranspose
```python 
ConvTranspose(
   in_features, out_features, kernel_size, stride = 1, padding = 0
)
```


---
Applies a 2D transposed convolution operator over an input image composed of several input planes.

# Arguments
in_features (int): Number of channels in the input image.
out_features (int): Number of channels produced by the convolution.
kernel_size (int or tuple): Size of the convolving kernel.
stride (int or tuple, optional): Stride of the convolution. Default: 1
padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0

---
# Attributes
    bias (Tensor): The learnable bias of the module of shape (out_features).


**Methods:**


### .forward
```python
.forward(
   inp
)
```

---
Defines the computation performed at every call.

# Arguments
inp (Tensor): The input tensor.

---
# Returns
    Tensor: The output tensor after applying 2D transposed convolution.
