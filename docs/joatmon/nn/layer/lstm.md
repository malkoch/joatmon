#


## LSTMCell
```python 
LSTMCell(
   input_size: int, hidden_size: int
)
```


---
A Long Short Term Memory (LSTM) cell.

# Arguments
input_size (int): The number of expected features in the input x
hidden_size (int): The number of features in the hidden state h

---
# Attributes
    bias_hh (Tensor): The learnable hidden-hidden bias, of shape (4*hidden_size)


**Methods:**


### .reset_parameters
```python
.reset_parameters()
```

---
Resets the parameters (weight, bias) to their initial values.

### .check_input
```python
.check_input(
   inp: Tensor
)
```

---
Validates the input shape.

# Arguments
inp (Tensor): The input tensor.

### .get_expected_hidden_size
```python
.get_expected_hidden_size(
   inp: Tensor
)
```

---
Returns the expected hidden state size.

# Arguments
inp (Tensor): The input tensor.

---
# Returns
    Tuple[int, int]: The expected hidden state size.

### .extra_repr
```python
.extra_repr()
```

---
Returns a string containing a brief description of the module.

# Returns
str: A string containing a brief description of the module.

### .all_weights
```python
.all_weights()
```

---
Returns a list of all weights for the LSTM cell.

# Returns
List[List[Parameter]]: A list of all weights for the LSTM cell.

### .get_expected_cell_size
```python
.get_expected_cell_size(
   inp: Tensor
)
```

---
Returns the expected cell state size.

# Arguments
inp (Tensor): The input tensor.

---
# Returns
    Tuple[int, int]: The expected cell state size.

### .check_forward_args
```python
.check_forward_args(
   inp: Tensor
)
```

---
Validates the input shape for the forward pass.

# Arguments
inp (Tensor): The input tensor.

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
    Tensor: The output tensor after applying LSTM cell.

----


## LSTM
```python 
LSTM(
   input_size: int, hidden_size: int, num_layers: int = 1
)
```


---
A Long Short Term Memory (LSTM) module.

# Arguments
input_size (int): The number of expected features in the input x
hidden_size (int): The number of features in the hidden state h
num_layers (int, optional): Number of recurrent layers. Default: 1

---
# Attributes
    num_layers (int): Number of recurrent layers.


**Methods:**


### .reset_parameters
```python
.reset_parameters()
```

---
Resets the parameters (weight, bias) to their initial values.

### .check_input
```python
.check_input(
   inp: Tensor
)
```

---
Validates the input shape.

# Arguments
inp (Tensor): The input tensor.

### .get_expected_hidden_size
```python
.get_expected_hidden_size(
   inp: Tensor
)
```

---
Returns the expected hidden state size.

# Arguments
inp (Tensor): The input tensor.

---
# Returns
    Tuple[int, int, int]: The expected hidden state size.

### .extra_repr
```python
.extra_repr()
```

---
Returns a string containing a brief description of the module.

# Returns
str: A string containing a brief description of the module.

### .all_weights
```python
.all_weights()
```

---
Returns a list of all weights for the LSTM module.

# Returns
List[List[Parameter]]: A list of all weights for the LSTM module.

### .get_expected_cell_size
```python
.get_expected_cell_size(
   inp: Tensor
)
```

---
Returns the expected cell state size.

# Arguments
inp (Tensor): The input tensor.

---
# Returns
    Tuple[int, int, int]: The expected cell state size.

### .check_forward_args
```python
.check_forward_args(
   inp: Tensor
)
```

---
Validates the input shape for the forward pass.

# Arguments
inp (Tensor): The input tensor.

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
    Tensor: The output tensor after applying LSTM.
