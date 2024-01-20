#


## Dropout
```python 
Dropout(
   keep_prob = 0.5
)
```


---
Applies Dropout to the input.

The Dropout layer randomly sets input units to 0 with a frequency of `keep_prob`
at each step during training time, which helps prevent overfitting.
Inputs not set to 0 are scaled up by 1/(1 - keep_prob) such that the sum over
all inputs is unchanged.

# Arguments
keep_prob (float): float between 0 and 1. Fraction of the input units to drop.

---
# Attributes
    _keep_prob (float): Fraction of the input units to drop.


**Methods:**


### .forward
```python
.forward(
   inp
)
```

---
Applies Dropout to the input.

# Arguments
inp (Tensor): The input tensor.

---
# Returns
    Tensor: The output tensor after applying dropout.
