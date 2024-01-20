#


### _calculate_fan_in_and_fan_out
```python
._calculate_fan_in_and_fan_out(
   param
)
```

---
Calculates the fan-in and fan-out of a tensor.

Fan-in and fan-out can be thought of as the number of input and output units, respectively, in a weight tensor.


**Args**

* **param** (Tensor) : Weight tensor


**Returns**

* **tuple**  : A tuple containing the fan-in and fan-out of the weight tensor.


----


### _calculate_correct_fan
```python
._calculate_correct_fan(
   param, mode
)
```

---
Calculates the correct fan value based on the mode.


**Args**

* **param** (Tensor) : Weight tensor
* **mode** (str) : Either 'fan_in' or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.


**Returns**

* **int**  : The correct fan value.


----


### calculate_gain
```python
.calculate_gain(
   nonlinearity, param = None
)
```

---
Returns the recommended gain value for the given nonlinearity function.

The values are as follows:
============ ==========================================
nonlinearity gain
============ ==========================================
linear       :math:`1`
conv{1,2,3}d :math:`1`
sigmoid      :math:`1`
tanh         :math:`5 / 3`
relu         :math:`\sqrt{2}`
leaky_relu   :math:`\sqrt{2 / (1 + negative\_slope^2)}`
selu         :math:`3 / 4`
============ ==========================================


**Args**

* **nonlinearity** (str) : The non-linear function (`nn.functional` name).
* **param** (float, optional) : Optional parameter for the non-linear function.


**Returns**

* **float**  : The recommended gain value.


----


### normal
```python
.normal(
   param, loc = 0.0, scale = 1.0
)
```

---
Fills the input Tensor with values drawn from a normal distribution.


**Args**

* **param** (Tensor) : an n-dimensional Tensor
* **loc** (float, optional) : the mean of the normal distribution
* **scale** (float, optional) : the standard deviation of the normal distribution


----


### uniform
```python
.uniform(
   param, low = -1.0, high = 1.0
)
```

---
Fills the input Tensor with values drawn from a uniform distribution.


**Args**

* **param** (Tensor) : an n-dimensional Tensor
* **low** (float, optional) : the lower bound of the uniform distribution
* **high** (float, optional) : the upper bound of the uniform distribution


----


### zeros
```python
.zeros(
   param
)
```

---
Fills the input Tensor with zeros.


**Args**

* **param** (Tensor) : an n-dimensional Tensor


----


### ones
```python
.ones(
   param
)
```

---
Fills the input Tensor with ones.


**Args**

* **param** (Tensor) : an n-dimensional Tensor


----


### xavier_uniform
```python
.xavier_uniform(
   param, gain = 1.0
)
```

---
Fills the input Tensor with values according to the method described in "Understanding the difficulty of training deep feedforward neural networks" - Glorot, X. & Bengio, Y. (2010), using a uniform distribution.


**Args**

* **param** (Tensor) : an n-dimensional Tensor
* **gain** (float, optional) : an optional scaling factor


----


### xavier_normal
```python
.xavier_normal(
   param, gain = 1.0
)
```

---
Fills the input Tensor with values according to the method described in "Understanding the difficulty of training deep feedforward neural networks" - Glorot, X. & Bengio, Y. (2010), using a normal distribution.


**Args**

* **param** (Tensor) : an n-dimensional Tensor
* **gain** (float, optional) : an optional scaling factor


----


### kaiming_uniform
```python
.kaiming_uniform(
   param, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu'
)
```

---
Fills the input Tensor with values according to the method described in "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification" - He, K. et al. (2015), using a uniform distribution.


**Args**

* **param** (Tensor) : an n-dimensional Tensor
* **a** (float, optional) : the negative slope of the rectifier used after this layer
* **mode** (str, optional) : either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.
* **nonlinearity** (str, optional) : the non-linear function (`nn.functional` name)


----


### kaiming_normal
```python
.kaiming_normal(
   param, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu'
)
```

---
Fills the input Tensor with values according to the method described in "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification" - He, K. et al. (2015), using a normal distribution.


**Args**

* **param** (Tensor) : an n-dimensional Tensor
* **a** (float, optional) : the negative slope of the rectifier used after this layer
* **mode** (str, optional) : either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.
* **nonlinearity** (str, optional) : the non-linear function (`nn.functional` name)

