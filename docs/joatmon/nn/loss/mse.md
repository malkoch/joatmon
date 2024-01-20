#


## MSELoss
```python 

```


---
Implements the Mean Squared Error (MSE) loss function.

MSE is a loss function used for regression models. It is the sum of the squared differences between the true and predicted values.

# Attributes
_loss (np.array): The computed loss value.


**Methods:**


### .forward
```python
.forward(
   prediction, target
)
```

---
Computes the MSE loss between the prediction and target.

# Arguments
prediction (np.array): The predicted values.
target (np.array): The true values.

---
# Returns
    np.array: The computed MSE loss.
