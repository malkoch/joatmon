#


## MAELoss
```python 

```


---
Implements the Mean Absolute Error (MAE) loss function.

MAE is a loss function used for regression models. It is the sum of the absolute differences between the true and predicted values.

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
Computes the MAE loss between the prediction and target.

# Arguments
prediction (np.array): The predicted values.
target (np.array): The true values.

---
# Returns
    np.array: The computed MAE loss.
