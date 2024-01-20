#


## CCELoss
```python 

```


---
Implements the Categorical Cross-Entropy (CCE) loss function.

CCE is a loss function that is used in multi-class classification tasks.
It is a measure of the dissimilarity between the predicted probability distribution and the true distribution.

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
Computes the CCE loss between the prediction and target.

# Arguments
prediction (np.array): The predicted probability distribution.
target (np.array): The true distribution.

---
# Returns
    np.array: The computed CCE loss.
