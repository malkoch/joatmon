#


## CELoss
```python 

```


---
Implements the Cross-Entropy (CE) loss function.

CE is a loss function that is used in binary classification tasks.
It is a measure of the dissimilarity between the predicted probability and the true label.

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
Computes the CE loss between the prediction and target.

# Arguments
prediction (np.array): The predicted probability.
target (np.array): The true label.

---
# Returns
    np.array: The computed CE loss.
