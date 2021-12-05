## Available Policies

| Name                                                 | Implementation                           |
|------------------------------------------------------|------------------------------------------|
| [GreedyQPolicy](/policy/greedy-q-policy)             | `joatmon.ai.policy.GreedyQPolicy`       |
| [EpsilonGreedyPolicy](/policy/epsilon-greedy-policy) | `joatmon.ai.policy.EpsilonGreedyPolicy` |

---

## Common API

All policies share a common API. This allows you to easily switch between different policies.

<span style="float:right;">[[source]](https://github.com/malkoch/joatmon/blob/master/joatmon/ai/core.py#L234)</span>
### CorePolicy

```python
joatmon.ai.core.CorePolicy.joatmon.ai.core.CorePolicy()
```


Abstract base class for all implemented policy.

Do not use this abstract base class directly but
instead use one of the concrete policy implemented.

To implement your own policy, you have to implement the following methods:

- `decay`
- `use_network`

----

<span style="float:right;">[[source]](https://github.com/malkoch/joatmon/blob/master/joatmon/ai/core.py#L250)</span>

### reset


```python
CorePolicy.reset(self)
```



reset

----

<span style="float:right;">[[source]](https://github.com/malkoch/joatmon/blob/master/joatmon/ai/core.py#L256)</span>

### decay


```python
CorePolicy.decay(self)
```



Decaying the epsilon / sigma value of the policy.

----

<span style="float:right;">[[source]](https://github.com/malkoch/joatmon/blob/master/joatmon/ai/core.py#L262)</span>

### use_network


```python
CorePolicy.use_network(self)
```



Sample an experience replay batch with size.

__Returns__

- __use__ (bool): Boolean value for using the nn.

