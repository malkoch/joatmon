## Available Memories

| Name                                | Implementation                  |
|-------------------------------------|---------------------------------|
| [RingMemory](/memories/ring-memory) | `joatmon.ai.memory.RingMemory` |

---

## Common API

All memories share a common API. This allows you to easily switch between different memories.

<span style="float:right;">[[source]](https://github.com/malkoch/joatmon/blob/master/joatmon/ai/core.py#L35)</span>

### CoreMemory

```python
joatmon.ai.core.CoreMemory.joatmon.ai.core.CoreMemory(buffer, batch_size)
```

Abstract base class for all implemented memory.

Do not use this abstract base class directly but instead use one of the concrete memory implemented.

To implement your own memory, you have to implement the following methods:

- `remember`
- `sample`

----

<span style="float:right;">[[source]](https://github.com/malkoch/joatmon/blob/master/joatmon/ai/core.py#L70)</span>

### remember

```python
CoreMemory.remember(self, element)
```

Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

__Arguments__

- __transaction__ (abstract): state, action, reward, next_state, terminal transaction.

----

<span style="float:right;">[[source]](https://github.com/malkoch/joatmon/blob/master/joatmon/ai/core.py#L81)</span>

### sample

```python
CoreMemory.sample(self)
```

Sample an experience replay batch with size.

__Returns__

- __batch__ (abstract): Randomly selected batch
  from experience replay memory.

