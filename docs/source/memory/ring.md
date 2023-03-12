### Introduction

---

<span style="float:right;">[[source]](https://github.com/malkoch/joatmon/blob/master/joatmon/ai/memory.py#L49)</span>

### RingMemory

```python
joatmon.ai.memory.RingMemory.joatmon.ai.memory.RingMemory(batch_size=32, size=960000)
```

Ring Memory

__Arguments__

- __size__ (int): .

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

---
