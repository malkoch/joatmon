<span style="float:right;">[[source]](https://github.com/malkoch/joatmon/blob/master/joatmon/ai/core.py#L8)</span>
### CoreBuffer

```python
joatmon.ai.core.CoreBuffer.joatmon.ai.core.CoreBuffer(values, batch_size)
```

----

<span style="float:right;">[[source]](https://github.com/malkoch/joatmon/blob/master/joatmon/ai/core.py#L92)</span>
### CoreCallback

```python
joatmon.ai.core.CoreCallback.joatmon.ai.core.CoreCallback()
```


Abstract base class for all implemented callback.

Do not use this abstract base class directly but instead use one of the concrete callback implemented.

To implement your own callback, you have to implement the following methods:

- `on_action_begin`
- `on_action_end`
- `on_replay_begin`
- `on_replay_end`
- `on_episode_begin`
- `on_episode_end`
- `on_agent_begin`
- `on_agent_end`

----

<span style="float:right;">[[source]](https://github.com/malkoch/joatmon/blob/master/joatmon/ai/core.py#L162)</span>
### CoreEnv

```python
joatmon.ai.core.CoreEnv.joatmon.ai.core.CoreEnv()
```


The abstract environment class that is used by all agents. This class has the exact same API that OpenAI Gym uses so that integrating
with it is trivial. In contrast to the OpenAI Gym implementation, this class only defines the abstract methods without any actual implementation.

To implement your own environment, you need to define the following methods:

- `seed`
- `reset`
- `step`
- `render`
- `close`

Refer to the [Gym documentation](https://gym.openai.com/docs/#environment).

----

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

<span style="float:right;">[[source]](https://github.com/malkoch/joatmon/blob/master/joatmon/ai/core.py#L312)</span>
### CoreModel

```python
joatmon.ai.core.CoreModel.joatmon.ai.core.CoreModel()
```


Abstract base class for all implemented nn.

Do not use this abstract base class directly
but instead use one of the concrete nn implemented.

To implement your own nn, you have to implement the following methods:

- `act`
- `replay`
- `load`
- `save`

----

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

<span style="float:right;">[[source]](https://github.com/malkoch/joatmon/blob/master/joatmon/ai/core.py#L272)</span>
### CoreRandom

```python
joatmon.ai.core.CoreRandom.joatmon.ai.core.CoreRandom()
```


Abstract base class for all implemented random processes.

Do not use this abstract base class directly but instead
use one of the concrete random processes implemented.

To implement your own random processes,
you have to implement the following methods:

- `decay`
- `sample`
- `reset`

