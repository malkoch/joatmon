#


## DQNTrainer
```python 
DQNTrainer(
   environment, memory, processor, model, callbacks, test_policy, train_policy,
   her = False, action_num = 4
)
```




**Methods:**


### .goal
```python
.goal()
```


### .get_step
```python
.get_step(
   action, mode = 'q_learning', action_number = 4
)
```


### .get_action
```python
.get_action(
   state, goal_state, policy
)
```


### .train
```python
.train(
   batch_size = 32, max_action = 200, max_episode = 12000, warmup = 120000
)
```


### .evaluate
```python
.evaluate(
   max_action = 50, max_episode = 12
)
```


----


## DDPGTrainer
```python 
DDPGTrainer(
   environment, random_process, processor, memory, model, callbacks, her = False
)
```




**Methods:**


### .goal
```python
.goal()
```


### .get_action
```python
.get_action(
   state, goal_state
)
```


### .train
```python
.train(
   batch_size = 32, max_action = 50, max_episode = 120, warmup = 0, replay_interval = 4,
   update_interval = 1, test_interval = 1000
)
```


### .evaluate
```python
.evaluate(
   max_action = 50, max_episode = 12
)
```


----


## TD3Trainer
```python 
TD3Trainer(
   environment, random_process, processor, memory, model, callbacks, her = False
)
```




**Methods:**


### .goal
```python
.goal()
```


### .get_action
```python
.get_action(
   state, goal_state
)
```


### .train
```python
.train(
   batch_size = 32, max_action = 50, max_episode = 120, warmup = 0, replay_interval = 4,
   update_interval = 1, test_interval = 1000
)
```


### .evaluate
```python
.evaluate(
   max_action = 50, max_episode = 12
)
```

