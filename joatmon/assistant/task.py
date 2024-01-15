import dataclasses
import datetime
import enum

from joatmon.assistant.runnable import Runnable


@dataclasses.dataclass
class Task:
    id: str
    name: str
    description: str
    priority: int
    status: bool
    on: str
    script: str
    arguments: dict
    created_at: datetime.datetime
    updated_at: datetime.datetime


class TaskState(enum.Enum):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    running = enum.auto()
    finished = enum.auto()


# create from json and to json methods
@dataclasses.dataclass
class TaskInfo:
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    task: Task
    state: TaskState
    runnable: Runnable


class BaseTask(Runnable):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, task: Task, api, **kwargs):  # another parameter called cache output
        super().__init__(task, api, 'task', **kwargs)
