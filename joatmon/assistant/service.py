import dataclasses
import datetime
import enum

from joatmon.assistant.runnable import Runnable


@dataclasses.dataclass
class Service:
    id: str
    name: str
    description: str
    priority: int
    status: str
    mode: str
    script: str
    arguments: dict
    created_at: datetime.datetime
    updated_at: datetime.datetime


class ServiceState(enum.Enum):
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
    stopped = enum.auto()


# create from json and to json methods
@dataclasses.dataclass
class ServiceInfo:
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

    service: Service
    state: ServiceState
    runnable: Runnable


class BaseService(Runnable):
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

    def __init__(self, service: Service, api, **kwargs):
        super().__init__(service, api, 'service', **kwargs)
