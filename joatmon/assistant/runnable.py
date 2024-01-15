import asyncio
import uuid

from transitions import Machine

from joatmon.core.event import Event


class Runnable:
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

    def __init__(self, info, api, type, **kwargs):  # another parameter called cache output
        states = ['none', 'started', 'stopped', 'running', 'exception', 'starting', 'stopping']
        self.machine = Machine(model=self, states=states, initial='none')

        self.process_id = uuid.uuid4()
        self.info = info
        self.api = api
        self.type = type
        self.kwargs = kwargs
        self.event = asyncio.Event()
        self.task = asyncio.ensure_future(self.run(), loop=self.api.loop)

    @staticmethod
    def help():
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return {}

    async def run(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        raise NotImplementedError

    def running(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return not self.task.done()

    def start(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.machine.set_state('starting')
        events['begin'].fire()
        self.machine.set_state('started')

    def stop(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.machine.set_state('stopping')
        events['end'].fire()
        self.event.set()
        self.machine.set_state('stopped')
        self.task.cancel()


events = {
    'begin': Event(),
    'end': Event(),
    'error': Event(),
}
