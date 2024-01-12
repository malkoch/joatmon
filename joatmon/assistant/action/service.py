from joatmon.assistant import service
from joatmon.assistant.task import BaseTask


class Task(BaseTask):
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

    def __init__(self, name, **kwargs):
        super(Task, self).__init__(name, **kwargs)

    @staticmethod
    def help():
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return {
            'name': 'service',
            'description': 'a function for user to create service or task',
            'parameters': {
                'type': 'object',
                'properties': {'mode': {'type': 'string', 'enum': ['task', 'service']}},
                'required': ['mode'],
            },
        }

    def run(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        action = self.kwargs.get('action', '')
        mode = self.kwargs.get('mode', '')
        name = self.kwargs.get('name', '')
        priority = self.kwargs.get('priority', 0)
        script = self.kwargs.get('script', '')
        kwargs = self.kwargs.get('kwargs', {})

        if action == 'create':
            service.create(name, priority, mode, script, kwargs)

        if not self.stop_event.is_set():
            self.stop_event.set()
