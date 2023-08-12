import numpy as np
from PIL import Image


class RLProcessor:
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

    def __init__(self):
        super(RLProcessor, self).__init__()

    @staticmethod
    def process_batch(batch):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        states = []
        actions = []
        rewards = []
        next_states = []
        terminals = []
        for state, action, reward, next_state, terminal in batch:
            states.extend(state)
            actions.append(action)
            rewards.append(reward)
            next_states.extend(next_state)
            terminals.append(terminal)
        states = np.asarray(states).astype('float32')
        actions = np.asarray(actions).astype('float32')
        rewards = np.asarray(rewards).astype('float32')
        next_states = np.asarray(next_states).astype('float32')
        terminals = np.asarray(terminals).astype('float32')
        return states, actions, rewards, next_states, terminals

    @staticmethod
    def process_state(state):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if state is None:
            return

        state = Image.fromarray(state, 'RGB')
        state = state.resize((84, 84))
        state = np.array(state)
        state = np.expand_dims(state, 0)
        state = np.transpose(state, (0, 3, 1, 2))
        state = state.astype('uint8')
        return state
