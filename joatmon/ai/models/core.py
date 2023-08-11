class CoreModel(object):
    """
    Abstract base class for all implemented nn.

    Do not use this abstract base class directly
    but instead use one of the concrete nn implemented.

    To implement your own nn, you have to implement the following methods:

    - `act`
    - `replay`
    - `load`
    - `save`
    """

    def __init__(self):
        super(CoreModel, self).__init__()

    def load(self):
        """
        load
        """
        raise NotImplementedError

    def save(self):
        """
        save
        """
        raise NotImplementedError

    def initialize(self):
        """
        initialize
        """
        raise NotImplementedError

    def predict(self):
        """
        Get the action for given state.

        Accepts a state and returns an abstract action.

        # Arguments
            state (abstract): Current state of the game.

        # Returns
            action (abstract): Network's predicted action for given state.
        """
        raise NotImplementedError

    def train(self):
        """
        Train the nn with given batch.

        # Arguments
            batch (abstract): Mini Batch from Experience Replay Memory.
        """
        raise NotImplementedError

    def evaluate(self):
        """
        evaluate
        """
        raise NotImplementedError
