class CoreModel(object):
    """
    This is an abstract base class for all implemented neural networks (nn).
    It should not be used directly, but instead, one of the concrete neural networks should be used.

    To implement your own neural network, you need to implement the following methods:
    - `load`
    - `save`
    - `initialize`
    - `predict`
    - `train`
    - `evaluate`
    """

    def __init__(self):
        """
        Constructor for the CoreModel class.
        """
        super(CoreModel, self).__init__()

    def load(self):
        """
        Abstract method to load a neural network model.
        This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def save(self):
        """
        Abstract method to save a neural network model.
        This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def initialize(self):
        """
        Abstract method to initialize a neural network model.
        This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def predict(self):
        """
        Abstract method to get the action for a given state.

        This method accepts a state and returns an abstract action.
        It should be overridden in a subclass.

        # Arguments
            state (abstract): Current state of the game.

        # Returns
            action (abstract): Network's predicted action for given state.
        """
        raise NotImplementedError

    def train(self):
        """
        Abstract method to train the neural network with a given batch.

        This method should be overridden in a subclass.

        # Arguments
            batch (abstract): Mini Batch from Experience Replay Memory.
        """
        raise NotImplementedError

    def evaluate(self):
        """
        Abstract method to evaluate the neural network model.
        This method should be overridden in a subclass.
        """
        raise NotImplementedError
