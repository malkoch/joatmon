from joatmon.serializable import Serializable


class Index(Serializable):
    def __init__(self, field):
        super(Index, self).__init__()

        self.field = field
