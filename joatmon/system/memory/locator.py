import copy
import struct

from joatmon.system.memory.address import Address


class Locator(object):
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

    def __init__(self, worker, of_type='unknown', start=None, end=None):
        self.worker = worker
        self.of_type = of_type
        self.last_iteration = {}
        self.last_value = None
        self.start = start
        self.end = end

    def find(self, value, erase_last=True):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return self.feed(value, erase_last)

    def feed(self, value, erase_last=True):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.last_value = value
        new_iter = copy.copy(self.last_iteration)
        if self.of_type == 'unknown':
            all_types = ['uint', 'int', 'long', 'ulong', 'float', 'double', 'short', 'ushort']
        else:
            all_types = [self.of_type]

        for of_type in all_types:
            if of_type not in new_iter:
                try:
                    new_iter[of_type] = [
                        Address(address, self.worker.process, of_type)
                        for address in self.worker.memory_search(
                            value, of_type, start_offset=self.start, end_offset=self.end
                        )
                    ]
                except struct.error:
                    new_iter[of_type] = []
            else:
                addresses = []
                for address in new_iter[of_type]:
                    try:
                        found = self.worker.process.read(address, of_type)
                        if int(found) == int(value):
                            addresses.append(Address(address, self.worker.process, of_type))
                    except Exception as ex:
                        print(str(ex))

                new_iter[of_type] = addresses

        if erase_last:
            del self.last_iteration
            self.last_iteration = new_iter
        return new_iter

    def get_addresses(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return self.last_iteration

    def diff(self, erase_last=False):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return self.get_modified_address(erase_last)

    def get_modified_address(self, erase_last=False):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        last = self.last_iteration
        new = self.feed(self.last_value, erase_last=erase_last)
        ret = {}
        for of_type, addresses in last.items():
            typeset = set(new[of_type])
            for _address in addresses:
                if _address not in typeset:
                    if of_type not in ret:
                        ret[of_type] = []
                    ret[of_type].append(_address)

        return ret
