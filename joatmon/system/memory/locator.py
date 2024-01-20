import copy
import struct

from joatmon.system.memory.address import Address


class Locator(object):
    """
    A class used to represent a Locator.

    Attributes
    ----------
    worker : object
        The worker object that the locator belongs to.
    of_type : str
        The type of the locator.
    last_iteration : dict
        The last iteration of the locator.
    last_value : object
        The last value of the locator.
    start : int
        The start of the locator.
    end : int
        The end of the locator.

    Methods
    -------
    __init__(self, worker, of_type='unknown', start=None, end=None)
        Initializes a new instance of the Locator class.
    find(self, value, erase_last=True)
        Finds the given value in the locator.
    feed(self, value, erase_last=True)
        Feeds the given value into the locator.
    get_addresses(self)
        Gets the addresses of the locator.
    diff(self, erase_last=False)
        Gets the difference of the locator.
    get_modified_address(self, erase_last=False)
        Gets the modified address of the locator.
    """

    def __init__(self, worker, of_type='unknown', start=None, end=None):
        """
        Initializes a new instance of the Locator class.

        Args:
            worker (object): The worker object that the locator belongs to.
            of_type (str, optional): The type of the locator. Defaults to 'unknown'.
            start (int, optional): The start of the locator. Defaults to None.
            end (int, optional): The end of the locator. Defaults to None.
        """
        self.worker = worker
        self.of_type = of_type
        self.last_iteration = {}
        self.last_value = None
        self.start = start
        self.end = end

    def find(self, value, erase_last=True):
        """
        Finds the given value in the locator.

        Args:
            value (object): The value to find.
            erase_last (bool, optional): Whether to erase the last value. Defaults to True.

        Returns:
            dict: The new iteration of the locator.
        """
        return self.feed(value, erase_last)

    def feed(self, value, erase_last=True):
        """
        Feeds the given value into the locator.

        Args:
            value (object): The value to feed.
            erase_last (bool, optional): Whether to erase the last value. Defaults to True.

        Returns:
            dict: The new iteration of the locator.
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
        Gets the addresses of the locator.

        Returns:
            dict: The addresses of the locator.
        """
        return self.last_iteration

    def diff(self, erase_last=False):
        """
        Gets the difference of the locator.

        Args:
            erase_last (bool, optional): Whether to erase the last value. Defaults to False.

        Returns:
            dict: The difference of the locator.
        """
        return self.get_modified_address(erase_last)

    def get_modified_address(self, erase_last=False):
        """
        Gets the modified address of the locator.

        Args:
            erase_last (bool, optional): Whether to erase the last value. Defaults to False.

        Returns:
            dict: The modified address of the locator.
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
