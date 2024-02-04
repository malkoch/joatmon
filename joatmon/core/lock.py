from contextlib import contextmanager
from threading import Lock

__all__ = ['RWLock']


class RWLock(object):
    """
    A class used to represent a Read-Write Lock.

    Attributes
    ----------
    w_lock : Lock
        The lock for write operations.
    num_r_lock : Lock
        The lock for read operations.
    num_r : int
        The number of read operations.

    Methods
    -------
    __init__(self)
        Initializes a new instance of the RWLock class.
    r_acquire(self)
        Acquires the read lock.
    r_release(self)
        Releases the read lock.
    r_locked(self)
        Context manager for read lock.
    w_acquire(self)
        Acquires the write lock.
    w_release(self)
        Releases the write lock.
    w_locked(self)
        Context manager for write lock.
    """

    def __init__(self, modes=None, max_read=0):
        """
        Initializes a new instance of the RWLock class.
        """
        self.w_lock = Lock()
        self.num_r_lock = Lock()
        self.num_r = 0

    def __enter__(self):
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...

    def __aenter__(self):
        ...

    def __aexit__(self, exc_type, exc_val, exc_tb):
        ...

    async def read(self):
        ...

    async def write(self):
        ...

    def r_acquire(self):
        """
        Acquires the read lock.

        This method increases the number of read operations and acquires the write lock if it's the first read operation.
        """
        # print('attempting to acquire read lock')
        self.num_r_lock.acquire()
        self.num_r += 1
        if self.num_r == 1:
            self.w_lock.acquire()
        self.num_r_lock.release()

    def r_release(self):
        """
        Releases the read lock.

        This method decreases the number of read operations and releases the write lock if there are no more read operations.
        """
        # print('attempting to release read lock')
        assert self.num_r > 0
        self.num_r_lock.acquire()
        self.num_r -= 1
        if self.num_r == 0:
            self.w_lock.release()
        self.num_r_lock.release()

    @contextmanager
    def r_locked(self):
        """
        Context manager for read lock.

        This method acquires the read lock before entering the context and releases it after exiting the context.
        """
        try:
            self.r_acquire()
            yield
        finally:
            self.r_release()

    def w_acquire(self):
        """
        Acquires the write lock.
        """
        # print('attempting to acquire write lock')
        self.w_lock.acquire()

    def w_release(self):
        """
        Releases the write lock.
        """
        # print('attempting to release write lock')
        self.w_lock.release()

    @contextmanager
    def w_locked(self):
        """
        Context manager for write lock.

        This method acquires the write lock before entering the context and releases it after exiting the context.
        """
        try:
            self.w_acquire()
            yield
        finally:
            self.w_release()
