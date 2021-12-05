class Reusable:
    ...


class ReusablePool:
    def __init__(self, size):
        self.size = size

    def acquire(self) -> Reusable:
        ...

    def release(self, reusable: Reusable):
        ...


class PoolManager:
    def __init__(self, pool: ReusablePool):
        self.pool = pool

    def __enter__(self):
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...
