import struct

import joatmon


class AddressError(Exception):
    pass


class MemoryRegionError(Exception):
    pass


class ProcessError(Exception):
    pass


class Address:
    def __init__(self, task, address):
        self.task = task
        self.address = address

    def read(self):
        ...


class Region:
    def __init__(self, task, start, length):
        ...

    def _iterate(self, dtype):
        ...


class Process:
    def __init__(self, pid, name):
        ...

    def _iterate(self):
        ...


def get_size(dtype: str) -> int:
    if dtype in ['char', 'schar', 'uchar', 'bool']:
        return 1
    elif dtype in ['short', 'ushort', 'half']:
        return 2
    elif dtype in ['int', 'uint', 'single']:
        return 4
    elif dtype in ['long', 'ulong', 'double']:
        return 8
    else:
        raise joatmon.bot.memory.core.AddressError(f"Unsupported data type: {dtype}")


def get_letter(dtype: str) -> str:
    return {
        'char': 'c',
        'schar': 'b',
        'uchar': 'B',
        'bool': '?',
        'short': 'h',
        'ushort': 'H',
        'half': 'e',
        'int': 'i',
        'uint': 'I',
        'single': 'f',
        'long': 'l',
        'ulong': 'L',
        'double': 'd'
    }.get(dtype, None)


def parse(byte_array, dtype: str):
    data_count = len(byte_array) // get_size(dtype)
    letter = get_letter(dtype)

    if not letter:
        raise ValueError(f"Unsupported data type: {dtype}")

    return struct.unpack(f'{data_count}{letter}', byte_array)
