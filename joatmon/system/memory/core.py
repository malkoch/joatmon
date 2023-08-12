import ctypes
import struct
import sys
from ctypes import wintypes

if sys.platform != 'win32':
    raise Exception('The ram module should only be used on a Windows system.')


def type_unpack(of_type):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    of_type = of_type.lower()
    if of_type == 'short':
        structure_type = 'h'
        structure_length = 2
    elif of_type == 'ushort':
        structure_type = 'H'
        structure_length = 2
    elif of_type == 'int':
        structure_type = 'i'
        structure_length = 4
    elif of_type == 'uint':
        structure_type = 'I'
        structure_length = 4
    elif of_type == 'long':
        structure_type = 'l'
        structure_length = 4
    elif of_type == 'ulong':
        structure_type = 'L'
        structure_length = 4
    elif of_type == 'float':
        structure_type = 'f'
        structure_length = 4
    elif of_type == 'double':
        structure_type = 'd'
        structure_length = 8
    else:
        raise TypeError(f'Unknown type {of_type}')
    return '<' + structure_type, structure_length


def hex_dump(data, address=0, prefix='', of_type='bytes'):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    dump = prefix
    piece = ''
    if of_type != 'bytes':
        structure_type, structure_len = type_unpack(of_type)
        for i in range(0, len(data), structure_len):
            if address % 16 == 0:
                dump += ' '
                for char in piece:
                    if 32 <= ord(char) <= 126:
                        dump += char
                    else:
                        dump += '.'

                dump += '\n%s%08X: ' % (prefix, address)
                piece = ''
            temp_val = 'NaN'
            try:
                packed_val = data[i : i + structure_len]
                temp_val = struct.unpack(structure_type, packed_val)[0]
            except Exception as ex_hex_dump:
                print(str(ex_hex_dump))

            if temp_val == 'NaN':
                dump += '{:<15} '.format(temp_val)
            elif of_type == 'float':
                dump += '{:<15.4f} '.format(temp_val)
            else:
                dump += '{:<15} '.format(temp_val)
            address += structure_len
    else:
        for byte in data:
            if address % 16 == 0:
                dump += ' '
                for char in piece:
                    if 32 <= ord(char) <= 126:
                        dump += char
                    else:
                        dump += '.'

                dump += '\n%s%08X: ' % (prefix, address)
                piece = ''

            dump += '%02X ' % byte
            piece += chr(byte)
            address += 1

    remainder = address % 16
    if remainder != 0:
        dump += '   ' * (16 - remainder) + ' '
    for char in piece:
        if 32 <= ord(char) <= 126:
            dump += char
        else:
            dump += '.'

    return dump + '\n'


MEM_COMMIT = 4096
MEM_FREE = 65536
MEM_RESERVE = 8192

PAGE_EXECUTE_READWRITE = 64
PAGE_EXECUTE_READ = 32
PAGE_READONLY = 2
PAGE_READWRITE = 4
PAGE_NOCACHE = 512
PAGE_WRITECOMBINE = 1024
PAGE_GUARD = 256

UNPROTECTED_DACL_SECURITY_INFORMATION = 536870912
DACL_SECURITY_INFORMATION = 4

if ctypes.sizeof(ctypes.c_void_p) == 8:
    ULONG_PTR = ctypes.c_ulonglong
else:
    ULONG_PTR = ctypes.c_ulong


class ModuleEntry32(ctypes.Structure):
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

    _fields_ = [
        ('dwSize', ctypes.c_ulong),
        ('th32ModuleID', ctypes.c_ulong),
        ('th32ProcessID', ctypes.c_ulong),
        ('GlblcntUsage', ctypes.c_ulong),
        ('ProccntUsage', ctypes.c_ulong),
        ('modBaseAddr', ctypes.POINTER(ctypes.c_byte)),
        ('modBaseSize', ctypes.c_ulong),
        ('hModule', ctypes.c_void_p),
        ('szModule', ctypes.c_char * 256),
        ('szExePath', ctypes.c_char * 260),
    ]


class MemoryBasicInformation(ctypes.Structure):
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

    _fields_ = [
        ('BaseAddress', ctypes.c_void_p),
        ('AllocationBase', ctypes.c_void_p),
        ('AllocationProtect', wintypes.DWORD),
        ('RegionSize', ctypes.c_size_t),
        ('State', wintypes.DWORD),
        ('Protect', wintypes.DWORD),
        ('Type', wintypes.DWORD),
    ]


class SystemInfo(ctypes.Structure):
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

    _fields_ = [
        ('wProcessorArchitecture', wintypes.WORD),
        ('wReserved', wintypes.WORD),
        ('dwPageSize', wintypes.DWORD),
        ('lpMinimumApplicationAddress', wintypes.LPVOID),
        ('lpMaximumApplicationAddress', wintypes.LPVOID),
        ('dwActiveProcessorMask', ULONG_PTR),
        ('dwNumberOfProcessors', wintypes.DWORD),
        ('dwProcessorType', wintypes.DWORD),
        ('dwAllocationGranularity', wintypes.DWORD),
        ('wProcessorLevel', wintypes.WORD),
        ('wProcessorRevision', wintypes.WORD),
    ]


class SecurityDescriptor(ctypes.Structure):
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

    _fields_ = [
        ('SID', wintypes.DWORD),
        ('group', wintypes.DWORD),
        ('dacl', wintypes.DWORD),
        ('sacl', wintypes.DWORD),
        ('test', wintypes.DWORD),
    ]


class Th32csClass(object):
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

    INHERIT = 2147483648
    SNAPHEAPLIST = 1
    SNAPMODULE = 8
    SNAPMODULE32 = 16
    SNAPPROCESS = 2
    SNAPTHREAD = 4
    ALL = 2032639


PSecurityDescriptor = ctypes.POINTER(SecurityDescriptor)

CreateToolhelp32Snapshot = ctypes.windll.kernel32.CreateToolhelp32Snapshot
CreateToolhelp32Snapshot.argtypes = [ctypes.c_int, ctypes.c_int]
CreateToolhelp32Snapshot.reltype = ctypes.c_long

Module32First = ctypes.windll.kernel32.Module32First
Module32First.argtypes = [ctypes.c_void_p, ctypes.POINTER(ModuleEntry32)]
Module32First.rettype = ctypes.c_int

Module32Next = ctypes.windll.kernel32.Module32Next
Module32Next.argtypes = [ctypes.c_void_p, ctypes.POINTER(ModuleEntry32)]
Module32Next.rettype = ctypes.c_int

ReadProcessMemory = ctypes.windll.kernel32.ReadProcessMemory
ReadProcessMemory.argtypes = [
    wintypes.HANDLE,
    wintypes.LPCVOID,
    wintypes.LPVOID,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]

VirtualQueryEx = ctypes.windll.kernel32.VirtualQueryEx
VirtualQueryEx.argtypes = [wintypes.HANDLE, wintypes.LPCVOID, ctypes.POINTER(MemoryBasicInformation), ctypes.c_size_t]
VirtualQueryEx.restype = ctypes.c_size_t

IsWow64Process = None
if hasattr(ctypes.windll.kernel32, 'IsWow64Process'):
    IsWow64Process = ctypes.windll.kernel32.IsWow64Process
    IsWow64Process.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_bool)]
    IsWow64Process.restype = ctypes.c_bool

if ctypes.sizeof(ctypes.c_void_p) == 8:
    NtWow64ReadVirtualMemory64 = None
else:
    try:
        NtWow64ReadVirtualMemory64 = ctypes.windll.ntdll.NtWow64ReadVirtualMemory64
        NtWow64ReadVirtualMemory64.argtypes = [
            wintypes.HANDLE,
            ctypes.c_longlong,
            wintypes.LPVOID,
            ctypes.c_ulonglong,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        NtWow64ReadVirtualMemory64.restype = wintypes.BOOL
    except Exception as ex:
        print(str(ex))
        NtWow64ReadVirtualMemory64 = None
