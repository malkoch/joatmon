import binascii
import copy
import ctypes
import os
import platform
import re
import struct
import sys
import traceback
from ctypes import wintypes

__all__ = ['Worker', 'Locator', 'Address', 'Process', 'ProcessException']

if sys.platform != 'win32':
    raise Exception('The ram module should only be used on a Windows system.')


def type_unpack(of_type):
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
                packed_val = data[i:i + structure_len]
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


class Address(object):
    def __init__(self, value, process, default_type='uint'):
        self.value = int(value)
        self.process = process
        self.default_type = default_type
        self.symbolic_name = None

    def read(self, of_type=None, max_length=None, errors='raise'):
        if max_length is None:
            try:
                max_length = int(of_type)
                of_type = None
            except Exception as ex:
                print(str(ex))

        if not of_type:
            of_type = self.default_type

        if not max_length:
            return self.process.read(self.value, of_type=of_type, errors=errors)
        else:
            return self.process.read(self.value, of_type=of_type, max_length=max_length, errors=errors)

    def write(self, data, of_type=None):
        if not of_type:
            of_type = self.default_type
        return self.process.write(self.value, data, of_type=of_type)

    def symbol(self):
        return self.process.get_symbolic_name(self.value)

    def get_instruction(self):
        return self.process.get_instruction(self.value)

    def dump(self, of_type='bytes', size=512, before=32):
        buf = self.process.read_bytes(self.value - before, size)
        print(hex_dump(buf, self.value - before, of_type=of_type))

    def __nonzero__(self):
        return self.value is not None and self.value != 0

    def __add__(self, other):
        return Address(self.value + int(other), self.process, self.default_type)

    def __sub__(self, other):
        return Address(self.value - int(other), self.process, self.default_type)

    def __repr__(self):
        if not self.symbolic_name:
            self.symbolic_name = self.symbol()
        return str(f'<Addr: {self.symbolic_name}>')

    def __str__(self):
        if not self.symbolic_name:
            self.symbolic_name = self.symbol()
        return str(f'<Addr: {self.symbolic_name} : "{str(self.read())}" ({self.default_type})>')

    def __int__(self):
        return int(self.value)

    def __hex__(self):
        return hex(self.value)

    def __get__(self, instance, owner):
        return self.value

    def __set__(self, instance, value):
        self.value = int(value)

    def __lt__(self, other):
        return self.value < int(other)

    def __le__(self, other):
        return self.value <= int(other)

    def __eq__(self, other):
        return self.value == int(other)

    def __ne__(self, other):
        return self.value != int(other)

    def __gt__(self, other):
        return self.value > int(other)

    def __ge__(self, other):
        return self.value >= int(other)


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
        ('szExePath', ctypes.c_char * 260)
    ]


class MemoryBasicInformation(ctypes.Structure):
    _fields_ = [
        ('BaseAddress', ctypes.c_void_p),
        ('AllocationBase', ctypes.c_void_p),
        ('AllocationProtect', wintypes.DWORD),
        ('RegionSize', ctypes.c_size_t),
        ('State', wintypes.DWORD),
        ('Protect', wintypes.DWORD),
        ('Type', wintypes.DWORD)
    ]


class SystemInfo(ctypes.Structure):
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
        ('wProcessorRevision', wintypes.WORD)
    ]


class SecurityDescriptor(ctypes.Structure):
    _fields_ = [
        ('SID', wintypes.DWORD),
        ('group', wintypes.DWORD),
        ('dacl', wintypes.DWORD),
        ('sacl', wintypes.DWORD),
        ('test', wintypes.DWORD)
    ]


class Th32csClass(object):
    INHERIT = 2147483648
    SNAPHEAPLIST = 1
    SNAPMODULE = 8
    SNAPMODULE32 = 16
    SNAPPROCESS = 2
    SNAPTHREAD = 4
    ALL = 2032639


PSecurityDescriptor = ctypes.POINTER(SecurityDescriptor)

CreateToolhelp32Snapshot = ctypes.windll.kernel32.CreateToolhelp32Snapshot
CreateToolhelp32Snapshot.argtypes = [
    ctypes.c_int, ctypes.c_int
]
CreateToolhelp32Snapshot.reltype = ctypes.c_long

Module32First = ctypes.windll.kernel32.Module32First
Module32First.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ModuleEntry32)
]
Module32First.rettype = ctypes.c_int

Module32Next = ctypes.windll.kernel32.Module32Next
Module32Next.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ModuleEntry32)
]
Module32Next.rettype = ctypes.c_int

ReadProcessMemory = ctypes.windll.kernel32.ReadProcessMemory
ReadProcessMemory.argtypes = [
    wintypes.HANDLE, wintypes.LPCVOID, wintypes.LPVOID, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
]

VirtualQueryEx = ctypes.windll.kernel32.VirtualQueryEx
VirtualQueryEx.argtypes = [
    wintypes.HANDLE, wintypes.LPCVOID, ctypes.POINTER(MemoryBasicInformation), ctypes.c_size_t
]
VirtualQueryEx.restype = ctypes.c_size_t

IsWow64Process = None
if hasattr(ctypes.windll.kernel32, 'IsWow64Process'):
    IsWow64Process = ctypes.windll.kernel32.IsWow64Process
    IsWow64Process.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_bool)
    ]
    IsWow64Process.restype = ctypes.c_bool

if ctypes.sizeof(ctypes.c_void_p) == 8:
    NtWow64ReadVirtualMemory64 = None
else:
    try:
        NtWow64ReadVirtualMemory64 = ctypes.windll.ntdll.NtWow64ReadVirtualMemory64
        NtWow64ReadVirtualMemory64.argtypes = [
            wintypes.HANDLE, ctypes.c_longlong, wintypes.LPVOID, ctypes.c_ulonglong, ctypes.POINTER(ctypes.c_ulong)
        ]
        NtWow64ReadVirtualMemory64.restype = wintypes.BOOL
    except Exception as ex:
        print(str(ex))
        NtWow64ReadVirtualMemory64 = None


class ProcessException(Exception):
    pass


class BaseProcess(object):
    def __init__(self, *args, **kwargs):
        self.h_process = None
        self.pid = None
        self.is_process_open = False
        self.buffer = None
        self.buffer_len = 0

    def __del__(self):
        self.close()

    def close(self):
        pass

    def iter_region(self, *args, **kwargs):
        raise NotImplementedError

    def write_bytes(self, address, data):
        raise NotImplementedError

    def read_bytes(self, address, size=4):
        raise NotImplementedError

    @staticmethod
    def get_symbolic_name(a):
        return '0x%08X' % int(a)

    def read(self, address, of_type='uint', max_length=50, errors='raise'):
        if of_type == 's' or of_type == 'string':
            byte_array = self.read_bytes(int(address), size=max_length)
            new_string = ''
            for character in byte_array:
                if character == '\x00':
                    return new_string
                new_string += character
            if errors == 'ignore':
                return new_string
            raise ProcessException('string > maxlen')
        else:
            if of_type == 'bytes' or of_type == 'b':
                return self.read_bytes(int(address), size=max_length)
            struct_type, struct_length = type_unpack(of_type)
            return struct.unpack(struct_type, self.read_bytes(int(address), size=struct_length))[0]

    def write(self, address, data, of_type='uint'):
        if of_type != 'bytes':
            struct_type, struct_length = type_unpack(of_type)
            return self.write_bytes(int(address), struct.pack(struct_type, data))
        else:
            return self.write_bytes(int(address), data)


class Process(BaseProcess):
    def __init__(self, pid=None, name=None, debug=True):
        super(Process, self).__init__()
        if pid:
            self._open(int(pid), debug=debug)
        elif name:
            self._open_from_name(name, debug=debug)
        else:
            raise ValueError("You need to instanciate process with at least a name or a pid")

        if self.is_64():
            si = self.get_native_system_info()
            self.max_address = si.lpMaximumApplicationAddress
        else:
            si = self.get_system_info()
            self.max_address = 2147418111

        self.min_address = si.lpMinimumApplicationAddress

    def __del__(self):
        self.close()

    def is_64(self):
        if "64" not in platform.machine():
            return False
        is_wow_64 = ctypes.c_bool(False)
        if IsWow64Process is None:
            return False
        if not IsWow64Process(self.h_process, ctypes.byref(is_wow_64)):
            raise ctypes.WinError()
        return not is_wow_64.value

    @staticmethod
    def list():
        processes = []
        arr = ctypes.c_ulong * 256
        l_pid_process = arr()
        cb = ctypes.sizeof(l_pid_process)
        cb_needed = ctypes.c_ulong()
        h_module = ctypes.c_ulong()
        count = ctypes.c_ulong()
        modname = ctypes.create_string_buffer(100)
        process_query_information = 0x0400
        process_vm_read = 0x0010

        ctypes.windll.psapi.EnumProcesses(ctypes.byref(l_pid_process), cb, ctypes.byref(cb_needed))
        n_returned = cb_needed.value / ctypes.sizeof(ctypes.c_ulong())

        pid_process = [i for i in l_pid_process][:int(n_returned)]
        for pid in pid_process:
            proc = {
                "pid": int(pid)
            }
            h_process = ctypes.windll.kernel32.OpenProcess(process_query_information | process_vm_read, False, pid)
            if h_process:
                ctypes.windll.psapi.EnumProcessModules(
                    h_process,
                    ctypes.byref(h_module),
                    ctypes.sizeof(h_module),
                    ctypes.byref(count)
                )
                ctypes.windll.psapi.GetModuleBaseNameA(h_process, h_module.value, modname, ctypes.sizeof(modname))
                proc["name"] = modname.value.decode('utf-8')
                ctypes.windll.kernel32.CloseHandle(h_process)
            processes.append(proc)
        return processes

    @staticmethod
    def processes_from_name(process_name):
        processes = []
        for process in Process.list():
            if (
                process_name == process.get("name", None) or
                (process.get("name", "").lower().endswith(".exe") and
                 process.get("name", "")[:-4] == process_name)
            ):
                processes.append(process)

        if len(processes) > 0:
            return processes

    @staticmethod
    def name_from_process(dw_process_id):
        process_list = Process.list()
        for process in process_list:
            if process.get('pid', -1) == dw_process_id:
                return process.get("name", None)

        return False

    def _open(self, dw_process_id, debug=False):
        if debug:
            ppsid_owner = wintypes.DWORD()
            ppsid_group = wintypes.DWORD()
            pp_dacl = wintypes.DWORD()
            pp_sacl = wintypes.DWORD()
            pp_security_descriptor = SecurityDescriptor()

            process = ctypes.windll.kernel32.OpenProcess(262144, 0, dw_process_id)
            ctypes.windll.advapi32.GetSecurityInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                6,
                0,
                ctypes.byref(ppsid_owner),
                ctypes.byref(ppsid_group),
                ctypes.byref(pp_dacl),
                ctypes.byref(pp_sacl),
                ctypes.byref(pp_security_descriptor)
            )
            ctypes.windll.advapi32.SetSecurityInfo(
                process,
                6,
                DACL_SECURITY_INFORMATION | UNPROTECTED_DACL_SECURITY_INFORMATION,
                None,
                None,
                pp_security_descriptor.dacl,
                pp_security_descriptor.group
            )
            ctypes.windll.kernel32.CloseHandle(process)
        self.h_process = ctypes.windll.kernel32.OpenProcess(2035711, 0, dw_process_id)
        if self.h_process is not None:
            self.is_process_open = True
            self.pid = dw_process_id
            return True
        return False

    def close(self):
        if self.h_process is not None:
            ret = ctypes.windll.kernel32.CloseHandle(self.h_process) == 1
            if ret:
                self.h_process = None
                self.pid = None
                self.is_process_open = False
            return ret
        return False

    def _open_from_name(self, process_name, debug=False):
        processes = self.processes_from_name(process_name)
        # need to remove the self process
        processes = list(filter(lambda process: process.get('pid', -1) != os.getpid(), processes))
        if not processes:
            raise ProcessException("can't get pid from name %s" % process_name)
        elif len(processes) > 1:
            raise ValueError(
                "There is multiple processes with name %s. Please select a process from its pid instead" % process_name
            )

        if debug:
            self._open(processes[0]["pid"], debug=True)
        else:
            self._open(processes[0]["pid"], debug=False)

    @staticmethod
    def get_system_info():
        si = SystemInfo()
        ctypes.windll.kernel32.GetSystemInfo(ctypes.byref(si))
        return si

    @staticmethod
    def get_native_system_info():
        si = SystemInfo()
        ctypes.windll.kernel32.GetNativeSystemInfo(ctypes.byref(si))
        return si

    def virtual_query_ex(self, lp_address):
        mbi = MemoryBasicInformation()
        if not VirtualQueryEx(self.h_process, lp_address, ctypes.byref(mbi), ctypes.sizeof(mbi)):
            raise ProcessException('Error VirtualQueryEx: 0x%08X' % lp_address)
        return mbi

    def virtual_protect_ex(self, base_address, size, protection):
        old_protect = ctypes.c_ulong(0)
        if not ctypes.windll.kernel32.virtual_protect_ex(self.h_process, base_address, size, protection, ctypes.byref(old_protect)):
            raise ProcessException('Error: VirtualProtectEx(%08X, %d, %08X)' % (base_address, size, protection))
        return old_protect.value

    def iter_region(self, start_offset=None, end_offset=None, protect=None, optimizations=None):
        _offset_start = start_offset or self.min_address
        _offset_end = end_offset or self.max_address

        while True:
            if _offset_start >= _offset_end:
                break
            mbi = self.virtual_query_ex(_offset_start)
            _offset_start = mbi.BaseAddress
            _chunk = mbi.RegionSize
            _protect = mbi.Protect
            _state = mbi.State

            if _state & MEM_FREE or _state & MEM_RESERVE:
                _offset_start += _chunk
                continue
            if protect:
                if not _protect & protect or _protect & PAGE_NOCACHE or _protect & PAGE_WRITECOMBINE or _protect & PAGE_GUARD:
                    _offset_start += _chunk
                    continue
            yield _offset_start, _chunk
            _offset_start += _chunk

    def write_bytes(self, address, data):
        address = int(address)
        if not self.is_process_open:
            raise ProcessException("Can't write_bytes(%s, %s), process %s is not open" % (address, data, self.pid))
        buffer = ctypes.create_string_buffer(data)
        size_writen = ctypes.c_size_t(0)
        buffer_size = ctypes.sizeof(buffer) - 1
        _address = address
        _length = buffer_size + 1
        try:
            old_protect = self.virtual_protect_ex(_address, _length, PAGE_EXECUTE_READWRITE)
        except Exception as ex1:
            print(str(ex1))

        res = ctypes.windll.kernel32.WriteProcessMemory(
            self.h_process,
            address,
            buffer,
            buffer_size,
            ctypes.byref(size_writen)
        )
        try:
            self.virtual_protect_ex(_address, _length, old_protect)
        except Exception as ex2:
            print(str(ex2))

        return res

    def read_bytes(self, address, size=4, use_nt_wow64_read_virtual_memory64=False):
        if use_nt_wow64_read_virtual_memory64:
            if NtWow64ReadVirtualMemory64 is None:
                raise WindowsError("NtWow64ReadVirtualMemory64 is not available from a 64bit process")
            rpm = NtWow64ReadVirtualMemory64
        else:
            rpm = ReadProcessMemory

        address = int(address)
        buffer = ctypes.create_string_buffer(size)
        bytes_read = ctypes.c_size_t(0)
        data = b''
        length = size
        while length:
            if rpm(self.h_process, address, buffer, size, ctypes.byref(bytes_read)) or (use_nt_wow64_read_virtual_memory64 and ctypes.GetLastError() == 0):
                if bytes_read.value:
                    data += buffer.raw[:bytes_read.value]
                    length -= bytes_read.value
                    address += bytes_read.value
                if not len(data):
                    raise ProcessException(
                        'Error %s in ReadProcessMemory(%08x, %d, read=%d)' % (
                            ctypes.GetLastError(),
                            address,
                            length,
                            bytes_read.value
                        )
                    )
                return data
            else:
                if ctypes.GetLastError() == 299:
                    data += buffer.raw[:bytes_read.value]
                    return data
                raise ctypes.WinError()
        return data

    def list_modules(self):
        if self.pid is not None:
            h_module_snap = CreateToolhelp32Snapshot(Th32csClass.SNAPMODULE, self.pid)
            if h_module_snap is not None:
                module_entry = ModuleEntry32()
                module_entry.dwSize = ctypes.sizeof(module_entry)
                success = Module32First(h_module_snap, ctypes.byref(module_entry))
                while success:
                    if module_entry.th32ProcessID == self.pid:
                        yield module_entry
                    success = Module32Next(h_module_snap, ctypes.byref(module_entry))

                ctypes.windll.kernel32.CloseHandle(h_module_snap)

    def get_symbolic_name(self, address):
        for m in self.list_modules():
            if int(ctypes.addressof(m.modBaseAddr.contents)) <= int(address) < int(ctypes.addressof(m.modBaseAddr.contents) + m.modBaseSize):
                return '%s+0x%08X' % (m.szModule, int(address) - ctypes.addressof(m.modBaseAddr.contents))

        return '0x%08X' % int(address)

    def has_module(self, module):
        if module[-4:] != '.dll':
            module += '.dll'
        module_list = self.list_modules()
        for m in module_list:
            if module in m.szExePath.split('\\'):
                return True
        return False


REGEX_TYPE = type(re.compile("^plop$"))

PAGE_READONLY = 2
PAGE_READWRITE = 4


class Worker(object):
    def __init__(self, pid=None, name=None, debug=True):
        self.process = Process(name=name, pid=pid, debug=debug)

    def __enter__(self):
        return self

    def __exit__(self):
        self.process.close()

    def address(self, value, default_type='uint'):
        return Address(value, process=self.process, default_type=default_type)

    def memory_replace_unicode(self, regex, replace):
        regex = regex.encode('utf-16-le')
        replace = replace.encode('utf-16-le')
        return self.memory_replace(re.compile(regex, re.UNICODE), replace)

    def memory_replace(self, regex, replace):
        succeed = True
        for _, start_offset in self.memory_search(regex, of_type='re'):
            if self.process.write_bytes(start_offset, replace) == 1:
                print('Write at offset %s succeeded !' % start_offset)
            else:
                succeed = False
                print('Write at offset %s failed !' % start_offset)

        return succeed

    def memory_search_unicode(self, regex):
        regex = regex.encode('utf-16-le')
        for _name, _address in self.memory_search(regex, of_type='re'):
            yield _name, _address

    def group_search(self, group, start_offset=None, end_offset=None):
        regex = ''
        for value, of_type in group:
            if of_type == 'f' or of_type == 'float':
                float_value = struct.pack('<f', float(value))
                regex += '..' + float_value[2:4]
            else:
                raise NotImplementedError('unknown type %s' % of_type)

        return self.memory_search(regex, of_type='re', start_offset=start_offset, end_offset=end_offset)

    def search_address(self, search):
        _address = '%08X' % search
        print('searching address %s' % _address)
        regex = ''
        for i in range(len(_address) - 2, -1, -2):
            regex += binascii.unhexlify(_address[i:i + 2])

        for _, _address in self.memory_search(re.escape(regex), of_type='re'):
            yield _address

    def parse_re_function(self, byte_array, value, offset):
        for name, regex in value:
            for res in regex.finditer(byte_array):
                yield name, self.address(offset + res.start(), 'bytes')

    def parse_float_function(self, byte_array, value, offset):
        for index in range(0, len(byte_array)):
            try:
                structure_type, structure_len = type_unpack('float')
                temp_val = struct.unpack(structure_type, byte_array[index:index + 4])[0]
                if int(value) == int(temp_val):
                    s_offset = offset + index
                    yield self.address(s_offset, 'float')
            except Exception as ex:
                print(str(ex))

    @staticmethod
    def parse_named_groups_function(byte_array, value, _):
        for name, regex in value:
            for res in regex.finditer(byte_array):
                yield name, res.groupdict()

    @staticmethod
    def parse_groups_function(byte_array, value, _):
        for name, regex in value:
            for res in regex.finditer(byte_array):
                yield name, res.groups()

    def parse_any_function(self, byte_array, value, offset):
        index = byte_array.find(value)
        while index != -1:
            soffset = offset + index
            yield self.address(soffset, 'bytes')
            index = byte_array.find(value, index + 1)

    def memory_search(self, value, of_type='match', protect=PAGE_READWRITE | PAGE_READONLY, optimizations=None, start_offset=None, end_offset=None):
        if of_type == 're' or of_type == 'groups' or of_type == 'ngroups':
            if type(value) is not list:
                value = [value]

            temp = []
            for reg in value:
                if type(reg) is tuple:
                    name = reg[0]
                    if type(reg[1]) != REGEX_TYPE:
                        regex = re.compile(reg[1], re.IGNORECASE)
                    else:
                        regex = reg[1]
                elif type(reg) == REGEX_TYPE:
                    name = ''
                    regex = reg
                else:
                    name = ''
                    regex = re.compile(reg, re.IGNORECASE)

                temp.append((name, regex))
            value = temp
        elif (of_type != 'match' and
              of_type != 'group' and
              of_type != 're' and
              of_type != 'groups' and
              of_type != 'ngroups' and
              of_type != 'lambda'):
            structure_type, structure_len = type_unpack(of_type)
            value = struct.pack(structure_type, value)

        if of_type == 're':
            parser = self.parse_re_function
        elif of_type == 'groups':
            parser = self.parse_groups_function
        elif of_type == 'ngroups':
            parser = self.parse_named_groups_function
        elif of_type == 'float':
            parser = self.parse_float_function
        elif of_type == 'lambda':
            parser = value
        else:
            parser = self.parse_any_function

        if not self.process.is_process_open:
            raise ProcessException("Can't read_bytes, process %s is not open" % self.process.pid)

        for offset, chunk_size in self.process.iter_region(
            start_offset=start_offset,
            end_offset=end_offset,
            protect=protect,
            optimizations=optimizations
        ):
            bytes_array = b''
            current_offset = offset
            chunk_read = 0
            chunk_exc = False
            while chunk_read < chunk_size:
                try:
                    bytes_array += self.process.read_bytes(current_offset, chunk_size)
                except IOError as e:
                    print()
                    traceback.format_exc()
                    if e.errno == 13:
                        raise
                    else:
                        print(e)
                    chunk_exc = True
                    break
                except Exception as ex:
                    print(str(ex))
                    chunk_exc = True
                    break
                finally:
                    current_offset += chunk_size
                    chunk_read += chunk_size

            if chunk_exc:
                continue

            if bytes_array:
                if of_type == "lambda":
                    for result in parser(bytes_array, offset):
                        yield result
                else:
                    for result in parser(bytes_array, value, offset):
                        yield result


class Locator(object):
    def __init__(self, worker, of_type='unknown', start=None, end=None):
        self.worker = worker
        self.of_type = of_type
        self.last_iteration = {}
        self.last_value = None
        self.start = start
        self.end = end

    def find(self, value, erase_last=True):
        return self.feed(value, erase_last)

    def feed(self, value, erase_last=True):
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
                        Address(address, self.worker.process, of_type) for address in
                        self.worker.memory_search(value, of_type, start_offset=self.start, end_offset=self.end)
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
        return self.last_iteration

    def diff(self, erase_last=False):
        return self.get_modified_address(erase_last)

    def get_modified_address(self, erase_last=False):
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
