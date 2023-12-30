import ctypes
import os
import platform
import struct
from ctypes import wintypes

from joatmon.system.memory.core import (
    CreateToolhelp32Snapshot,
    DACL_SECURITY_INFORMATION,
    IsWow64Process,
    MEM_FREE,
    MEM_RESERVE,
    MemoryBasicInformation,
    Module32First,
    Module32Next,
    ModuleEntry32,
    NtWow64ReadVirtualMemory64,
    PAGE_EXECUTE_READWRITE,
    PAGE_GUARD,
    PAGE_NOCACHE,
    PAGE_WRITECOMBINE,
    ReadProcessMemory,
    SecurityDescriptor,
    SystemInfo,
    Th32csClass,
    type_unpack,
    UNPROTECTED_DACL_SECURITY_INFORMATION,
    VirtualQueryEx,
)


class ProcessException(Exception):
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


class BaseProcess(object):
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

    def __init__(self, *args, **kwargs):
        self.h_process = None
        self.pid = None
        self.is_process_open = False
        self.buffer = None
        self.buffer_len = 0

    def __del__(self):
        self.close()

    def close(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """

    def iter_region(self, *args, **kwargs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        raise NotImplementedError

    def write_bytes(self, address, data):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        raise NotImplementedError

    def read_bytes(self, address, size=4):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        raise NotImplementedError

    @staticmethod
    def get_symbolic_name(a):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return '0x%08X' % int(a)

    def read(self, address, of_type='uint', max_length=50, errors='raise'):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
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
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if of_type != 'bytes':
            struct_type, struct_length = type_unpack(of_type)
            return self.write_bytes(int(address), struct.pack(struct_type, data))
        else:
            return self.write_bytes(int(address), data)


class Process(BaseProcess):
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

    def __init__(self, pid=None, name=None, debug=True):
        super(Process, self).__init__()
        if pid:
            self._open(int(pid), debug=debug)
        elif name:
            self._open_from_name(name, debug=debug)
        else:
            raise ValueError('You need to instanciate process with at least a name or a pid')

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
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if '64' not in platform.machine():
            return False
        is_wow_64 = ctypes.c_bool(False)
        if IsWow64Process is None:
            return False
        if not IsWow64Process(self.h_process, ctypes.byref(is_wow_64)):
            raise ctypes.WinError()
        return not is_wow_64.value

    @staticmethod
    def list():
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
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

        pid_process = [i for i in l_pid_process][: int(n_returned)]
        for pid in pid_process:
            proc = {'pid': int(pid)}
            h_process = ctypes.windll.kernel32.OpenProcess(process_query_information | process_vm_read, False, pid)
            if h_process:
                ctypes.windll.psapi.EnumProcessModules(
                    h_process, ctypes.byref(h_module), ctypes.sizeof(h_module), ctypes.byref(count)
                )
                ctypes.windll.psapi.GetModuleBaseNameA(h_process, h_module.value, modname, ctypes.sizeof(modname))
                proc['name'] = modname.value.decode('utf-8')
                ctypes.windll.kernel32.CloseHandle(h_process)
            processes.append(proc)
        return processes

    @staticmethod
    def processes_from_name(process_name):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        processes = []
        for process in Process.list():
            if process_name == process.get('name', None) or (
                    process.get('name', '').lower().endswith('.exe') and process.get('name', '')[:-4] == process_name
            ):
                processes.append(process)

        if len(processes) > 0:
            return processes

    @staticmethod
    def name_from_process(dw_process_id):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        process_list = Process.list()
        for process in process_list:
            if process.get('pid', -1) == dw_process_id:
                return process.get('name', None)

        return False

    def _open(self, dw_process_id, debug=False):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
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
                ctypes.byref(pp_security_descriptor),
            )
            ctypes.windll.advapi32.SetSecurityInfo(
                process,
                6,
                DACL_SECURITY_INFORMATION | UNPROTECTED_DACL_SECURITY_INFORMATION,
                None,
                None,
                pp_security_descriptor.dacl,
                pp_security_descriptor.group,
            )
            ctypes.windll.kernel32.CloseHandle(process)
        self.h_process = ctypes.windll.kernel32.OpenProcess(2035711, 0, dw_process_id)
        if self.h_process is not None:
            self.is_process_open = True
            self.pid = dw_process_id
            return True
        return False

    def close(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if self.h_process is not None:
            ret = ctypes.windll.kernel32.CloseHandle(self.h_process) == 1
            if ret:
                self.h_process = None
                self.pid = None
                self.is_process_open = False
            return ret
        return False

    def _open_from_name(self, process_name, debug=False):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        processes = self.processes_from_name(process_name)
        # need to remove the self process
        processes = list(filter(lambda process: process.get('pid', -1) != os.getpid(), processes))
        if not processes:
            raise ProcessException("can't get pid from name %s" % process_name)
        elif len(processes) > 1:
            raise ValueError(
                'There is multiple processes with name %s. Please select a process from its pid instead' % process_name
            )

        if debug:
            self._open(processes[0]['pid'], debug=True)
        else:
            self._open(processes[0]['pid'], debug=False)

    @staticmethod
    def get_system_info():
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        si = SystemInfo()
        ctypes.windll.kernel32.GetSystemInfo(ctypes.byref(si))
        return si

    @staticmethod
    def get_native_system_info():
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        si = SystemInfo()
        ctypes.windll.kernel32.GetNativeSystemInfo(ctypes.byref(si))
        return si

    def virtual_query_ex(self, lp_address):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        mbi = MemoryBasicInformation()
        if not VirtualQueryEx(self.h_process, lp_address, ctypes.byref(mbi), ctypes.sizeof(mbi)):
            raise ProcessException('Error VirtualQueryEx: 0x%08X' % lp_address)
        return mbi

    def virtual_protect_ex(self, base_address, size, protection):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        old_protect = ctypes.c_ulong(0)
        if not ctypes.windll.kernel32.virtual_protect_ex(
                self.h_process, base_address, size, protection, ctypes.byref(old_protect)
        ):
            raise ProcessException('Error: VirtualProtectEx(%08X, %d, %08X)' % (base_address, size, protection))
        return old_protect.value

    def iter_region(self, start_offset=None, end_offset=None, protect=None, optimizations=None):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
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
                if (
                        not _protect & protect
                        or _protect & PAGE_NOCACHE
                        or _protect & PAGE_WRITECOMBINE
                        or _protect & PAGE_GUARD
                ):
                    _offset_start += _chunk
                    continue
            yield _offset_start, _chunk
            _offset_start += _chunk

    def write_bytes(self, address, data):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
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
            self.h_process, address, buffer, buffer_size, ctypes.byref(size_writen)
        )
        try:
            self.virtual_protect_ex(_address, _length, old_protect)
        except Exception as ex2:
            print(str(ex2))

        return res

    def read_bytes(self, address, size=4, use_nt_wow64_read_virtual_memory64=False):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if use_nt_wow64_read_virtual_memory64:
            if NtWow64ReadVirtualMemory64 is None:
                raise WindowsError('NtWow64ReadVirtualMemory64 is not available from a 64bit process')
            rpm = NtWow64ReadVirtualMemory64
        else:
            rpm = ReadProcessMemory

        address = int(address)
        buffer = ctypes.create_string_buffer(size)
        bytes_read = ctypes.c_size_t(0)
        data = b''
        length = size
        while length:
            if rpm(self.h_process, address, buffer, size, ctypes.byref(bytes_read)) or (
                    use_nt_wow64_read_virtual_memory64 and ctypes.GetLastError() == 0
            ):
                if bytes_read.value:
                    data += buffer.raw[: bytes_read.value]
                    length -= bytes_read.value
                    address += bytes_read.value
                if not len(data):
                    raise ProcessException(
                        'Error %s in ReadProcessMemory(%08x, %d, read=%d)'
                        % (ctypes.GetLastError(), address, length, bytes_read.value)
                    )
                return data
            else:
                if ctypes.GetLastError() == 299:
                    data += buffer.raw[: bytes_read.value]
                    return data
                raise ctypes.WinError()
        return data

    def list_modules(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
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
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for m in self.list_modules():
            if (
                    int(ctypes.addressof(m.modBaseAddr.contents))
                    <= int(address)
                    < int(ctypes.addressof(m.modBaseAddr.contents) + m.modBaseSize)
            ):
                return '%s+0x%08X' % (m.szModule, int(address) - ctypes.addressof(m.modBaseAddr.contents))

        return '0x%08X' % int(address)

    def has_module(self, module):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if module[-4:] != '.dll':
            module += '.dll'
        module_list = self.list_modules()
        for m in module_list:
            if module in m.szExePath.split('\\'):
                return True
        return False
