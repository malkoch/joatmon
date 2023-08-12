import binascii
import re
import struct
import traceback

from joatmon.system.memory.address import Address
from joatmon.system.memory.core import type_unpack
from joatmon.system.memory.process import (
    Process,
    ProcessException
)

REGEX_TYPE = type(re.compile('^plop$'))

PAGE_READONLY = 2
PAGE_READWRITE = 4


class Worker(object):
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
        self.process = Process(name=name, pid=pid, debug=debug)

    def __enter__(self):
        return self

    def __exit__(self):
        self.process.close()

    def address(self, value, default_type='uint'):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return Address(value, process=self.process, default_type=default_type)

    def memory_replace_unicode(self, regex, replace):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        regex = regex.encode('utf-16-le')
        replace = replace.encode('utf-16-le')
        return self.memory_replace(re.compile(regex, re.UNICODE), replace)

    def memory_replace(self, regex, replace):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        succeed = True
        for _, start_offset in self.memory_search(regex, of_type='re'):
            if self.process.write_bytes(start_offset, replace) == 1:
                print('Write at offset %s succeeded !' % start_offset)
            else:
                succeed = False
                print('Write at offset %s failed !' % start_offset)

        return succeed

    def memory_search_unicode(self, regex):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        regex = regex.encode('utf-16-le')
        for _name, _address in self.memory_search(regex, of_type='re'):
            yield _name, _address

    def group_search(self, group, start_offset=None, end_offset=None):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        regex = ''
        for value, of_type in group:
            if of_type == 'f' or of_type == 'float':
                float_value = struct.pack('<f', float(value))
                regex += '..' + float_value[2:4]
            else:
                raise NotImplementedError('unknown type %s' % of_type)

        return self.memory_search(regex, of_type='re', start_offset=start_offset, end_offset=end_offset)

    def search_address(self, search):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        _address = '%08X' % search
        print('searching address %s' % _address)
        regex = ''
        for i in range(len(_address) - 2, -1, -2):
            regex += binascii.unhexlify(_address[i: i + 2])

        for _, _address in self.memory_search(re.escape(regex), of_type='re'):
            yield _address

    def parse_re_function(self, byte_array, value, offset):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for name, regex in value:
            for res in regex.finditer(byte_array):
                yield name, self.address(offset + res.start(), 'bytes')

    def parse_float_function(self, byte_array, value, offset):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for index in range(0, len(byte_array)):
            try:
                structure_type, structure_len = type_unpack('float')
                temp_val = struct.unpack(structure_type, byte_array[index: index + 4])[0]
                if int(value) == int(temp_val):
                    s_offset = offset + index
                    yield self.address(s_offset, 'float')
            except Exception as ex:
                print(str(ex))

    @staticmethod
    def parse_named_groups_function(byte_array, value, _):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for name, regex in value:
            for res in regex.finditer(byte_array):
                yield name, res.groupdict()

    @staticmethod
    def parse_groups_function(byte_array, value, _):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for name, regex in value:
            for res in regex.finditer(byte_array):
                yield name, res.groups()

    def parse_any_function(self, byte_array, value, offset):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        index = byte_array.find(value)
        while index != -1:
            soffset = offset + index
            yield self.address(soffset, 'bytes')
            index = byte_array.find(value, index + 1)

    def memory_search(
            self,
            value,
            of_type='match',
            protect=PAGE_READWRITE | PAGE_READONLY,
            optimizations=None,
            start_offset=None,
            end_offset=None,
    ):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
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
        elif (
                of_type != 'match'
                and of_type != 'group'
                and of_type != 're'
                and of_type != 'groups'
                and of_type != 'ngroups'
                and of_type != 'lambda'
        ):
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
                start_offset=start_offset, end_offset=end_offset, protect=protect, optimizations=optimizations
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
                if of_type == 'lambda':
                    for result in parser(bytes_array, offset):
                        yield result
                else:
                    for result in parser(bytes_array, value, offset):
                        yield result
