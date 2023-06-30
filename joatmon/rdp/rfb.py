import asyncio
import io
import logging
import math
import sys
import zlib
from asyncio import (
    StreamReader,
    StreamWriter
)
from struct import (
    pack,
    unpack
)

from websockets.exceptions import ConnectionClosed
from websockets.protocol import WebSocketCommonProtocol


class ReaderAdapter:
    async def read(self, n=-1) -> bytes:
        ...

    def feed_eof(self):
        ...


class WriterAdapter:
    def write(self, data):
        ...

    async def drain(self):
        ...

    def get_peer_info(self):
        ...

    async def close(self):
        ...


class WebSocketsReader(ReaderAdapter):
    def __init__(self, protocol: WebSocketCommonProtocol):
        self._protocol = protocol
        self._stream = io.BytesIO(b'')

    async def read(self, n=-1) -> bytes:
        await self._feed_buffer(n)
        data = self._stream.read(n)
        return data

    async def _feed_buffer(self, n=1):
        buffer = bytearray(self._stream.read())
        while len(buffer) < n:
            try:
                message = await self._protocol.recv()
            except ConnectionClosed:
                message = None
            if message is None:
                break
            if not isinstance(message, bytes):
                raise TypeError("message must be bytes")
            buffer.extend(message)
        self._stream = io.BytesIO(buffer)


class WebSocketsWriter(WriterAdapter):
    def __init__(self, protocol: WebSocketCommonProtocol):
        self._protocol = protocol
        self._stream = io.BytesIO(b'')

    def write(self, data):
        self._stream.write(data)

    async def drain(self):
        data = self._stream.getvalue()
        if len(data):
            await self._protocol.send(data)
        self._stream = io.BytesIO(b'')

    def get_peer_info(self):
        return self._protocol.remote_address

    async def close(self):
        await self._protocol.close()


class StreamReaderAdapter(ReaderAdapter):
    def __init__(self, reader: StreamReader):
        self._reader = reader

    async def read(self, n=-1) -> bytes:
        if n == -1:
            data = await self._reader.read(n)
        else:
            data = await self._reader.readexactly(n)
        return data

    def feed_eof(self):
        return self._reader.feed_eof()


class StreamWriterAdapter(WriterAdapter):
    def __init__(self, writer: StreamWriter):
        self.logger = logging.getLogger(__name__)
        self._writer = writer
        self.is_closed = False

    def write(self, data):
        if not self.is_closed:
            self._writer.write(data)

    async def drain(self):
        if not self.is_closed:
            await self._writer.drain()

    def get_peer_info(self):
        extra_info = self._writer.get_extra_info('peername')
        return extra_info[0], extra_info[1]

    async def close(self):
        if not self.is_closed:
            self.is_closed = True
            await self._writer.drain()
            if self._writer.can_write_eof():
                self._writer.write_eof()
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except AttributeError:
                pass


class BufferReader(ReaderAdapter):
    def __init__(self, buffer: bytes):
        self._stream = io.BytesIO(buffer)

    async def read(self, n=-1) -> bytes:
        return self._stream.read(n)


class BufferWriter(WriterAdapter):
    def __init__(self, buffer=b''):
        self._stream = io.BytesIO(buffer)

    def write(self, data):
        self._stream.write(data)

    async def drain(self):
        pass

    def get_buffer(self):
        return self._stream.getvalue()

    def get_peer_info(self):
        return "BufferWriter", 0

    async def close(self):
        self._stream.close()


ECB = 0
CBC = 1

PAD_NORMAL = 1
PAD_PKCS5 = 2


class BaseDES(object):
    def __init__(self, mode=ECB, iv=None, pad=None, padmode=PAD_NORMAL):
        if iv:
            iv = self._guard_against_unicode(iv)
        if pad:
            pad = self._guard_against_unicode(pad)
        self.block_size = 8

        if pad and padmode == PAD_PKCS5:
            raise ValueError("Cannot use a pad character with PAD_PKCS5")
        if iv and len(iv) != self.block_size:
            raise ValueError("Invalid Initial Value (IV), must be a multiple of " + str(self.block_size) + " bytes")

        self._mode = mode
        self._iv = iv
        self._padding = pad
        self._padmode = padmode

        self.__key = b''

    def get_key(self):
        return self.__key

    def set_key(self, key):
        key = self._guard_against_unicode(key)
        self.__key = key

    def get_mode(self):
        return self._mode

    def set_mode(self, mode):
        self._mode = mode

    def get_padding(self):
        return self._padding

    def set_padding(self, pad):
        if pad is not None:
            pad = self._guard_against_unicode(pad)
        self._padding = pad

    def get_pad_mode(self):
        return self._padmode

    def set_pad_mode(self, mode):
        self._padmode = mode

    def get_iv(self):
        return self._iv

    def set_iv(self, iv):
        if not iv or len(iv) != self.block_size:
            raise ValueError("Invalid Initial Value (IV), must be a multiple of " + str(self.block_size) + " bytes")
        iv = self._guard_against_unicode(iv)
        self._iv = iv

    def _pad_data(self, data, pad, padmode):
        if padmode is None:
            padmode = self.get_pad_mode()
        if pad and padmode == PAD_PKCS5:
            raise ValueError("Cannot use a pad character with PAD_PKCS5")

        if padmode == PAD_NORMAL:
            if len(data) % self.block_size == 0:
                return data

            if not pad:
                pad = self.get_padding()
            if not pad:
                raise ValueError("Data must be a multiple of " + str(self.block_size) + " bytes in length. Use padmode=PAD_PKCS5 or set the pad character.")
            data += (self.block_size - (len(data) % self.block_size)) * pad

        elif padmode == PAD_PKCS5:
            pad_len = 8 - (len(data) % self.block_size)
            data += bytes([pad_len] * pad_len)

        return data

    def _unpad_data(self, data, pad, padmode):
        if not data:
            return data
        if pad and padmode == PAD_PKCS5:
            raise ValueError("Cannot use a pad character with PAD_PKCS5")
        if padmode is None:
            padmode = self.get_pad_mode()

        if padmode == PAD_NORMAL:
            if not pad:
                pad = self.get_padding()
            if pad:
                data = data[:-self.block_size] + data[-self.block_size:].rstrip(pad)

        elif padmode == PAD_PKCS5:
            pad_len = data[-1]
            data = data[:-pad_len]

        return data

    @staticmethod
    def _guard_against_unicode(data):
        if isinstance(data, str):
            try:
                return data.encode('ascii')
            except UnicodeEncodeError:
                pass
            raise ValueError("pyDes can only work with encoded strings, not Unicode.")
        return data


class DES(BaseDES):
    __pc1 = [56, 48, 40, 32, 24, 16, 8,
             0, 57, 49, 41, 33, 25, 17,
             9, 1, 58, 50, 42, 34, 26,
             18, 10, 2, 59, 51, 43, 35,
             62, 54, 46, 38, 30, 22, 14,
             6, 61, 53, 45, 37, 29, 21,
             13, 5, 60, 52, 44, 36, 28,
             20, 12, 4, 27, 19, 11, 3
             ]

    __left_rotations = [
        1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1
    ]

    __pc2 = [
        13, 16, 10, 23, 0, 4,
        2, 27, 14, 5, 20, 9,
        22, 18, 11, 3, 25, 7,
        15, 6, 26, 19, 12, 1,
        40, 51, 30, 36, 46, 54,
        29, 39, 50, 44, 32, 47,
        43, 48, 38, 55, 33, 52,
        45, 41, 49, 35, 28, 31
    ]

    __ip = [57, 49, 41, 33, 25, 17, 9, 1,
            59, 51, 43, 35, 27, 19, 11, 3,
            61, 53, 45, 37, 29, 21, 13, 5,
            63, 55, 47, 39, 31, 23, 15, 7,
            56, 48, 40, 32, 24, 16, 8, 0,
            58, 50, 42, 34, 26, 18, 10, 2,
            60, 52, 44, 36, 28, 20, 12, 4,
            62, 54, 46, 38, 30, 22, 14, 6
            ]

    __expansion_table = [
        31, 0, 1, 2, 3, 4,
        3, 4, 5, 6, 7, 8,
        7, 8, 9, 10, 11, 12,
        11, 12, 13, 14, 15, 16,
        15, 16, 17, 18, 19, 20,
        19, 20, 21, 22, 23, 24,
        23, 24, 25, 26, 27, 28,
        27, 28, 29, 30, 31, 0
    ]

    __sbox = [
        [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7,
         0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8,
         4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0,
         15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],

        [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10,
         3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5,
         0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15,
         13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],

        [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8,
         13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1,
         13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7,
         1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],

        [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15,
         13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9,
         10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4,
         3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],

        [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9,
         14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6,
         4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14,
         11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],

        [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11,
         10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8,
         9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6,
         4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],

        [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1,
         13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6,
         1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2,
         6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],

        [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7,
         1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2,
         7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8,
         2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
    ]

    __p = [
        15, 6, 19, 20, 28, 11,
        27, 16, 0, 14, 22, 25,
        4, 17, 30, 9, 1, 7,
        23, 13, 31, 26, 2, 8,
        18, 12, 29, 5, 21, 10,
        3, 24
    ]

    __fp = [
        39, 7, 47, 15, 55, 23, 63, 31,
        38, 6, 46, 14, 54, 22, 62, 30,
        37, 5, 45, 13, 53, 21, 61, 29,
        36, 4, 44, 12, 52, 20, 60, 28,
        35, 3, 43, 11, 51, 19, 59, 27,
        34, 2, 42, 10, 50, 18, 58, 26,
        33, 1, 41, 9, 49, 17, 57, 25,
        32, 0, 40, 8, 48, 16, 56, 24
    ]

    ENCRYPT = 0x00
    DECRYPT = 0x01

    def __init__(self, key, mode=ECB, iv=None, pad=None, padmode=PAD_NORMAL):
        if len(key) != 8:
            raise ValueError("Invalid DES key size. Key must be exactly 8 bytes long.")
        BaseDES.__init__(self, mode, iv, pad, padmode)
        self.key_size = 8

        self.L = []
        self.R = []
        self.Kn = [[0] * 48] * 16
        self.final = []

        self.set_key(key)

    def set_key(self, key):
        BaseDES.set_key(self, key)
        self.__create_sub_keys()

    @staticmethod
    def __string_to_bit_list(data):
        if isinstance(data[0], str):
            data = [ord(c) for c in data]
        length = len(data) * 8
        result = [0] * length
        pos = 0
        for ch in data:
            i = 7
            while i >= 0:
                if ch & (1 << i) != 0:
                    result[pos] = 1
                else:
                    result[pos] = 0
                pos += 1
                i -= 1

        return result

    @staticmethod
    def __bit_list_to_string(data):
        result = []
        pos = 0
        c = 0
        while pos < len(data):
            c += data[pos] << (7 - (pos % 8))
            if (pos % 8) == 7:
                result.append(c)
                c = 0
            pos += 1

        return bytes(result)

    @staticmethod
    def __permutate(table, block):
        return list(map(lambda x: block[x], table))

    def __create_sub_keys(self):
        key = self.__permutate(DES.__pc1, self.__string_to_bit_list(self.get_key()))
        i = 0
        self.L = key[:28]
        self.R = key[28:]
        while i < 16:
            j = 0
            while j < DES.__left_rotations[i]:
                self.L.append(self.L[0])
                del self.L[0]

                self.R.append(self.R[0])
                del self.R[0]

                j += 1

            self.Kn[i] = self.__permutate(DES.__pc2, self.L + self.R)

            i += 1

    def __des_crypt(self, block, crypt_type):
        block = self.__permutate(DES.__ip, block)
        self.L = block[:32]
        self.R = block[32:]

        if crypt_type == DES.ENCRYPT:
            iteration = 0
            iteration_adjustment = 1
        else:
            iteration = 15
            iteration_adjustment = -1

        i = 0
        while i < 16:
            temp_r = self.R[:]

            self.R = self.__permutate(DES.__expansion_table, self.R)

            self.R = list(map(lambda x, y: x ^ y, self.R, self.Kn[iteration]))
            b = [self.R[:6], self.R[6:12], self.R[12:18], self.R[18:24], self.R[24:30], self.R[30:36], self.R[36:42], self.R[42:]]

            j = 0
            bn = [0] * 32
            pos = 0
            while j < 8:
                m = (b[j][0] << 1) + b[j][5]
                n = (b[j][1] << 3) + (b[j][2] << 2) + (b[j][3] << 1) + b[j][4]

                v = DES.__sbox[j][(m << 4) + n]

                bn[pos] = (v & 8) >> 3
                bn[pos + 1] = (v & 4) >> 2
                bn[pos + 2] = (v & 2) >> 1
                bn[pos + 3] = v & 1

                pos += 4
                j += 1

            self.R = self.__permutate(DES.__p, bn)

            self.R = list(map(lambda x, y: x ^ y, self.R, self.L))
            self.L = temp_r

            i += 1
            iteration += iteration_adjustment

        self.final = self.__permutate(DES.__fp, self.R + self.L)
        return self.final

    def crypt(self, data, crypt_type):
        if not data:
            return ''
        if len(data) % self.block_size != 0:
            if crypt_type == DES.DECRYPT:
                raise ValueError("Invalid data length, data must be a multiple of " + str(self.block_size) + " bytes\n.")
            if not self.get_padding():
                raise ValueError("Invalid data length, data must be a multiple of " + str(self.block_size) + " bytes\n. Try setting the optional padding character")
            else:
                data += (self.block_size - (len(data) % self.block_size)) * self.get_padding()

        if self.get_mode() == CBC:
            if self.get_iv():
                iv = self.__string_to_bit_list(self.get_iv())
            else:
                raise ValueError("For CBC mode, you must supply the Initial Value (IV) for ciphering")

        i = 0
        result = []
        while i < len(data):
            block = self.__string_to_bit_list(data[i:i + 8])

            if self.get_mode() == CBC:
                if crypt_type == DES.ENCRYPT:
                    block = list(map(lambda x, y: x ^ y, block, iv))

                processed_block = self.__des_crypt(block, crypt_type)

                if crypt_type == DES.DECRYPT:
                    processed_block = list(map(lambda x, y: x ^ y, processed_block, iv))
                    iv = block
                else:
                    iv = processed_block
            else:
                processed_block = self.__des_crypt(block, crypt_type)

            result.append(self.__bit_list_to_string(processed_block))
            i += 8

        return bytes.fromhex('').join(result)

    def encrypt(self, data, pad=None, padmode=None):
        data = self._guard_against_unicode(data)
        if pad is not None:
            pad = self._guard_against_unicode(pad)
        data = self._pad_data(data, pad, padmode)
        return self.crypt(data, DES.ENCRYPT)

    def decrypt(self, data, pad=None, padmode=None):
        data = self._guard_against_unicode(data)
        if pad is not None:
            pad = self._guard_against_unicode(pad)
        data = self.crypt(data, DES.DECRYPT)
        return self._unpad_data(data, pad, padmode)


class TripleDES(BaseDES):
    def __init__(self, key, mode=ECB, iv=None, pad=None, padmode=PAD_NORMAL):
        BaseDES.__init__(self, mode, iv, pad, padmode)
        self.set_key(key)

        self.key_size = 0
        self.__key1 = None
        self.__key2 = None
        self.__key3 = None

    def set_key(self, key):
        self.key_size = 24
        if len(key) != self.key_size:
            if len(key) == 16:
                self.key_size = 16
            else:
                raise ValueError("Invalid triple DES key size. Key must be either 16 or 24 bytes long")
        if self.get_mode() == CBC:
            if not self.get_iv():
                self._iv = key[:self.block_size]
            if len(self.get_iv()) != self.block_size:
                raise ValueError("Invalid IV, must be 8 bytes in length")
        self.__key1 = DES(key[:8], self._mode, self._iv, self._padding, self._padmode)
        self.__key2 = DES(key[8:16], self._mode, self._iv, self._padding, self._padmode)
        if self.key_size == 16:
            self.__key3 = self.__key1
        else:
            self.__key3 = DES(key[16:], self._mode, self._iv, self._padding, self._padmode)
        BaseDES.set_key(self, key)

    def set_mode(self, mode):
        BaseDES.set_mode(self, mode)
        for key in (self.__key1, self.__key2, self.__key3):
            key.set_mode(mode)

    def set_padding(self, pad):
        BaseDES.set_padding(self, pad)
        for key in (self.__key1, self.__key2, self.__key3):
            key.set_padding(pad)

    def set_pad_mode(self, mode):
        BaseDES.set_pad_mode(self, mode)
        for key in (self.__key1, self.__key2, self.__key3):
            key.set_pad_mode(mode)

    def set_iv(self, iv):
        BaseDES.set_iv(self, iv)
        for key in (self.__key1, self.__key2, self.__key3):
            key.set_iv(iv)

    def encrypt(self, data, pad=None, padmode=None):
        encrypt = DES.ENCRYPT
        decrypt = DES.DECRYPT
        data = self._guard_against_unicode(data)
        if pad is not None:
            pad = self._guard_against_unicode(pad)

        data = self._pad_data(data, pad, padmode)
        if self.get_mode() == CBC:
            self.__key1.set_iv(self.get_iv())
            self.__key2.set_iv(self.get_iv())
            self.__key3.set_iv(self.get_iv())
            i = 0
            result = []
            while i < len(data):
                block = self.__key1.crypt(data[i:i + 8], encrypt)
                block = self.__key2.crypt(block, decrypt)
                block = self.__key3.crypt(block, encrypt)
                self.__key1.set_iv(block)
                self.__key2.set_iv(block)
                self.__key3.set_iv(block)
                result.append(block)
                i += 8
            return bytes.fromhex('').join(result)
        else:
            data = self.__key1.crypt(data, encrypt)
            data = self.__key2.crypt(data, decrypt)
            return self.__key3.crypt(data, encrypt)

    def decrypt(self, data, pad=None, padmode=None):
        encrypt = DES.ENCRYPT
        decrypt = DES.DECRYPT
        data = self._guard_against_unicode(data)
        if pad is not None:
            pad = self._guard_against_unicode(pad)
        if self.get_mode() == CBC:
            self.__key1.set_iv(self.get_iv())
            self.__key2.set_iv(self.get_iv())
            self.__key3.set_iv(self.get_iv())
            i = 0
            result = []
            while i < len(data):
                iv = data[i:i + 8]
                block = self.__key3.crypt(iv, decrypt)
                block = self.__key2.crypt(block, encrypt)
                block = self.__key1.crypt(block, decrypt)
                self.__key1.set_iv(iv)
                self.__key2.set_iv(iv)
                self.__key3.set_iv(iv)
                result.append(block)
                i += 8
            data = bytes.fromhex('').join(result)
        else:
            data = self.__key3.crypt(data, decrypt)
            data = self.__key2.crypt(data, encrypt)
            data = self.__key1.crypt(data, decrypt)
        return self._unpad_data(data, pad, padmode)


if sys.version_info[0] >= 3:
    original_ord = ord


    def ord(x):
        if isinstance(x, int):
            return x
        elif isinstance(x, bytes) or isinstance(x, str):
            return original_ord(x)
        else:
            raise TypeError(f"our customized ord takes an int, a byte, or a str. Got {type(x)} : {x}")

RAW_ENCODING = 0
COPY_RECTANGLE_ENCODING = 1
RRE_ENCODING = 2
CORRE_ENCODING = 4
HEXTILE_ENCODING = 5
ZLIB_ENCODING = 6
TIGHT_ENCODING = 7
ZLIBHEX_ENCODING = 8
ZRLE_ENCODING = 16
PSEUDO_CURSOR_ENCODING = -239
PSEUDO_DESKTOP_SIZE_ENCODING = -223

KEY_BackSpace = 0xff08
KEY_Tab = 0xff09
KEY_Return = 0xff0d
KEY_Escape = 0xff1b
KEY_Insert = 0xff63
KEY_Delete = 0xffff
KEY_Home = 0xff50
KEY_End = 0xff57
KEY_PageUp = 0xff55
KEY_PageDown = 0xff56
KEY_Left = 0xff51
KEY_Up = 0xff52
KEY_Right = 0xff53
KEY_Down = 0xff54
KEY_F1 = 0xffbe
KEY_F2 = 0xffbf
KEY_F3 = 0xffc0
KEY_F4 = 0xffc1
KEY_F5 = 0xffc2
KEY_F6 = 0xffc3
KEY_F7 = 0xffc4
KEY_F8 = 0xffc5
KEY_F9 = 0xffc6
KEY_F10 = 0xffc7
KEY_F11 = 0xffc8
KEY_F12 = 0xffc9
KEY_F13 = 0xFFCA
KEY_F14 = 0xFFCB
KEY_F15 = 0xFFCC
KEY_F16 = 0xFFCD
KEY_F17 = 0xFFCE
KEY_F18 = 0xFFCF
KEY_F19 = 0xFFD0
KEY_F20 = 0xFFD1
KEY_ShiftLeft = 0xffe1
KEY_ShiftRight = 0xffe2
KEY_ControlLeft = 0xffe3
KEY_ControlRight = 0xffe4
KEY_MetaLeft = 0xffe7
KEY_MetaRight = 0xffe8
KEY_AltLeft = 0xffe9
KEY_AltRight = 0xffea

KEY_Scroll_Lock = 0xFF14
KEY_Sys_Req = 0xFF15
KEY_Num_Lock = 0xFF7F
KEY_Caps_Lock = 0xFFE5
KEY_Pause = 0xFF13
KEY_Super_L = 0xFFEB
KEY_Super_R = 0xFFEC
KEY_Hyper_L = 0xFFED
KEY_Hyper_R = 0xFFEE

KEY_KP_0 = 0xFFB0
KEY_KP_1 = 0xFFB1
KEY_KP_2 = 0xFFB2
KEY_KP_3 = 0xFFB3
KEY_KP_4 = 0xFFB4
KEY_KP_5 = 0xFFB5
KEY_KP_6 = 0xFFB6
KEY_KP_7 = 0xFFB7
KEY_KP_8 = 0xFFB8
KEY_KP_9 = 0xFFB9
KEY_KP_Enter = 0xFF8D

KEY_ForwardSlash = 0x002F
KEY_BackSlash = 0x005C
KEY_SpaceBar = 0x0020


def _zrle_next_bit(it, pixels_in_tile):
    num_pixels = 0
    while True:
        b = ord(next(it))

        for n in range(8):
            value = b >> (7 - n)
            yield value & 1

            num_pixels += 1
            if num_pixels == pixels_in_tile:
                return


def _zrle_next_dibit(it, pixels_in_tile):
    num_pixels = 0
    while True:
        b = ord(next(it))

        for n in range(0, 8, 2):
            value = b >> (6 - n)
            yield value & 3

            num_pixels += 1
            if num_pixels == pixels_in_tile:
                return


def _zrle_next_nibble(it, pixels_in_tile):
    num_pixels = 0
    while True:
        b = ord(next(it))

        for n in range(0, 8, 4):
            value = b >> (4 - n)
            yield value & 15

            num_pixels += 1
            if num_pixels == pixels_in_tile:
                return


class RFBClient:
    def __init__(self, host, port, loop):
        self.host = host
        self.port = port
        self.loop = loop

        self.reader = None
        self.writer = None

        self.version = None
        self.server_version = None
        self.password = None
        self.shared = 0

        self._zlib_stream = zlib.decompressobj(0)

    async def connect(self):
        conn_reader, conn_writer = await asyncio.open_connection(self.host, self.port, loop=self.loop)
        self.reader = StreamReaderAdapter(conn_reader)
        self.writer = StreamWriterAdapter(conn_writer)

        buffer = b''
        while True:
            byte = await self.reader.read(1)
            buffer += byte
            if byte == b'\n':
                break

        version = 3.3
        if buffer[:3] == b'RFB':
            version_server = float(buffer[3:-1].replace(b'0', b''))
            self.server_version = version_server
            supported_versions = (3.3, 3.7, 3.8)
            if version_server in supported_versions:
                version = version_server
            else:
                print("Protocol version %.3f not supported" % version_server)
                version = max(filter(lambda x: x <= version_server, supported_versions))
        print("Using protocol version %.3f" % version)
        parts = str(version).split('.')
        self.writer.write(bytes(b"RFB %03d.%03d\n" % (int(parts[0]), int(parts[1]))))

        self.version = version
        if version < 3.7:
            await self._handle_auth()
        else:
            await self._handle_number_security_types()

    async def _handle_auth(self):
        buffer = await self.reader.read(4)

        (auth,) = unpack("!I", buffer)
        if auth == 0:
            await self._handle_conn_failed()
        elif auth == 1:
            await self._do_client_initialization()
            return
        elif auth == 2:
            await self._handle_vnc_auth()
        else:
            print("unknown auth response (%d)" % auth)

    async def _handle_number_security_types(self):
        buffer = await self.reader.read(1)
        num_types, = unpack("!B", buffer)
        if num_types:
            num_types = await self.reader.read(num_types)
            types = unpack("!%dB" % len(num_types), num_types)
            supported_types = (1, 2)
            valid_types = [sec_type for sec_type in types if sec_type in supported_types]
            if valid_types:
                sec_type = max(valid_types)
                self.writer.write(pack("!B", sec_type))
                if sec_type == 1:
                    if self.version < 3.8:
                        await self._do_client_initialization()
                    else:
                        await self._handle_vnc_auth_result()
                else:
                    await self._handle_vnc_auth()
            else:
                print("unknown security types: %s" % repr(types))
        else:
            await self._handle_conn_failed()

    async def _handle_vnc_auth(self):
        buffer = await self.reader.read(16)
        self._challenge = buffer
        await self.vnc_request_password()
        await self._handle_vnc_auth_result()

    async def _handle_vnc_auth_result(self):
        buffer = await self.reader.read(4)
        (result,) = unpack("!I", buffer)
        if result == 0:
            await self._do_client_initialization()
            return
        elif result == 1:
            if self.version < 3.8:
                await self.vnc_auth_failed("authentication failed")
                await self.writer.close()
            else:
                await self._handle_auth_failed()
        elif result == 2:
            if self.version < 3.8:
                await self.vnc_auth_failed("too many tries to log in")
                await self.writer.close()
            else:
                await self._handle_auth_failed()
        else:
            print("unknown auth response (%d)" % result)

    async def _do_client_initialization(self):
        self.writer.write(pack("!B", self.shared))

        buffer = await self.reader.read(24)
        self.width, self.height, pixformat, namelen = unpack("!HH16sI", buffer)
        self.bpp, self.depth, self.bigendian, self.truecolor, self.redmax, self.greenmax, self.bluemax, self.redshift, self.greenshift, self.blueshift = unpack(
            "!BBBBHHHBBBxxx",
            pixformat
            )
        self.bypp = self.bpp // 8

        buffer = await self.reader.read(namelen)
        self.name = buffer

        await self.vnc_connection_made()
        # await self._handle_connection()

        asyncio.ensure_future(self._handle_connection(), loop=self.loop)

    async def _handle_auth_failed(self):
        buffer = await self.reader.read(4)
        waitfor, = unpack("!I", buffer)
        await self._handle_auth_failed_message(waitfor)

    async def _handle_auth_failed_message(self, waitfor):
        buffer = await self.reader.read(waitfor)
        await self.vnc_auth_failed(buffer)
        await self.writer.close()

    async def _handle_connection(self):
        while True:
            await asyncio.sleep(0.01, loop=self.loop)

            buffer = await self.reader.read(1)

            msgid, = unpack("!B", buffer)
            if msgid == 0:
                await self._handle_framebuffer_update()
            elif msgid == 2:
                await self.bell()
            elif msgid == 3:
                await self._handle_server_cut_text()
            else:
                print("unknown message received (id %d)" % msgid)

    async def _handle_conn_failed(self):
        buffer = await self.reader.read(4)
        waitfor, = unpack("!I", buffer)
        await self._handle_conn_message(waitfor)

    async def _handle_conn_message(self, waitfor):
        buffer = await self.reader.read(waitfor)
        print("Connection refused: %r" % buffer)

    async def _do_connection(self):
        if self.rectangles:
            await self._handle_rectangle()
        else:
            await self.commit_update(self.rectanglePos)

    async def _handle_rectangle(self):
        buffer = await self.reader.read(12)

        (x, y, width, height, encoding) = unpack("!HHHHi", buffer)
        if self.rectangles:
            self.rectangles -= 1
            self.rectanglePos.append((x, y, width, height))
            if encoding == COPY_RECTANGLE_ENCODING:
                await self._handle_decode_copyrect(x, y, width, height)
            elif encoding == RAW_ENCODING:
                await self._handle_decode_raw(x, y, width, height)
            elif encoding == HEXTILE_ENCODING:
                await self._do_next_hextile_subrect(None, None, x, y, width, height, None, None)
            elif encoding == CORRE_ENCODING:
                await self._handle_decode_corre(x, y, width, height)
            elif encoding == RRE_ENCODING:
                await self._handle_decode_rre(x, y, width, height)
            elif encoding == ZRLE_ENCODING:
                await self._handle_decode_zrle(x, y, width, height)
            elif encoding == PSEUDO_CURSOR_ENCODING:
                length = width * height * self.bypp
                length += int(math.floor((width + 7.0) / 8)) * height
                await self._handle_decode_psuedo_cursor(length, x, y, width, height)
            elif encoding == PSEUDO_DESKTOP_SIZE_ENCODING:
                await self._handle_decode_desktop_size(width, height)
            else:
                print("unknown encoding received (encoding %d)" % encoding)
                await self._do_connection()
        else:
            await self._do_connection()

    async def _handle_decode_copyrect(self, x, y, width, height):
        buffer = await self.reader.read(4)

        (srcx, srcy) = unpack("!HH", buffer)
        await self.copy_rectangle(srcx, srcy, x, y, width, height)
        await self._do_connection()

    async def _handle_decode_raw(self, x, y, width, height):
        buffer = await self.reader.read(width * height * self.bypp)

        await self.update_rectangle(x, y, width, height, buffer)
        await self._do_connection()

    async def _do_next_hextile_subrect(self, bg, color, x, y, width, height, tx, ty):
        if tx is not None:
            tx += 16
            if tx >= x + width:
                tx = x
                ty += 16
        else:
            tx = x
            ty = y

        if ty >= y + height:
            await self._do_connection()
        else:
            await self._handle_decode_hextile(bg, color, x, y, width, height, tx, ty)

    async def _handle_decode_hextile(self, bg, color, x, y, width, height, tx, ty):
        buffer = await self.reader.read(1)
        (subencoding,) = unpack("!B", buffer)

        tw = th = 16
        if x + width - tx < 16:
            tw = x + width - tx
        if y + height - ty < 16:
            th = y + height - ty

        if subencoding & 1:
            await self._handle_decode_hextile_raw(bg, color, x, y, width, height, tx, ty, tw, th)
        else:
            numbytes = 0
            if subencoding & 2:
                numbytes += self.bypp
            if subencoding & 4:
                numbytes += self.bypp
            if subencoding & 8:
                numbytes += 1
            if numbytes:
                await self._handle_decode_hextile_subrect(subencoding, bg, color, x, y, width, height, tx, ty, tw, th, numbytes)
            else:
                await self.fill_rectangle(tx, ty, tw, th, bg)
                await self._do_next_hextile_subrect(bg, color, x, y, width, height, tx, ty)

    async def _handle_decode_hextile_raw(self, bg, color, x, y, width, height, tx, ty, tw, th):
        buffer = self.reader.read(tw * th * self.bypp)
        await self.update_rectangle(tx, ty, tw, th, buffer)
        await self._do_next_hextile_subrect(bg, color, x, y, width, height, tx, ty)

    async def _handle_decode_hextile_subrect(self, subencoding, bg, color, x, y, width, height, tx, ty, tw, th, numbytes):
        buffer = self.reader.read(numbytes)

        subrects = 0
        pos = 0
        if subencoding & 2:
            bg = buffer[:self.bypp]
            pos += self.bypp
        await self.fill_rectangle(tx, ty, tw, th, bg)
        if subencoding & 4:
            color = buffer[pos:pos + self.bypp]
            pos += self.bypp
        if subencoding & 8:
            subrects = ord(buffer[pos])

        if subrects:
            if subencoding & 16:
                await self._handle_decode_hextile_subrects_coloured(bg, color, subrects, x, y, width, height, tx, ty, tw, th)
            else:
                await self._handle_decode_hextile_subrects_fg(bg, color, subrects, x, y, width, height, tx, ty, tw, th)
        else:
            await self._do_next_hextile_subrect(bg, color, x, y, width, height, tx, ty)

    async def _handle_decode_hextile_subrects_coloured(self, bg, color, subrects, x, y, width, height, tx, ty, tw, th):
        buffer = self.reader.read((self.bypp + 2) * subrects)
        sz = self.bypp + 2
        pos = 0
        end = len(buffer)
        while pos < end:
            pos2 = pos + self.bypp
            color = buffer[pos:pos2]
            xy = ord(buffer[pos2])
            wh = ord(buffer[pos2 + 1])
            sx = xy >> 4
            sy = xy & 0xf
            sw = (wh >> 4) + 1
            sh = (wh & 0xf) + 1
            await self.fill_rectangle(tx + sx, ty + sy, sw, sh, color)
            pos += sz
        await self._do_next_hextile_subrect(bg, color, x, y, width, height, tx, ty)

    async def _handle_decode_hextile_subrects_fg(self, bg, color, subrects, x, y, width, height, tx, ty, tw, th):
        buffer = self.reader.read(2 * subrects)
        pos = 0
        end = len(buffer)
        while pos < end:
            xy = ord(buffer[pos])
            wh = ord(buffer[pos + 1])
            sx = xy >> 4
            sy = xy & 0xf
            sw = (wh >> 4) + 1
            sh = (wh & 0xf) + 1
            await self.fill_rectangle(tx + sx, ty + sy, sw, sh, color)
            pos += 2
        await self._do_next_hextile_subrect(bg, color, x, y, width, height, tx, ty)

    async def _handle_decode_corre(self, x, y, width, height):
        buffer = await self.reader.read(4 + self.bypp)

        (subrects,) = unpack("!I", buffer[:4])
        color = buffer[4:]
        await self.fill_rectangle(x, y, width, height, color)
        if subrects:
            await self._handle_decode_corre_rectangles(x, y, subrects)
        else:
            await self._do_connection()

    async def _handle_decode_corre_rectangles(self, topx, topy, subrects):
        buffer = await self.reader.read((8 + self.bypp) * subrects)

        pos = 0
        end = len(buffer)
        sz = self.bypp + 4
        fmt = "!%dsBBBB" % self.bypp
        while pos < sz:
            (color, x, y, width, height) = unpack(fmt, buffer[pos:pos + sz])
            await self.fill_rectangle(topx + x, topy + y, width, height, color)
            pos += sz
        await self._do_connection()

    async def _handle_decode_rre(self, x, y, width, height):
        buffer = await self.reader.read(4 + self.bypp)
        (subrects,) = unpack("!I", buffer[:4])
        color = buffer[4:]
        await self.fill_rectangle(x, y, width, height, color)
        if subrects:
            await self._handle_rre_sub_rectangles(x, y, subrects)
        else:
            await self._do_connection()

    async def _handle_rre_sub_rectangles(self, topx, topy, subrects):
        buffer = await self.reader.read((8 + self.bypp) * subrects)

        pos = 0
        end = len(buffer)
        sz = self.bypp + 8
        fmt = "!%dsHHHH" % self.bypp
        while pos < end:
            (color, x, y, width, height) = unpack(fmt, buffer[pos:pos + sz])
            await self.fill_rectangle(topx + x, topy + y, width, height, color)
            pos += sz
        await self._do_connection()

    async def _handle_decode_zrle(self, x, y, width, height):
        buffer = await self.reader.read(4)

        (compressed_bytes,) = unpack("!L", buffer)
        await self._handle_decode_zrl_edata(x, y, width, height, compressed_bytes)

    async def _handle_decode_zrl_edata(self, x, y, width, height, num):
        buffer = await self.reader.read(num)

        tx = x
        ty = y

        data = self._zlib_stream.decompress(buffer)
        it = iter(data)

        def cpixel(i):
            yield next(i)
            yield next(i)
            yield next(i)
            yield 0xff

        while True:
            try:
                subencoding = ord(next(it))
            except StopIteration:
                break

            tw = th = 64
            if x + width - tx < 64:
                tw = x + width - tx
            if y + height - ty < 64:
                th = y + height - ty

            pixels_in_tile = tw * th

            num_pixels = 0
            pixel_data = bytearray()
            palette_size = subencoding & 127
            if subencoding & 0x80:
                def do_rle(pixel):
                    run_length_next = ord(next(it))
                    run_length = run_length_next
                    while run_length_next == 255:
                        run_length_next = ord(next(it))
                        run_length += run_length_next
                    pixel_data.extend(pixel * (run_length + 1))
                    return run_length + 1

                if palette_size == 0:
                    while num_pixels < pixels_in_tile:
                        color = bytearray(cpixel(it))
                        num_pixels += do_rle(color)
                    if num_pixels != pixels_in_tile:
                        raise ValueError("too many pixels")
                else:
                    palette = [bytearray(cpixel(it)) for p in range(palette_size)]

                    while num_pixels < pixels_in_tile:
                        palette_index = ord(next(it))
                        if palette_index & 0x80:
                            palette_index &= 0x7F
                            num_pixels += do_rle(palette[palette_index])
                        else:
                            pixel_data.extend(palette[palette_index])
                            num_pixels += 1
                    if num_pixels != pixels_in_tile:
                        raise ValueError("too many pixels")

                await self.update_rectangle(tx, ty, tw, th, bytes(pixel_data))
            else:
                if palette_size == 0:
                    pixel_data = b''.join(bytes(cpixel(it)) for _ in range(pixels_in_tile))
                    await self.update_rectangle(tx, ty, tw, th, bytes(pixel_data))
                elif palette_size == 1:
                    color = bytearray(cpixel(it))
                    await self.fill_rectangle(tx, ty, tw, th, bytes(color))
                else:
                    if palette_size > 16:
                        raise ValueError("Palette of size {0} is not allowed".format(palette_size))

                    palette = [bytearray(cpixel(it)) for _ in range(palette_size)]
                    if palette_size == 2:
                        next_index = _zrle_next_bit(it, pixels_in_tile)
                    elif palette_size == 3 or palette_size == 4:
                        next_index = _zrle_next_dibit(it, pixels_in_tile)
                    else:
                        next_index = _zrle_next_nibble(it, pixels_in_tile)

                    for palette_index in next_index:
                        pixel_data.extend(palette[palette_index])
                    await self.update_rectangle(tx, ty, tw, th, bytes(pixel_data))

            tx = tx + 64
            if tx >= x + width:
                tx = x
                ty = ty + 64

        await self._do_connection()

    async def _handle_decode_psuedo_cursor(self, length, x, y, width, height):
        buffer = await self.reader.read(length)

        split = width * height * self.bypp
        image = buffer[:split]
        mask = buffer[split:]
        await self.update_cursor(x, y, width, height, image, mask)
        await self._do_connection()

    async def _handle_decode_desktop_size(self, width, height):
        await self.update_desktop_size(width, height)
        await self._do_connection()

    async def _handle_framebuffer_update(self):
        buffer = await self.reader.read(3)

        self.rectangles, = unpack("!xH", buffer)
        self.rectanglePos = []
        await self.begin_update()
        await self._do_connection()

    async def _handle_server_cut_text(self):
        buffer = await self.reader.read(7)
        (length,) = unpack("!xxxI", buffer)
        await self._handle_server_cut_text_value(length)

    async def _handle_server_cut_text_value(self, length):
        buffer = await self.reader.read(length)

        await self.copy_text(buffer)

    async def vnc_request_password(self):
        if self.password is None:
            print("need a password")
            await self.writer.close()
            return
        await self.send_password(self.password)

    async def vnc_connection_made(self):
        ...

    async def vnc_auth_failed(self, reason):
        print("Cannot connect %s" % reason)

    async def send_password(self, password):
        pw = (password + '\0' * 8)[:8]
        response = RFBDes(pw).encrypt(self._challenge)
        self.writer.write(response)

    async def begin_update(self):
        ...

    async def commit_update(self, positions):
        ...

    async def copy_rectangle(self, srcx, srcy, x, y, width, height):
        ...

    async def update_rectangle(self, x, y, width, height, buffer):
        ...

    async def fill_rectangle(self, x, y, width, height, color):
        await self.update_rectangle(x, y, width, height, color * width * height)

    async def update_cursor(self, x, y, width, height, image, mask):
        ...

    async def update_desktop_size(self, width, height):
        ...

    async def copy_text(self, text):
        ...

    async def bell(self):
        print('!!BELL!!')

    async def set_pixel_format(self, bpp=32, depth=24, bigendian=0, truecolor=1, redmax=255, greenmax=255, bluemax=255, redshift=0, greenshift=8, blueshift=16):
        pixformat = pack("!BBBBHHHBBBxxx", bpp, depth, bigendian, truecolor, redmax, greenmax, bluemax, redshift, greenshift, blueshift)
        self.writer.write(pack("!Bxxx16s", 0, pixformat))
        self.bpp, self.depth, self.bigendian, self.truecolor = bpp, depth, bigendian, truecolor
        self.redmax, self.greenmax, self.bluemax = redmax, greenmax, bluemax
        self.redshift, self.greenshift, self.blueshift = redshift, greenshift, blueshift
        self.bypp = self.bpp // 8

    async def set_encodings(self, list_of_encodings):
        self.writer.write(pack("!BxH", 2, len(list_of_encodings)))
        for encoding in list_of_encodings:
            self.writer.write(pack("!i", encoding))

    async def framebuffer_update_request(self, x=0, y=0, width=None, height=None, incremental=0):
        if width is None:
            width = self.width - x
        if height is None:
            height = self.height - y
        self.writer.write(pack("!BBHHHH", 3, incremental, x, y, width, height))

    async def key_event(self, key, down=1):
        self.writer.write(pack("!BBxxI", 4, down, key))

    async def pointer_event(self, x, y, buttonmask=0):
        self.writer.write(pack("!BBHH", 5, buttonmask, x, y))

    async def client_cut_text(self, message):
        self.writer.write(pack("!BxxxI", 6, len(message)) + message)


class RFBDes(DES):
    def set_key(self, key):
        newkey = []
        for ki in range(len(key)):
            bsrc = ord(key[ki])
            btgt = 0
            for i in range(8):
                if bsrc & (1 << i):
                    btgt = btgt | (1 << 7 - i)
            newkey.append(chr(btgt))
        super(RFBDes, self).set_key(newkey)
