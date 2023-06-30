import io
import logging
from asyncio import (
    StreamReader,
    StreamWriter
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
        if self.is_closed:
            return
        self._writer.write(data)

    async def drain(self):
        if self.is_closed:
            return
        await self._writer.drain()

    def get_peer_info(self):
        extra_info = self._writer.get_extra_info('peername')
        return extra_info[0], extra_info[1]

    async def close(self):
        if self.is_closed:
            return

        self.is_closed = True
        await self._writer.drain()
        if self._writer.can_write_eof():
            self._writer.write_eof()
        self._writer.close()

        try:
            await self._writer.wait_closed()
        except AttributeError:
            pass
