import asyncio
from datetime import datetime
from struct import unpack

from ..adapters import (
    ReaderAdapter,
    WriterAdapter
)
from ..codecs import (
    bytes_to_hex_str,
    bytes_to_int,
    decode_data_with_length,
    decode_packet_id,
    decode_string,
    encode_data_with_length,
    encode_string,
    int_to_bytes,
    read_or_raise
)
from ..errors import (
    CodecException,
    HZMQTTException,
    MQTTException,
    NoDataException
)
from ..utils import gen_client_id

RESERVED_0 = 0x00
CONNECT = 0x01
CONNACK = 0x02
PUBLISH = 0x03
PUBACK = 0x04
PUBREC = 0x05
PUBREL = 0x06
PUBCOMP = 0x07
SUBSCRIBE = 0x08
SUBACK = 0x09
UNSUBSCRIBE = 0x0a
UNSUBACK = 0x0b
PINGREQ = 0x0c
PINGRESP = 0x0d
DISCONNECT = 0x0e
RESERVED_15 = 0x0f

CONNECTION_ACCEPTED = 0x00
UNACCEPTABLE_PROTOCOL_VERSION = 0x01
IDENTIFIER_REJECTED = 0x02
SERVER_UNAVAILABLE = 0x03
BAD_USERNAME_PASSWORD = 0x04
NOT_AUTHORIZED = 0x05


class MQTTFixedHeader:
    __slots__ = ('packet_type', 'remaining_length', 'flags')

    def __init__(self, packet_type, flags=0, length=0):
        self.packet_type = packet_type
        self.remaining_length = length
        self.flags = flags

    def to_bytes(self):
        def encode_remaining_length(length: int):
            encoded = bytearray()
            while True:
                length_byte = length % 0x80
                length //= 0x80
                if length > 0:
                    length_byte |= 0x80
                encoded.append(length_byte)
                if length <= 0:
                    break
            return encoded

        out = bytearray()
        packet_type = 0
        try:
            packet_type = (self.packet_type << 4) | self.flags
            out.append(packet_type)
        except OverflowError:
            raise CodecException('packet_type encoding exceed 1 byte length: value=%d', packet_type)

        encoded_length = encode_remaining_length(self.remaining_length)
        out.extend(encoded_length)

        return out

    async def to_stream(self, writer: WriterAdapter):
        writer.write(self.to_bytes())

    @property
    def bytes_length(self):
        return len(self.to_bytes())

    @classmethod
    async def from_stream(cls, reader: ReaderAdapter):
        async def decode_remaining_length():
            multiplier = 1
            value = 0
            buffer = bytearray()
            while True:
                encoded_byte = await reader.read(1)
                int_byte = unpack('!B', encoded_byte)
                buffer.append(int_byte[0])
                value += (int_byte[0] & 0x7f) * multiplier
                if (int_byte[0] & 0x80) == 0:
                    break
                else:
                    multiplier *= 128
                    if multiplier > 128 * 128 * 128:
                        raise MQTTException("Invalid remaining length bytes:%s, packet_type=%d" % (bytes_to_hex_str(buffer), msg_type))
            return value

        try:
            byte1 = await read_or_raise(reader, 1)
            int1 = unpack('!B', byte1)
            msg_type = (int1[0] & 0xf0) >> 4
            flags = int1[0] & 0x0f
            remain_length = await decode_remaining_length()

            return cls(msg_type, flags, remain_length)
        except NoDataException:
            return None

    def __repr__(self):
        return type(self).__name__ + '(length={0}, flags={1})'. \
            format(self.remaining_length, hex(self.flags))


class MQTTVariableHeader:
    def __init__(self):
        pass

    async def to_stream(self, writer: asyncio.StreamWriter):
        writer.write(self.to_bytes())
        await writer.drain()

    def to_bytes(self) -> bytes:
        ...

    @property
    def bytes_length(self):
        return len(self.to_bytes())

    @classmethod
    async def from_stream(cls, reader: asyncio.StreamReader, fixed_header: MQTTFixedHeader):
        pass


class PacketIdVariableHeader(MQTTVariableHeader):
    __slots__ = ('packet_id',)

    def __init__(self, packet_id):
        super().__init__()
        self.packet_id = packet_id

    def to_bytes(self):
        out = b''
        out += int_to_bytes(self.packet_id, 2)
        return out

    @classmethod
    async def from_stream(cls, reader: ReaderAdapter, fixed_header: MQTTFixedHeader):
        packet_id = await decode_packet_id(reader)
        return cls(packet_id)

    def __repr__(self):
        return type(self).__name__ + '(packet_id={0})'.format(self.packet_id)


class MQTTPayload:
    def __init__(self):
        pass

    async def to_stream(self, writer: asyncio.StreamWriter):
        writer.write(self.to_bytes())
        await writer.drain()

    def to_bytes(self, fixed_header: MQTTFixedHeader, variable_header: MQTTVariableHeader):
        pass

    @classmethod
    async def from_stream(cls, reader: asyncio.StreamReader, fixed_header: MQTTFixedHeader, variable_header: MQTTVariableHeader):
        pass


class MQTTPacket:
    __slots__ = ('fixed_header', 'variable_header', 'payload', 'protocol_ts')

    FIXED_HEADER = MQTTFixedHeader
    VARIABLE_HEADER = None
    PAYLOAD = None

    def __init__(self, fixed: MQTTFixedHeader, variable_header: MQTTVariableHeader = None, payload: MQTTPayload = None):
        self.fixed_header = fixed
        self.variable_header = variable_header
        self.payload = payload
        self.protocol_ts = None

    async def to_stream(self, writer: asyncio.StreamWriter):
        writer.write(self.to_bytes())
        await writer.drain()
        self.protocol_ts = datetime.now()

    def to_bytes(self) -> bytes:
        if self.variable_header:
            variable_header_bytes = self.variable_header.to_bytes()
        else:
            variable_header_bytes = b''
        if self.payload:
            payload_bytes = self.payload.to_bytes(self.fixed_header, self.variable_header)
        else:
            payload_bytes = b''

        self.fixed_header.remaining_length = len(variable_header_bytes) + len(payload_bytes)
        fixed_header_bytes = self.fixed_header.to_bytes()

        return fixed_header_bytes + variable_header_bytes + payload_bytes

    @classmethod
    async def from_stream(cls, reader: ReaderAdapter, fixed_header=None, variable_header=None):
        if fixed_header is None:
            fixed_header = await cls.FIXED_HEADER.from_stream(reader)
        if cls.VARIABLE_HEADER:
            if variable_header is None:
                variable_header = await cls.VARIABLE_HEADER.from_stream(reader, fixed_header)
        else:
            variable_header = None
        if cls.PAYLOAD:
            payload = await cls.PAYLOAD.from_stream(reader, fixed_header, variable_header)
        else:
            payload = None

        if fixed_header and not variable_header and not payload:
            instance = cls(fixed_header)
        elif fixed_header and not payload:
            instance = cls(fixed_header, variable_header)
        else:
            instance = cls(fixed_header, variable_header, payload)
        instance.protocol_ts = datetime.now()
        return instance

    @property
    def bytes_length(self):
        return len(self.to_bytes())

    def __repr__(self):
        return type(self).__name__ + '(ts={0!s}, fixed={1!r}, variable={2!r}, payload={3!r})'.format(self.protocol_ts, self.fixed_header, self.variable_header, self.payload)


class ConnackVariableHeader(MQTTVariableHeader):
    __slots__ = ('session_parent', 'return_code')

    def __init__(self, session_parent=None, return_code=None):
        super().__init__()
        self.session_parent = session_parent
        self.return_code = return_code

    @classmethod
    async def from_stream(cls, reader: ReaderAdapter, fixed_header: MQTTFixedHeader):
        data = await read_or_raise(reader, 2)
        session_parent = data[0] & 0x01
        return_code = bytes_to_int(data[1])
        return cls(session_parent, return_code)

    def to_bytes(self):
        out = bytearray(2)
        if self.session_parent:
            out[0] = 1
        else:
            out[0] = 0
        out[1] = self.return_code
        return out

    def __repr__(self):
        return type(self).__name__ + '(session_parent={0}, return_code={1})'.format(hex(self.session_parent), hex(self.return_code))


class ConnackPacket(MQTTPacket):
    VARIABLE_HEADER = ConnackVariableHeader
    PAYLOAD = None

    @property
    def return_code(self):
        return self.variable_header.return_code

    @return_code.setter
    def return_code(self, return_code):
        self.variable_header.return_code = return_code

    @property
    def session_parent(self):
        return self.variable_header.session_parent

    @session_parent.setter
    def session_parent(self, session_parent):
        self.variable_header.session_parent = session_parent

    def __init__(self, fixed: MQTTFixedHeader = None, variable_header: ConnackVariableHeader = None, payload=None):
        if fixed is None:
            header = MQTTFixedHeader(CONNACK, 0x00)
        else:
            if fixed.packet_type is not CONNACK:
                raise HZMQTTException("Invalid fixed packet type %s for ConnackPacket init" % fixed.packet_type)
            header = fixed
        super().__init__(header)
        self.variable_header = variable_header
        self.payload = None

    @classmethod
    def build(cls, session_parent=None, return_code=None):
        v_header = ConnackVariableHeader(session_parent, return_code)
        packet = ConnackPacket(variable_header=v_header)
        return packet


class ConnectVariableHeader(MQTTVariableHeader):
    __slots__ = ('proto_name', 'proto_level', 'flags', 'keep_alive')

    USERNAME_FLAG = 0x80
    PASSWORD_FLAG = 0x40
    WILL_RETAIN_FLAG = 0x20
    WILL_FLAG = 0x04
    WILL_QOS_MASK = 0x18
    CLEAN_SESSION_FLAG = 0x02
    RESERVED_FLAG = 0x01

    def __init__(self, connect_flags=0x00, keep_alive=0, proto_name='MQTT', proto_level=0x04):
        super().__init__()
        self.proto_name = proto_name
        self.proto_level = proto_level
        self.flags = connect_flags
        self.keep_alive = keep_alive

    def __repr__(self):
        return "ConnectVariableHeader(proto_name={0}, proto_level={1}, flags={2}, keepalive={3})".format(self.proto_name, self.proto_level, hex(self.flags), self.keep_alive)

    def _set_flag(self, val, mask):
        if val:
            self.flags |= mask
        else:
            self.flags &= ~mask

    def _get_flag(self, mask):
        if self.flags & mask:
            return True
        else:
            return False

    @property
    def username_flag(self) -> bool:
        return self._get_flag(self.USERNAME_FLAG)

    @username_flag.setter
    def username_flag(self, val: bool):
        self._set_flag(val, self.USERNAME_FLAG)

    @property
    def password_flag(self) -> bool:
        return self._get_flag(self.PASSWORD_FLAG)

    @password_flag.setter
    def password_flag(self, val: bool):
        self._set_flag(val, self.PASSWORD_FLAG)

    @property
    def will_retain_flag(self) -> bool:
        return self._get_flag(self.WILL_RETAIN_FLAG)

    @will_retain_flag.setter
    def will_retain_flag(self, val: bool):
        self._set_flag(val, self.WILL_RETAIN_FLAG)

    @property
    def will_flag(self) -> bool:
        return self._get_flag(self.WILL_FLAG)

    @will_flag.setter
    def will_flag(self, val: bool):
        self._set_flag(val, self.WILL_FLAG)

    @property
    def clean_session_flag(self) -> bool:
        return self._get_flag(self.CLEAN_SESSION_FLAG)

    @clean_session_flag.setter
    def clean_session_flag(self, val: bool):
        self._set_flag(val, self.CLEAN_SESSION_FLAG)

    @property
    def reserved_flag(self) -> bool:
        return self._get_flag(self.RESERVED_FLAG)

    @property
    def will_qos(self):
        return (self.flags & 0x18) >> 3

    @will_qos.setter
    def will_qos(self, val: int):
        self.flags &= 0xe7  # Reset QOS flags
        self.flags |= (val << 3)

    @classmethod
    async def from_stream(cls, reader: ReaderAdapter, fixed_header: MQTTFixedHeader):
        protocol_name = await decode_string(reader)

        protocol_level_byte = await read_or_raise(reader, 1)
        protocol_level = bytes_to_int(protocol_level_byte)

        flags_byte = await read_or_raise(reader, 1)
        flags = bytes_to_int(flags_byte)

        keep_alive_byte = await read_or_raise(reader, 2)
        keep_alive = bytes_to_int(keep_alive_byte)

        return cls(flags, keep_alive, protocol_name, protocol_level)

    def to_bytes(self):
        out = bytearray()
        out.extend(encode_string(self.proto_name))
        out.append(self.proto_level)
        out.append(self.flags)
        out.extend(int_to_bytes(self.keep_alive, 2))
        return out


class ConnectPayload(MQTTPayload):
    __slots__ = ('client_id', 'will_topic', 'will_message', 'username', 'password', 'client_id_is_random')

    def __init__(self, client_id=None, will_topic=None, will_message=None, username=None, password=None):
        super().__init__()
        self.client_id_is_random = False
        self.client_id = client_id
        self.will_topic = will_topic
        self.will_message = will_message
        self.username = username
        self.password = password

    def __repr__(self):
        return "ConnectVariableHeader(client_id={0}, will_topic={1}, will_message={2}, username={3}, password={4})".format(
            self.client_id, self.will_topic, self.will_message, self.username, self.password
            )

    @classmethod
    async def from_stream(cls, reader: ReaderAdapter, fixed_header: MQTTFixedHeader, variable_header: ConnectVariableHeader):
        payload = cls()
        try:
            payload.client_id = await decode_string(reader)
        except NoDataException:
            payload.client_id = None

        if payload.client_id is None or payload.client_id == "":
            payload.client_id = gen_client_id()
            payload.client_id_is_random = True

        if variable_header.will_flag:
            try:
                payload.will_topic = await decode_string(reader)
                payload.will_message = await decode_data_with_length(reader)
            except NoDataException:
                payload.will_topic = None
                payload.will_message = None

        if variable_header.username_flag:
            try:
                payload.username = await decode_string(reader)
            except NoDataException:
                payload.username = None

        if variable_header.password_flag:
            try:
                payload.password = await decode_string(reader)
            except NoDataException:
                payload.password = None

        return payload

    def to_bytes(self, fixed_header: MQTTFixedHeader, variable_header: ConnectVariableHeader):
        out = bytearray()
        out.extend(encode_string(self.client_id))
        if variable_header.will_flag:
            out.extend(encode_string(self.will_topic))
            out.extend(encode_data_with_length(self.will_message))
        if variable_header.username_flag:
            out.extend(encode_string(self.username))
        if variable_header.password_flag:
            out.extend(encode_string(self.password))
        return out


class ConnectPacket(MQTTPacket):
    VARIABLE_HEADER = ConnectVariableHeader
    PAYLOAD = ConnectPayload

    @property
    def proto_name(self):
        return self.variable_header.proto_name

    @proto_name.setter
    def proto_name(self, name: str):
        self.variable_header.proto_name = name

    @property
    def proto_level(self):
        return self.variable_header.proto_level

    @proto_level.setter
    def proto_level(self, level):
        self.variable_header.proto_level = level

    @property
    def username_flag(self):
        return self.variable_header.username_flag

    @username_flag.setter
    def username_flag(self, flag):
        self.variable_header.username_flag = flag

    @property
    def password_flag(self):
        return self.variable_header.password_flag

    @password_flag.setter
    def password_flag(self, flag):
        self.variable_header.password_flag = flag

    @property
    def clean_session_flag(self):
        return self.variable_header.clean_session_flag

    @clean_session_flag.setter
    def clean_session_flag(self, flag):
        self.variable_header.clean_session_flag = flag

    @property
    def will_retain_flag(self):
        return self.variable_header.will_retain_flag

    @will_retain_flag.setter
    def will_retain_flag(self, flag):
        self.variable_header.will_retain_flag = flag

    @property
    def will_qos(self):
        return self.variable_header.will_qos

    @will_qos.setter
    def will_qos(self, flag):
        self.variable_header.will_qos = flag

    @property
    def will_flag(self):
        return self.variable_header.will_flag

    @will_flag.setter
    def will_flag(self, flag):
        self.variable_header.will_flag = flag

    @property
    def reserved_flag(self):
        return self.variable_header.reserved_flag

    @reserved_flag.setter
    def reserved_flag(self, flag):
        self.variable_header.reserved_flag = flag

    @property
    def client_id(self):
        return self.payload.client_id

    @client_id.setter
    def client_id(self, client_id):
        self.payload.client_id = client_id

    @property
    def client_id_is_random(self) -> bool:
        return self.payload.client_id_is_random

    @client_id_is_random.setter
    def client_id_is_random(self, client_id_is_random: bool):
        self.payload.client_id_is_random = client_id_is_random

    @property
    def will_topic(self):
        return self.payload.will_topic

    @will_topic.setter
    def will_topic(self, will_topic):
        self.payload.will_topic = will_topic

    @property
    def will_message(self):
        return self.payload.will_message

    @will_message.setter
    def will_message(self, will_message):
        self.payload.will_message = will_message

    @property
    def username(self):
        return self.payload.username

    @username.setter
    def username(self, username):
        self.payload.username = username

    @property
    def password(self):
        return self.payload.password

    @password.setter
    def password(self, password):
        self.payload.password = password

    @property
    def keep_alive(self):
        return self.variable_header.keep_alive

    @keep_alive.setter
    def keep_alive(self, keep_alive):
        self.variable_header.keep_alive = keep_alive

    def __init__(self, fixed: MQTTFixedHeader = None, vh: ConnectVariableHeader = None, payload: ConnectPayload = None):
        if fixed is None:
            header = MQTTFixedHeader(CONNECT, 0x00)
        else:
            if fixed.packet_type is not CONNECT:
                raise HZMQTTException("Invalid fixed packet type %s for ConnectPacket init" % fixed.packet_type)
            header = fixed
        super().__init__(header)
        self.variable_header = vh
        self.payload = payload


class DisconnectPacket(MQTTPacket):
    VARIABLE_HEADER = None
    PAYLOAD = None

    def __init__(self, fixed: MQTTFixedHeader = None):
        if fixed is None:
            header = MQTTFixedHeader(DISCONNECT, 0x00)
        else:
            if fixed.packet_type is not DISCONNECT:
                raise HZMQTTException("Invalid fixed packet type %s for DisconnectPacket init" % fixed.packet_type)
            header = fixed
        super().__init__(header)
        self.variable_header = None
        self.payload = None


class PingReqPacket(MQTTPacket):
    VARIABLE_HEADER = None
    PAYLOAD = None

    def __init__(self, fixed: MQTTFixedHeader = None):
        if fixed is None:
            header = MQTTFixedHeader(PINGREQ, 0x00)
        else:
            if fixed.packet_type is not PINGREQ:
                raise HZMQTTException("Invalid fixed packet type %s for PingReqPacket init" % fixed.packet_type)
            header = fixed
        super().__init__(header)
        self.variable_header = None
        self.payload = None


class PingRespPacket(MQTTPacket):
    VARIABLE_HEADER = None
    PAYLOAD = None

    def __init__(self, fixed: MQTTFixedHeader = None):
        if fixed is None:
            header = MQTTFixedHeader(PINGRESP, 0x00)
        else:
            if fixed.packet_type is not PINGRESP:
                raise HZMQTTException("Invalid fixed packet type %s for PingRespPacket init" % fixed.packet_type)
            header = fixed
        super().__init__(header)
        self.variable_header = None
        self.payload = None

    @classmethod
    def build(cls):
        return cls()


class PubackPacket(MQTTPacket):
    VARIABLE_HEADER = PacketIdVariableHeader
    PAYLOAD = None

    @property
    def packet_id(self):
        return self.variable_header.packet_id

    @packet_id.setter
    def packet_id(self, val: int):
        self.variable_header.packet_id = val

    def __init__(self, fixed: MQTTFixedHeader = None, variable_header: PacketIdVariableHeader = None):
        if fixed is None:
            header = MQTTFixedHeader(PUBACK, 0x00)
        else:
            if fixed.packet_type is not PUBACK:
                raise HZMQTTException("Invalid fixed packet type %s for PubackPacket init" % fixed.packet_type)
            header = fixed
        super().__init__(header)
        self.variable_header = variable_header
        self.payload = None

    @classmethod
    def build(cls, packet_id: int):
        v_header = PacketIdVariableHeader(packet_id)
        packet = PubackPacket(variable_header=v_header)
        return packet


class PubcompPacket(MQTTPacket):
    VARIABLE_HEADER = PacketIdVariableHeader
    PAYLOAD = None

    @property
    def packet_id(self):
        return self.variable_header.packet_id

    @packet_id.setter
    def packet_id(self, val: int):
        self.variable_header.packet_id = val

    def __init__(self, fixed: MQTTFixedHeader = None, variable_header: PacketIdVariableHeader = None):
        if fixed is None:
            header = MQTTFixedHeader(PUBCOMP, 0x00)
        else:
            if fixed.packet_type is not PUBCOMP:
                raise HZMQTTException("Invalid fixed packet type %s for PubcompPacket init" % fixed.packet_type)
            header = fixed
        super().__init__(header)
        self.variable_header = variable_header
        self.payload = None

    @classmethod
    def build(cls, packet_id: int):
        v_header = PacketIdVariableHeader(packet_id)
        packet = PubcompPacket(variable_header=v_header)
        return packet


class PublishVariableHeader(MQTTVariableHeader):
    __slots__ = ('topic_name', 'packet_id')

    def __init__(self, topic_name: str, packet_id: int = None):
        super().__init__()
        if '*' in topic_name:
            raise MQTTException("[MQTT-3.3.2-2] Topic name in the PUBLISH Packet MUST NOT contain wildcard characters.")
        self.topic_name = topic_name
        self.packet_id = packet_id

    def __repr__(self):
        return type(self).__name__ + '(topic={0}, packet_id={1})'.format(self.topic_name, self.packet_id)

    def to_bytes(self):
        out = bytearray()
        out.extend(encode_string(self.topic_name))
        if self.packet_id is not None:
            out.extend(int_to_bytes(self.packet_id, 2))
        return out

    @classmethod
    async def from_stream(cls, reader: asyncio.StreamReader, fixed_header: MQTTFixedHeader):
        topic_name = await decode_string(reader)
        has_qos = (fixed_header.flags >> 1) & 0x03
        if has_qos:
            packet_id = await decode_packet_id(reader)
        else:
            packet_id = None
        return cls(topic_name, packet_id)


class PublishPayload(MQTTPayload):
    __slots__ = ('data',)

    def __init__(self, data: bytes = None):
        super().__init__()
        self.data = data

    def to_bytes(self, fixed_header: MQTTFixedHeader, variable_header: MQTTVariableHeader):
        return self.data

    @classmethod
    async def from_stream(cls, reader: asyncio.StreamReader, fixed_header: MQTTFixedHeader, variable_header: MQTTVariableHeader):
        data = bytearray()
        data_length = fixed_header.remaining_length - variable_header.bytes_length
        length_read = 0
        while length_read < data_length:
            buffer = await reader.read(data_length - length_read)
            data.extend(buffer)
            length_read = len(data)
        return cls(data)

    def __repr__(self):
        return type(self).__name__ + '(data={0!r})'.format(repr(self.data))


class PublishPacket(MQTTPacket):
    VARIABLE_HEADER = PublishVariableHeader
    PAYLOAD = PublishPayload

    DUP_FLAG = 0x08
    RETAIN_FLAG = 0x01
    QOS_FLAG = 0x06

    def __init__(self, fixed: MQTTFixedHeader = None, variable_header: PublishVariableHeader = None, payload=None):
        if fixed is None:
            header = MQTTFixedHeader(PUBLISH, 0x00)
        else:
            if fixed.packet_type is not PUBLISH:
                raise HZMQTTException("Invalid fixed packet type %s for PublishPacket init" % fixed.packet_type)
            header = fixed

        super().__init__(header)
        self.variable_header = variable_header
        self.payload = payload

    def set_flags(self, dup_flag=False, qos=0, retain_flag=False):
        self.dup_flag = dup_flag
        self.retain_flag = retain_flag
        self.qos = qos

    def _set_header_flag(self, val, mask):
        if val:
            self.fixed_header.flags |= mask
        else:
            self.fixed_header.flags &= ~mask

    def _get_header_flag(self, mask):
        if self.fixed_header.flags & mask:
            return True
        else:
            return False

    @property
    def dup_flag(self) -> bool:
        return self._get_header_flag(self.DUP_FLAG)

    @dup_flag.setter
    def dup_flag(self, val: bool):
        self._set_header_flag(val, self.DUP_FLAG)

    @property
    def retain_flag(self) -> bool:
        return self._get_header_flag(self.RETAIN_FLAG)

    @retain_flag.setter
    def retain_flag(self, val: bool):
        self._set_header_flag(val, self.RETAIN_FLAG)

    @property
    def qos(self):
        return (self.fixed_header.flags & self.QOS_FLAG) >> 1

    @qos.setter
    def qos(self, val: int):
        self.fixed_header.flags &= 0xf9
        self.fixed_header.flags |= (val << 1)

    @property
    def packet_id(self):
        return self.variable_header.packet_id

    @packet_id.setter
    def packet_id(self, val: int):
        self.variable_header.packet_id = val

    @property
    def data(self):
        return self.payload.data

    @data.setter
    def data(self, data: bytes):
        self.payload.data = data

    @property
    def topic_name(self):
        return self.variable_header.topic_name

    @topic_name.setter
    def topic_name(self, name: str):
        self.variable_header.topic_name = name

    @classmethod
    def build(cls, topic_name: str, message: bytes, packet_id: int, dup_flag, qos, retain):
        v_header = PublishVariableHeader(topic_name, packet_id)
        payload = PublishPayload(message)
        packet = PublishPacket(variable_header=v_header, payload=payload)
        packet.dup_flag = dup_flag
        packet.retain_flag = retain
        packet.qos = qos
        return packet


class PubrecPacket(MQTTPacket):
    VARIABLE_HEADER = PacketIdVariableHeader
    PAYLOAD = None

    @property
    def packet_id(self):
        return self.variable_header.packet_id

    @packet_id.setter
    def packet_id(self, val: int):
        self.variable_header.packet_id = val

    def __init__(self, fixed: MQTTFixedHeader = None, variable_header: PacketIdVariableHeader = None):
        if fixed is None:
            header = MQTTFixedHeader(PUBREC, 0x00)
        else:
            if fixed.packet_type is not PUBREC:
                raise HZMQTTException("Invalid fixed packet type %s for PubrecPacket init" % fixed.packet_type)
            header = fixed
        super().__init__(header)
        self.variable_header = variable_header
        self.payload = None

    @classmethod
    def build(cls, packet_id: int):
        v_header = PacketIdVariableHeader(packet_id)
        packet = PubrecPacket(variable_header=v_header)
        return packet


class PubrelPacket(MQTTPacket):
    VARIABLE_HEADER = PacketIdVariableHeader
    PAYLOAD = None

    @property
    def packet_id(self):
        return self.variable_header.packet_id

    @packet_id.setter
    def packet_id(self, val: int):
        self.variable_header.packet_id = val

    def __init__(self, fixed: MQTTFixedHeader = None, variable_header: PacketIdVariableHeader = None):
        if fixed is None:
            header = MQTTFixedHeader(PUBREL, 0x02)
        else:
            if fixed.packet_type is not PUBREL:
                raise HZMQTTException("Invalid fixed packet type %s for PubrelPacket init" % fixed.packet_type)
            header = fixed
        super().__init__(header)
        self.variable_header = variable_header
        self.payload = None

    @classmethod
    def build(cls, packet_id):
        variable_header = PacketIdVariableHeader(packet_id)
        return PubrelPacket(variable_header=variable_header)


class SubackPayload(MQTTPayload):
    __slots__ = ('return_codes',)

    RETURN_CODE_00 = 0x00
    RETURN_CODE_01 = 0x01
    RETURN_CODE_02 = 0x02
    RETURN_CODE_80 = 0x80

    def __init__(self, return_codes=[]):
        super().__init__()
        self.return_codes = return_codes

    def __repr__(self):
        return type(self).__name__ + '(return_codes={0})'.format(repr(self.return_codes))

    def to_bytes(self, fixed_header: MQTTFixedHeader, variable_header: MQTTVariableHeader):
        out = b''
        for return_code in self.return_codes:
            out += int_to_bytes(return_code, 1)
        return out

    @classmethod
    async def from_stream(cls, reader: ReaderAdapter, fixed_header: MQTTFixedHeader, variable_header: MQTTVariableHeader):
        return_codes = []
        bytes_to_read = fixed_header.remaining_length - variable_header.bytes_length
        for i in range(0, bytes_to_read):
            try:
                return_code_byte = await read_or_raise(reader, 1)
                return_code = bytes_to_int(return_code_byte)
                return_codes.append(return_code)
            except NoDataException:
                break
        return cls(return_codes)


class SubackPacket(MQTTPacket):
    VARIABLE_HEADER = PacketIdVariableHeader
    PAYLOAD = SubackPayload

    def __init__(self, fixed: MQTTFixedHeader = None, variable_header: PacketIdVariableHeader = None, payload=None):
        if fixed is None:
            header = MQTTFixedHeader(SUBACK, 0x00)
        else:
            if fixed.packet_type is not SUBACK:
                raise HZMQTTException("Invalid fixed packet type %s for SubackPacket init" % fixed.packet_type)
            header = fixed

        super().__init__(header)
        self.variable_header = variable_header
        self.payload = payload

    @classmethod
    def build(cls, packet_id, return_codes):
        variable_header = cls.VARIABLE_HEADER(packet_id)
        payload = cls.PAYLOAD(return_codes)
        return cls(variable_header=variable_header, payload=payload)


class SubscribePayload(MQTTPayload):
    __slots__ = ('topics',)

    def __init__(self, topics=[]):
        super().__init__()
        self.topics = topics

    def to_bytes(self, fixed_header: MQTTFixedHeader, variable_header: MQTTVariableHeader):
        out = b''
        for topic in self.topics:
            out += encode_string(topic[0])
            out += int_to_bytes(topic[1], 1)
        return out

    @classmethod
    async def from_stream(cls, reader: asyncio.StreamReader, fixed_header: MQTTFixedHeader, variable_header: MQTTVariableHeader):
        topics = []
        payload_length = fixed_header.remaining_length - variable_header.bytes_length
        read_bytes = 0
        while read_bytes < payload_length:
            try:
                topic = await decode_string(reader)
                qos_byte = await read_or_raise(reader, 1)
                qos = bytes_to_int(qos_byte)
                topics.append((topic, qos))
                read_bytes += 2 + len(topic.encode('utf-8')) + 1
            except NoDataException as exc:
                break
        return cls(topics)

    def __repr__(self):
        return type(self).__name__ + '(topics={0!r})'.format(self.topics)


class SubscribePacket(MQTTPacket):
    VARIABLE_HEADER = PacketIdVariableHeader
    PAYLOAD = SubscribePayload

    def __init__(self, fixed: MQTTFixedHeader = None, variable_header: PacketIdVariableHeader = None, payload=None):
        if fixed is None:
            header = MQTTFixedHeader(SUBSCRIBE, 0x02)  # [MQTT-3.8.1-1]
        else:
            if fixed.packet_type is not SUBSCRIBE:
                raise HZMQTTException("Invalid fixed packet type %s for SubscribePacket init" % fixed.packet_type)
            header = fixed

        super().__init__(header)
        self.variable_header = variable_header
        self.payload = payload

    @classmethod
    def build(cls, topics, packet_id):
        v_header = PacketIdVariableHeader(packet_id)
        payload = SubscribePayload(topics)
        return SubscribePacket(variable_header=v_header, payload=payload)


class UnsubackPacket(MQTTPacket):
    VARIABLE_HEADER = PacketIdVariableHeader
    PAYLOAD = None

    def __init__(self, fixed: MQTTFixedHeader = None, variable_header: PacketIdVariableHeader = None, payload=None):
        if fixed is None:
            header = MQTTFixedHeader(UNSUBACK, 0x00)
        else:
            if fixed.packet_type is not UNSUBACK:
                raise HZMQTTException("Invalid fixed packet type %s for UnsubackPacket init" % fixed.packet_type)
            header = fixed

        super().__init__(header)
        self.variable_header = variable_header
        self.payload = payload

    @classmethod
    def build(cls, packet_id):
        variable_header = PacketIdVariableHeader(packet_id)
        return cls(variable_header=variable_header)


class UnubscribePayload(MQTTPayload):
    __slots__ = ('topics',)

    def __init__(self, topics=[]):
        super().__init__()
        self.topics = topics

    def to_bytes(self, fixed_header: MQTTFixedHeader, variable_header: MQTTVariableHeader):
        out = b''
        for topic in self.topics:
            out += encode_string(topic)
        return out

    @classmethod
    async def from_stream(
            cls, reader: asyncio.StreamReader, fixed_header: MQTTFixedHeader,
            variable_header: MQTTVariableHeader
            ):
        topics = []
        payload_length = fixed_header.remaining_length - variable_header.bytes_length
        read_bytes = 0
        while read_bytes < payload_length:
            try:
                topic = await decode_string(reader)
                topics.append(topic)
                read_bytes += 2 + len(topic.encode('utf-8'))
            except NoDataException:
                break
        return cls(topics)


class UnsubscribePacket(MQTTPacket):
    VARIABLE_HEADER = PacketIdVariableHeader
    PAYLOAD = UnubscribePayload

    def __init__(self, fixed: MQTTFixedHeader = None, variable_header: PacketIdVariableHeader = None, payload=None):
        if fixed is None:
            header = MQTTFixedHeader(UNSUBSCRIBE, 0x02)
        else:
            if fixed.packet_type is not UNSUBSCRIBE:
                raise HZMQTTException("Invalid fixed packet type %s for UnsubscribePacket init" % fixed.packet_type)
            header = fixed

        super().__init__(header)
        self.variable_header = variable_header
        self.payload = payload

    @classmethod
    def build(cls, topics, packet_id):
        v_header = PacketIdVariableHeader(packet_id)
        payload = UnubscribePayload(topics)
        return UnsubscribePacket(variable_header=v_header, payload=payload)
