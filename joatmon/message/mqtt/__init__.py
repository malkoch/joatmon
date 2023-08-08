from joatmon.message.errors import HZMQTTException
from joatmon.message.mqtt.packet import (
    CONNACK,
    ConnackPacket,
    CONNECT,
    ConnectPacket,
    DISCONNECT,
    DisconnectPacket,
    MQTTFixedHeader,
    PINGREQ,
    PingReqPacket,
    PINGRESP,
    PingRespPacket,
    PUBACK,
    PubackPacket,
    PUBCOMP,
    PubcompPacket,
    PUBLISH,
    PublishPacket,
    PUBREC,
    PubrecPacket,
    PUBREL,
    PubrelPacket,
    SUBACK,
    SubackPacket,
    SUBSCRIBE,
    SubscribePacket,
    UNSUBACK,
    UnsubackPacket,
    UNSUBSCRIBE,
    UnsubscribePacket
)

packet_dict = {
    CONNECT: ConnectPacket,
    CONNACK: ConnackPacket,
    PUBLISH: PublishPacket,
    PUBACK: PubackPacket,
    PUBREC: PubrecPacket,
    PUBREL: PubrelPacket,
    PUBCOMP: PubcompPacket,
    SUBSCRIBE: SubscribePacket,
    SUBACK: SubackPacket,
    UNSUBSCRIBE: UnsubscribePacket,
    UNSUBACK: UnsubackPacket,
    PINGREQ: PingReqPacket,
    PINGRESP: PingRespPacket,
    DISCONNECT: DisconnectPacket
}


def packet_class(fixed_header: MQTTFixedHeader):
    try:
        cls = packet_dict[fixed_header.packet_type]
        return cls
    except KeyError:
        raise HZMQTTException("Unexpected packet Type '%s'" % fixed_header.packet_type)
