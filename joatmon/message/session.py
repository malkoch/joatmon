import asyncio
from asyncio import Queue
from collections import OrderedDict

from transitions import Machine

from joatmon.message.errors import HZMQTTException
from joatmon.message.mqtt.packet import PublishPacket

OUTGOING = 0
INCOMING = 1


class ApplicationMessage:
    __slots__ = ('packet_id', 'topic', 'qos', 'data', 'retain', 'publish_packet', 'puback_packet', 'pubrec_packet', 'pubrel_packet', 'pubcomp_packet')

    def __init__(self, packet_id, topic, qos, data, retain):
        self.packet_id = packet_id
        self.topic = topic
        self.qos = qos
        self.data = data
        self.retain = retain
        self.publish_packet = None
        self.puback_packet = None
        self.pubrec_packet = None
        self.pubrel_packet = None
        self.pubcomp_packet = None

    def build_publish_packet(self, dup=False):
        return PublishPacket.build(self.topic, self.data, self.packet_id, dup, self.qos, self.retain)

    def __eq__(self, other):
        return self.packet_id == other.packet_id


class IncomingApplicationMessage(ApplicationMessage):
    __slots__ = ('direction',)

    def __init__(self, packet_id, topic, qos, data, retain):
        super().__init__(packet_id, topic, qos, data, retain)
        self.direction = INCOMING


class OutgoingApplicationMessage(ApplicationMessage):
    __slots__ = ('direction',)

    def __init__(self, packet_id, topic, qos, data, retain):
        super().__init__(packet_id, topic, qos, data, retain)
        self.direction = OUTGOING


class Session:
    states = ['new', 'connected', 'disconnected']

    def __init__(self, loop=None):
        self._init_states()
        self.remote_address = None
        self.remote_port = None
        self.client_id = None
        self.clean_session = None
        self.will_flag = False
        self.will_message = None
        self.will_qos = None
        self.will_retain = None
        self.will_topic = None
        self.keep_alive = 0
        self.publish_retry_delay = 0
        self.broker_uri = None
        self.username = None
        self.password = None
        self.cafile = None
        self.capath = None
        self.cadata = None
        self._packet_id = 0
        self.parent = 0
        if loop is not None:
            self._loop = loop
        else:
            self._loop = asyncio.get_event_loop()

        self.inflight_out = OrderedDict()
        self.inflight_in = OrderedDict()
        self.retained_messages = Queue(loop=self._loop)
        self.delivered_message_queue = Queue(loop=self._loop)

    def _init_states(self):
        self.transitions = Machine(states=Session.states, initial='new')
        self.transitions.add_transition(trigger='connect', source='new', dest='connected')
        self.transitions.add_transition(trigger='connect', source='disconnected', dest='connected')
        self.transitions.add_transition(trigger='disconnect', source='connected', dest='disconnected')
        self.transitions.add_transition(trigger='disconnect', source='new', dest='disconnected')
        self.transitions.add_transition(trigger='disconnect', source='disconnected', dest='disconnected')

    @property
    def next_packet_id(self):
        self._packet_id += 1
        if self._packet_id > 65535:
            self._packet_id = 1
        while self._packet_id in self.inflight_in or self._packet_id in self.inflight_out:
            self._packet_id += 1
            if self._packet_id > 65535:
                raise HZMQTTException("More than 65525 messages pending. No free packet ID")

        return self._packet_id

    @property
    def inflight_in_count(self):
        return len(self.inflight_in)

    @property
    def inflight_out_count(self):
        return len(self.inflight_out)

    @property
    def retained_messages_count(self):
        return self.retained_messages.qsize()

    def __repr__(self):
        return type(self).__name__ + '(clientId={0}, state={1})'.format(self.client_id, self.transitions.state)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['retained_messages']
        del state['delivered_message_queue']
        return state

    def __setstate(self, state):
        self.__dict__.update(state)
        self.retained_messages = Queue()
        self.delivered_message_queue = Queue()

    def __eq__(self, other):
        return self.client_id == other.client_id
