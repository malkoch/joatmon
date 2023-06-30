import asyncio
import collections
import itertools
import logging
from asyncio import (
    futures,
    InvalidStateError,
    Queue
)

from ..adapters import (
    ReaderAdapter,
    WriterAdapter
)
from ..errors import (
    HZMQTTException,
    MQTTException,
    NoDataException
)
from ..mqtt import packet_class
from ..mqtt.constants import (
    QOS_0,
    QOS_1,
    QOS_2
)
from ..mqtt.packet import (
    BAD_USERNAME_PASSWORD,
    CONNACK,
    ConnackPacket,
    CONNECT,
    CONNECTION_ACCEPTED,
    ConnectPacket,
    ConnectPayload,
    ConnectVariableHeader,
    DISCONNECT,
    DisconnectPacket,
    IDENTIFIER_REJECTED,
    MQTTFixedHeader,
    NOT_AUTHORIZED,
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
    RESERVED_0,
    RESERVED_15,
    SUBACK,
    SubackPacket,
    SUBSCRIBE,
    SubscribePacket,
    UNACCEPTABLE_PROTOCOL_VERSION,
    UNSUBACK,
    UnsubackPacket,
    UNSUBSCRIBE,
    UnsubscribePacket
)
from ..session import (
    INCOMING,
    IncomingApplicationMessage,
    OUTGOING,
    OutgoingApplicationMessage,
    Session
)
from ..utils import format_client_message


class ProtocolHandlerException(BaseException):
    pass


class ProtocolHandler:
    def __init__(self, session: Session = None, loop=None):
        self.logger = logging.getLogger(__name__)
        if session:
            self._init_session(session)
        else:
            self.session = None
        self.reader = None
        self.writer = None

        if loop is None:
            self._loop = asyncio.get_event_loop()
        else:
            self._loop = loop
        self._reader_task = None
        self._keepalive_task = None
        self._reader_ready = None
        self._reader_stopped = asyncio.Event(loop=self._loop)

        self._puback_waiters = dict()
        self._pubrec_waiters = dict()
        self._pubrel_waiters = dict()
        self._pubcomp_waiters = dict()

        self._write_lock = asyncio.Lock(loop=self._loop)

    def _init_session(self, session: Session):
        assert session
        log = logging.getLogger(__name__)
        self.session = session
        self.logger = logging.LoggerAdapter(log, {'client_id': self.session.client_id})
        self.keepalive_timeout = self.session.keep_alive
        if self.keepalive_timeout <= 0:
            self.keepalive_timeout = None

    def attach(self, session, reader: ReaderAdapter, writer: WriterAdapter):
        if self.session:
            raise ProtocolHandlerException("Handler is already attached to a session")
        self._init_session(session)
        self.reader = reader
        self.writer = writer

    def detach(self):
        self.session = None
        self.reader = None
        self.writer = None

    def _is_attached(self):
        if self.session:
            return True
        else:
            return False

    async def start(self):
        if not self._is_attached():
            raise ProtocolHandlerException("Handler is not attached to a stream")
        self._reader_ready = asyncio.Event(loop=self._loop)
        self._reader_task = asyncio.Task(self._reader_loop(), loop=self._loop)
        await asyncio.wait([self._reader_ready.wait()], loop=self._loop)
        if self.keepalive_timeout:
            self._keepalive_task = self._loop.call_later(self.keepalive_timeout, self.handle_write_timeout)

        print("Handler tasks started")
        await self._retry_deliveries()
        print("Handler ready")

    async def stop(self):
        self._stop_waiters()
        if self._keepalive_task:
            self._keepalive_task.cancel()
        print("waiting for tasks to be stopped")
        if not self._reader_task.done():
            self._reader_task.cancel()
            await asyncio.wait([self._reader_stopped.wait()], loop=self._loop)
        print("closing writer")
        try:
            await self.writer.close()
        except Exception as e:
            print("Handler writer close failed: %s" % e)

    def _stop_waiters(self):
        print("Stopping %d puback waiters" % len(self._puback_waiters))
        print("Stopping %d pucomp waiters" % len(self._pubcomp_waiters))
        print("Stopping %d purec waiters" % len(self._pubrec_waiters))
        print("Stopping %d purel waiters" % len(self._pubrel_waiters))
        for waiter in itertools.chain(
                self._puback_waiters.values(),
                self._pubcomp_waiters.values(),
                self._pubrec_waiters.values(),
                self._pubrel_waiters.values()
        ):
            waiter.cancel()

    async def _retry_deliveries(self):
        print("Begin messages delivery retries")
        tasks = []
        for message in itertools.chain(self.session.inflight_in.values(), self.session.inflight_out.values()):
            tasks.append(asyncio.wait_for(self._handle_message_flow(message), 10, loop=self._loop))
        if tasks:
            done, pending = await asyncio.wait(tasks, loop=self._loop)
            print("%d messages redelivered" % len(done))
            print("%d messages not redelivered due to timeout" % len(pending))
        print("End messages delivery retries")

    async def mqtt_publish(self, topic, data, qos, retain, ack_timeout=None):
        if qos in (QOS_1, QOS_2):
            packet_id = self.session.next_packet_id
            if packet_id in self.session.inflight_out:
                raise HZMQTTException("A message with the same packet ID '%d' is already in flight" % packet_id)
        else:
            packet_id = None

        message = OutgoingApplicationMessage(packet_id, topic, qos, data, retain)
        if ack_timeout is not None and ack_timeout > 0:
            await asyncio.wait_for(self._handle_message_flow(message), ack_timeout, loop=self._loop)
        else:
            await self._handle_message_flow(message)

        return message

    async def _handle_message_flow(self, app_message):
        if app_message.qos == QOS_0:
            await self._handle_qos0_message_flow(app_message)
        elif app_message.qos == QOS_1:
            await self._handle_qos1_message_flow(app_message)
        elif app_message.qos == QOS_2:
            await self._handle_qos2_message_flow(app_message)
        else:
            raise HZMQTTException("Unexcepted QOS value '%d" % str(app_message.qos))

    async def _handle_qos0_message_flow(self, app_message):
        assert app_message.qos == QOS_0
        if app_message.direction == OUTGOING:
            packet = app_message.build_publish_packet()
            await self._send_packet(packet)
            app_message.publish_packet = packet
        elif app_message.direction == INCOMING:
            if app_message.publish_packet.dup_flag:
                print("[MQTT-3.3.1-2] DUP flag must set to 0 for QOS 0 message. Message ignored: %s" % repr(app_message.publish_packet))
            else:
                try:
                    self.session.delivered_message_queue.put_nowait(app_message)
                except:
                    print("delivered messages queue full. QOS_0 message discarded")

    async def _handle_qos1_message_flow(self, app_message):
        assert app_message.qos == QOS_1
        if app_message.puback_packet:
            raise HZMQTTException("Message '%d' has already been acknowledged" % app_message.packet_id)
        if app_message.direction == OUTGOING:
            if app_message.packet_id not in self.session.inflight_out:
                self.session.inflight_out[app_message.packet_id] = app_message
            if app_message.publish_packet is not None:
                publish_packet = app_message.build_publish_packet(dup=True)
            else:
                publish_packet = app_message.build_publish_packet()

            await self._send_packet(publish_packet)
            app_message.publish_packet = publish_packet

            waiter = asyncio.Future(loop=self._loop)
            self._puback_waiters[app_message.packet_id] = waiter
            await waiter
            del self._puback_waiters[app_message.packet_id]
            app_message.puback_packet = waiter.result()

            del self.session.inflight_out[app_message.packet_id]
        elif app_message.direction == INCOMING:
            print("Add message to delivery")
            await self.session.delivered_message_queue.put(app_message)

            puback = PubackPacket.build(app_message.packet_id)
            await self._send_packet(puback)
            app_message.puback_packet = puback

    async def _handle_qos2_message_flow(self, app_message):
        assert app_message.qos == QOS_2
        if app_message.direction == OUTGOING:
            if app_message.pubrel_packet and app_message.pubcomp_packet:
                raise HZMQTTException("Message '%d' has already been acknowledged" % app_message.packet_id)
            if not app_message.pubrel_packet:
                if app_message.publish_packet is not None:
                    if app_message.packet_id not in self.session.inflight_out:
                        raise HZMQTTException("Unknown inflight message '%d' in session" % app_message.packet_id)
                    publish_packet = app_message.build_publish_packet(dup=True)
                else:
                    self.session.inflight_out[app_message.packet_id] = app_message
                    publish_packet = app_message.build_publish_packet()

                await self._send_packet(publish_packet)
                app_message.publish_packet = publish_packet

                if app_message.packet_id in self._pubrec_waiters:
                    message = "Can't add PUBREC waiter, a waiter already exists for message Id '%s'" % app_message.packet_id
                    print(message)
                    raise HZMQTTException(message)
                waiter = asyncio.Future(loop=self._loop)
                self._pubrec_waiters[app_message.packet_id] = waiter
                await waiter
                del self._pubrec_waiters[app_message.packet_id]
                app_message.pubrec_packet = waiter.result()
            if not app_message.pubcomp_packet:
                app_message.pubrel_packet = PubrelPacket.build(app_message.packet_id)
                await self._send_packet(app_message.pubrel_packet)

                waiter = asyncio.Future(loop=self._loop)
                self._pubcomp_waiters[app_message.packet_id] = waiter
                await waiter
                del self._pubcomp_waiters[app_message.packet_id]
                app_message.pubcomp_packet = waiter.result()

            del self.session.inflight_out[app_message.packet_id]
        elif app_message.direction == INCOMING:
            self.session.inflight_in[app_message.packet_id] = app_message

            pubrec_packet = PubrecPacket.build(app_message.packet_id)
            await self._send_packet(pubrec_packet)
            app_message.pubrec_packet = pubrec_packet

            if app_message.packet_id in self._pubrel_waiters and not self._pubrel_waiters[app_message.packet_id].done():
                message = "A waiter already exists for message Id '%s', canceling it" % app_message.packet_id
                print(message)
                self._pubrel_waiters[app_message.packet_id].cancel()
            try:
                waiter = asyncio.Future(loop=self._loop)
                self._pubrel_waiters[app_message.packet_id] = waiter
                await waiter
                del self._pubrel_waiters[app_message.packet_id]
                app_message.pubrel_packet = waiter.result()

                await self.session.delivered_message_queue.put(app_message)
                del self.session.inflight_in[app_message.packet_id]

                pubcomp_packet = PubcompPacket.build(app_message.packet_id)
                await self._send_packet(pubcomp_packet)
                app_message.pubcomp_packet = pubcomp_packet
            except asyncio.CancelledError:
                print("Message flow cancelled")

    async def _reader_loop(self):
        print("%s Starting reader coro" % self.session.client_id)
        running_tasks = collections.deque()
        keepalive_timeout = self.session.keep_alive
        if keepalive_timeout <= 0:
            keepalive_timeout = None
        while True:
            try:
                self._reader_ready.set()
                while running_tasks and running_tasks[0].done():
                    running_tasks.popleft()
                if len(running_tasks) > 1:
                    print("handler running tasks: %d" % len(running_tasks))

                fixed_header = await asyncio.wait_for(
                    MQTTFixedHeader.from_stream(self.reader),
                    keepalive_timeout, loop=self._loop
                )
                if fixed_header:
                    if fixed_header.packet_type == RESERVED_0 or fixed_header.packet_type == RESERVED_15:
                        print("%s Received reserved packet, which is forbidden: closing connection" % self.session.client_id)
                        await self.handle_connection_closed()
                    else:
                        cls = packet_class(fixed_header)
                        packet = await cls.from_stream(self.reader, fixed_header=fixed_header)
                        print(f'EVENT_MQTT_PACKET_RECEIVED: {packet}')
                        task = None
                        if packet.fixed_header.packet_type == CONNACK:
                            task = asyncio.ensure_future(self.handle_connack(packet), loop=self._loop)
                        elif packet.fixed_header.packet_type == SUBSCRIBE:
                            task = asyncio.ensure_future(self.handle_subscribe(packet), loop=self._loop)
                        elif packet.fixed_header.packet_type == UNSUBSCRIBE:
                            task = asyncio.ensure_future(self.handle_unsubscribe(packet), loop=self._loop)
                        elif packet.fixed_header.packet_type == SUBACK:
                            task = asyncio.ensure_future(self.handle_suback(packet), loop=self._loop)
                        elif packet.fixed_header.packet_type == UNSUBACK:
                            task = asyncio.ensure_future(self.handle_unsuback(packet), loop=self._loop)
                        elif packet.fixed_header.packet_type == PUBACK:
                            task = asyncio.ensure_future(self.handle_puback(packet), loop=self._loop)
                        elif packet.fixed_header.packet_type == PUBREC:
                            task = asyncio.ensure_future(self.handle_pubrec(packet), loop=self._loop)
                        elif packet.fixed_header.packet_type == PUBREL:
                            task = asyncio.ensure_future(self.handle_pubrel(packet), loop=self._loop)
                        elif packet.fixed_header.packet_type == PUBCOMP:
                            task = asyncio.ensure_future(self.handle_pubcomp(packet), loop=self._loop)
                        elif packet.fixed_header.packet_type == PINGREQ:
                            task = asyncio.ensure_future(self.handle_pingreq(packet), loop=self._loop)
                        elif packet.fixed_header.packet_type == PINGRESP:
                            task = asyncio.ensure_future(self.handle_pingresp(packet), loop=self._loop)
                        elif packet.fixed_header.packet_type == PUBLISH:
                            task = asyncio.ensure_future(self.handle_publish(packet), loop=self._loop)
                        elif packet.fixed_header.packet_type == DISCONNECT:
                            task = asyncio.ensure_future(self.handle_disconnect(packet), loop=self._loop)
                        elif packet.fixed_header.packet_type == CONNECT:
                            self.handle_connect(packet)
                        else:
                            print("%s Unhandled packet type: %s" % (self.session.client_id, packet.fixed_header.packet_type))
                        if task:
                            running_tasks.append(task)
                else:
                    print("No more data (EOF received), stopping reader coro")
                    break
            except MQTTException:
                print("Message discarded")
            except asyncio.CancelledError:
                print("Task cancelled, reader loop ending")
                break
            except asyncio.TimeoutError:
                print("Input stream read timeout")
                self.handle_read_timeout()
            except NoDataException:
                print("No data available")
            except BaseException as e:
                print("%s Unhandled exception in reader coro: %r" % (type(self).__name__, e))
                break
        while running_tasks:
            running_tasks.popleft().cancel()
        await self.handle_connection_closed()
        self._reader_stopped.set()
        print("Reader coro stopped")
        await self.stop()

    async def _send_packet(self, packet):
        try:
            with (await self._write_lock):
                await packet.to_stream(self.writer)
            if self._keepalive_task:
                self._keepalive_task.cancel()
                self._keepalive_task = self._loop.call_later(self.keepalive_timeout, self.handle_write_timeout)

            print(f'EVENT_MQTT_PACKET_SENT: {packet}')
        except (ConnectionResetError, BrokenPipeError):
            await self.handle_connection_closed()
        except asyncio.CancelledError:
            raise
        except BaseException as e:
            print("Unhandled exception: %s" % e)
            raise

    async def mqtt_deliver_next_message(self):
        if not self._is_attached():
            return None
        if self.logger.isEnabledFor(logging.DEBUG):
            print("%d message(s) available for delivery" % self.session.delivered_message_queue.qsize())
        try:
            message = await self.session.delivered_message_queue.get()
        except asyncio.CancelledError:
            message = None
        if self.logger.isEnabledFor(logging.DEBUG):
            print("Delivering message %s" % message)
        return message

    def handle_write_timeout(self):
        print('%s write timeout unhandled' % self.session.client_id)

    def handle_read_timeout(self):
        print('%s read timeout unhandled' % self.session.client_id)

    async def handle_connack(self, connack: ConnackPacket):
        print('%s CONNACK unhandled' % self.session.client_id)

    async def handle_connect(self, connect: ConnectPacket):
        print('%s CONNECT unhandled' % self.session.client_id)

    async def handle_subscribe(self, subscribe: SubscribePacket):
        print('%s SUBSCRIBE unhandled' % self.session.client_id)

    async def handle_unsubscribe(self, subscribe: UnsubscribePacket):
        print('%s UNSUBSCRIBE unhandled' % self.session.client_id)

    async def handle_suback(self, suback: SubackPacket):
        print('%s SUBACK unhandled' % self.session.client_id)

    async def handle_unsuback(self, unsuback: UnsubackPacket):
        print('%s UNSUBACK unhandled' % self.session.client_id)

    async def handle_pingresp(self, pingresp: PingRespPacket):
        print('%s PINGRESP unhandled' % self.session.client_id)

    async def handle_pingreq(self, pingreq: PingReqPacket):
        print('%s PINGREQ unhandled' % self.session.client_id)

    async def handle_disconnect(self, disconnect: DisconnectPacket):
        print('%s DISCONNECT unhandled' % self.session.client_id)

    async def handle_connection_closed(self):
        print('%s Connection closed unhandled' % self.session.client_id)

    async def handle_puback(self, puback: PubackPacket):
        packet_id = puback.variable_header.packet_id
        try:
            waiter = self._puback_waiters[packet_id]
            waiter.set_result(puback)
        except KeyError:
            print("Received PUBACK for unknown pending message Id: '%d'" % packet_id)
        except InvalidStateError:
            print("PUBACK waiter with Id '%d' already done" % packet_id)

    async def handle_pubrec(self, pubrec: PubrecPacket):
        packet_id = pubrec.packet_id
        try:
            waiter = self._pubrec_waiters[packet_id]
            waiter.set_result(pubrec)
        except KeyError:
            print("Received PUBREC for unknown pending message with Id: %d" % packet_id)
        except InvalidStateError:
            print("PUBREC waiter with Id '%d' already done" % packet_id)

    async def handle_pubcomp(self, pubcomp: PubcompPacket):
        packet_id = pubcomp.packet_id
        try:
            waiter = self._pubcomp_waiters[packet_id]
            waiter.set_result(pubcomp)
        except KeyError:
            print("Received PUBCOMP for unknown pending message with Id: %d" % packet_id)
        except InvalidStateError:
            print("PUBCOMP waiter with Id '%d' already done" % packet_id)

    async def handle_pubrel(self, pubrel: PubrelPacket):
        packet_id = pubrel.packet_id
        try:
            waiter = self._pubrel_waiters[packet_id]
            waiter.set_result(pubrel)
        except KeyError:
            print("Received PUBREL for unknown pending message with Id: %d" % packet_id)
        except InvalidStateError:
            print("PUBREL waiter with Id '%d' already done" % packet_id)

    async def handle_publish(self, publish_packet: PublishPacket):
        packet_id = publish_packet.variable_header.packet_id
        qos = publish_packet.qos

        incoming_message = IncomingApplicationMessage(packet_id, publish_packet.topic_name, qos, publish_packet.data, publish_packet.retain_flag)
        incoming_message.publish_packet = publish_packet
        await self._handle_message_flow(incoming_message)
        print("Message queue size: %d" % self.session.delivered_message_queue.qsize())


class ClientProtocolHandler(ProtocolHandler):
    def __init__(self, session: Session = None, loop=None):
        super().__init__(session, loop=loop)
        self._ping_task = None
        self._pingresp_queue = asyncio.Queue(loop=self._loop)
        self._subscriptions_waiter = dict()
        self._unsubscriptions_waiter = dict()
        self._disconnect_waiter = None

    async def start(self):
        await super().start()
        if self._disconnect_waiter is None:
            self._disconnect_waiter = futures.Future(loop=self._loop)

    async def stop(self):
        await super().stop()
        if self._ping_task:
            try:
                print("Cancel ping task")
                self._ping_task.cancel()
            except BaseException:
                pass
        if not self._disconnect_waiter.done():
            self._disconnect_waiter.cancel()
        self._disconnect_waiter = None

    def _build_connect_packet(self):
        vh = ConnectVariableHeader()
        payload = ConnectPayload()

        vh.keep_alive = self.session.keep_alive
        vh.clean_session_flag = self.session.clean_session
        vh.will_retain_flag = self.session.will_retain
        payload.client_id = self.session.client_id

        if self.session.username:
            vh.username_flag = True
            payload.username = self.session.username
        else:
            vh.username_flag = False

        if self.session.password:
            vh.password_flag = True
            payload.password = self.session.password
        else:
            vh.password_flag = False
        if self.session.will_flag:
            vh.will_flag = True
            vh.will_qos = self.session.will_qos
            payload.will_message = self.session.will_message
            payload.will_topic = self.session.will_topic
        else:
            vh.will_flag = False

        packet = ConnectPacket(vh=vh, payload=payload)
        return packet

    async def mqtt_connect(self):
        connect_packet = self._build_connect_packet()
        await self._send_packet(connect_packet)
        connack = await ConnackPacket.from_stream(self.reader)
        print(f'EVENT_MQTT_PACKET_RECEIVED: {connack}')
        return connack.return_code

    def handle_write_timeout(self):
        try:
            if not self._ping_task:
                print("Scheduling Ping")
                self._ping_task = asyncio.ensure_future(self.mqtt_ping())
        except BaseException as be:
            print("Exception ignored in ping task: %r" % be)

    def handle_read_timeout(self):
        pass

    async def mqtt_subscribe(self, topics, packet_id):
        subscribe = SubscribePacket.build(topics, packet_id)
        await self._send_packet(subscribe)

        waiter = futures.Future(loop=self._loop)
        self._subscriptions_waiter[subscribe.variable_header.packet_id] = waiter
        return_codes = await waiter

        del self._subscriptions_waiter[subscribe.variable_header.packet_id]
        return return_codes

    async def handle_suback(self, suback: SubackPacket):
        packet_id = suback.variable_header.packet_id
        try:
            waiter = self._subscriptions_waiter.get(packet_id)
            waiter.set_result(suback.payload.return_codes)
        except KeyError as ke:
            print("Received SUBACK for unknown pending subscription with Id: %s" % packet_id)

    async def mqtt_unsubscribe(self, topics, packet_id):
        unsubscribe = UnsubscribePacket.build(topics, packet_id)
        await self._send_packet(unsubscribe)
        waiter = futures.Future(loop=self._loop)
        self._unsubscriptions_waiter[unsubscribe.variable_header.packet_id] = waiter
        await waiter
        del self._unsubscriptions_waiter[unsubscribe.variable_header.packet_id]

    async def handle_unsuback(self, unsuback: UnsubackPacket):
        packet_id = unsuback.variable_header.packet_id
        try:
            waiter = self._unsubscriptions_waiter.get(packet_id)
            waiter.set_result(None)
        except KeyError:
            print("Received UNSUBACK for unknown pending subscription with Id: %s" % packet_id)

    async def mqtt_disconnect(self):
        disconnect_packet = DisconnectPacket()
        await self._send_packet(disconnect_packet)

    async def mqtt_ping(self):
        ping_packet = PingReqPacket()
        await self._send_packet(ping_packet)
        resp = await self._pingresp_queue.get()
        if self._ping_task:
            self._ping_task = None
        return resp

    async def handle_pingresp(self, pingresp: PingRespPacket):
        await self._pingresp_queue.put(pingresp)

    async def handle_connection_closed(self):
        print("Broker closed connection")
        if not self._disconnect_waiter.done():
            self._disconnect_waiter.set_result(None)

    async def wait_disconnect(self):
        await self._disconnect_waiter


class BrokerProtocolHandler(ProtocolHandler):
    def __init__(self, session: Session = None, loop=None):
        super().__init__(session, loop)
        self._disconnect_waiter = None
        self._pending_subscriptions = Queue(loop=self._loop)
        self._pending_unsubscriptions = Queue(loop=self._loop)

    async def start(self):
        await super().start()
        if self._disconnect_waiter is None:
            self._disconnect_waiter = futures.Future(loop=self._loop)

    async def stop(self):
        await super().stop()
        if self._disconnect_waiter is not None and not self._disconnect_waiter.done():
            self._disconnect_waiter.set_result(None)

    async def wait_disconnect(self):
        return await self._disconnect_waiter

    def handle_write_timeout(self):
        pass

    def handle_read_timeout(self):
        if self._disconnect_waiter is not None and not self._disconnect_waiter.done():
            self._disconnect_waiter.set_result(None)

    async def handle_disconnect(self, disconnect):
        print("Client disconnecting")
        if self._disconnect_waiter and not self._disconnect_waiter.done():
            print("Setting waiter result to %r" % disconnect)
            self._disconnect_waiter.set_result(disconnect)

    async def handle_connection_closed(self):
        await self.handle_disconnect(None)

    async def handle_connect(self, connect: ConnectPacket):
        print('%s [MQTT-3.1.0-2] %s : CONNECT message received during messages handling' % (self.session.client_id, format_client_message(self.session)))
        if self._disconnect_waiter is not None and not self._disconnect_waiter.done():
            self._disconnect_waiter.set_result(None)

    async def handle_pingreq(self, pingreq: PingReqPacket):
        await self._send_packet(PingRespPacket.build())

    async def handle_subscribe(self, subscribe: SubscribePacket):
        subscription = {'packet_id': subscribe.variable_header.packet_id, 'topics': subscribe.payload.topics}
        await self._pending_subscriptions.put(subscription)

    async def handle_unsubscribe(self, unsubscribe: UnsubscribePacket):
        unsubscription = {'packet_id': unsubscribe.variable_header.packet_id, 'topics': unsubscribe.payload.topics}
        await self._pending_unsubscriptions.put(unsubscription)

    async def get_next_pending_subscription(self):
        subscription = await self._pending_subscriptions.get()
        return subscription

    async def get_next_pending_unsubscription(self):
        unsubscription = await self._pending_unsubscriptions.get()
        return unsubscription

    async def mqtt_acknowledge_subscription(self, packet_id, return_codes):
        suback = SubackPacket.build(packet_id, return_codes)
        await self._send_packet(suback)

    async def mqtt_acknowledge_unsubscription(self, packet_id):
        unsuback = UnsubackPacket.build(packet_id)
        await self._send_packet(unsuback)

    async def mqtt_connack_authorize(self, authorize: bool):
        if authorize:
            connack = ConnackPacket.build(self.session.parent, CONNECTION_ACCEPTED)
        else:
            connack = ConnackPacket.build(self.session.parent, NOT_AUTHORIZED)
        await self._send_packet(connack)

    @classmethod
    async def init_from_connect(cls, reader: ReaderAdapter, writer: WriterAdapter, loop=None):
        remote_address, remote_port = writer.get_peer_info()
        connect = await ConnectPacket.from_stream(reader)

        if connect.payload.client_id is None:
            raise MQTTException('[[MQTT-3.1.3-3]] : Client identifier must be present')

        if connect.variable_header.will_flag:
            if connect.payload.will_topic is None or connect.payload.will_message is None:
                raise MQTTException('will flag set, but will topic/message not present in payload')

        if connect.variable_header.reserved_flag:
            raise MQTTException('[MQTT-3.1.2-3] CONNECT reserved flag must be set to 0')
        if connect.proto_name != "MQTT":
            raise MQTTException('[MQTT-3.1.2-1] Incorrect protocol name: "%s"' % connect.proto_name)

        connack = None
        error_msg = None
        if connect.proto_level != 4:
            error_msg = 'Invalid protocol from %s: %d' % (format_client_message(address=remote_address, port=remote_port), connect.proto_level)
            connack = ConnackPacket.build(0, UNACCEPTABLE_PROTOCOL_VERSION)
        elif not connect.username_flag and connect.password_flag:
            connack = ConnackPacket.build(0, BAD_USERNAME_PASSWORD)
        elif connect.username_flag and not connect.password_flag:
            connack = ConnackPacket.build(0, BAD_USERNAME_PASSWORD)
        elif connect.username_flag and connect.username is None:
            error_msg = 'Invalid username from %s' % (format_client_message(address=remote_address, port=remote_port))
            connack = ConnackPacket.build(0, BAD_USERNAME_PASSWORD)
        elif connect.password_flag and connect.password is None:
            error_msg = 'Invalid password %s' % (format_client_message(address=remote_address, port=remote_port))
            connack = ConnackPacket.build(0, BAD_USERNAME_PASSWORD)  # [MQTT-3.2.2-4] session_parent=0
        elif connect.clean_session_flag is False and connect.payload.client_id_is_random:
            error_msg = '[MQTT-3.1.3-8] [MQTT-3.1.3-9] %s: No client Id provided (cleansession=0)' % (format_client_message(address=remote_address, port=remote_port))
            connack = ConnackPacket.build(0, IDENTIFIER_REJECTED)
        if connack is not None:
            await connack.to_stream(writer)
            await writer.close()
            raise MQTTException(error_msg)

        incoming_session = Session(loop)
        incoming_session.client_id = connect.client_id
        incoming_session.clean_session = connect.clean_session_flag
        incoming_session.will_flag = connect.will_flag
        incoming_session.will_retain = connect.will_retain_flag
        incoming_session.will_qos = connect.will_qos
        incoming_session.will_topic = connect.will_topic
        incoming_session.will_message = connect.will_message
        incoming_session.username = connect.username
        incoming_session.password = connect.password
        if connect.keep_alive > 0:
            incoming_session.keep_alive = connect.keep_alive
        else:
            incoming_session.keep_alive = 0

        handler = cls(loop=loop)
        return handler, incoming_session
