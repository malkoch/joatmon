import asyncio
import logging
import re
from asyncio import CancelledError
from collections import deque
from functools import partial

import websockets
from transitions import (
    Machine,
    MachineError
)

from .adapters import (
    ReaderAdapter,
    StreamReaderAdapter,
    StreamWriterAdapter,
    WebSocketsReader,
    WebSocketsWriter,
    WriterAdapter
)
from .errors import (
    HZMQTTException,
    MQTTException
)
from .mqtt.handler import BrokerProtocolHandler
from .session import Session
from .utils import (
    format_client_message,
    gen_client_id
)

_defaults = {
    'timeout-disconnect-delay': 2,
    'auth': {
        'allow-anonymous': True,
        'password-file': None
    },
}

EVENT_BROKER_PRE_START = 'broker_pre_start'
EVENT_BROKER_POST_START = 'broker_post_start'
EVENT_BROKER_PRE_SHUTDOWN = 'broker_pre_shutdown'
EVENT_BROKER_POST_SHUTDOWN = 'broker_post_shutdown'
EVENT_BROKER_CLIENT_CONNECTED = 'broker_client_connected'
EVENT_BROKER_CLIENT_DISCONNECTED = 'broker_client_disconnected'
EVENT_BROKER_CLIENT_SUBSCRIBED = 'broker_client_subscribed'
EVENT_BROKER_CLIENT_UNSUBSCRIBED = 'broker_client_unsubscribed'
EVENT_BROKER_MESSAGE_RECEIVED = 'broker_message_received'


class BrokerException(BaseException):
    pass


class RetainedApplicationMessage:
    __slots__ = ('source_session', 'topic', 'data', 'qos')

    def __init__(self, source_session, topic, data, qos=None):
        self.source_session = source_session
        self.topic = topic
        self.data = data
        self.qos = qos


class Server:
    def __init__(self, listener_name, server_instance, max_connections=-1, loop=None):
        self.logger = logging.getLogger(__name__)
        self.instance = server_instance
        self.conn_count = 0
        self.listener_name = listener_name
        if loop is not None:
            self._loop = loop
        else:
            self._loop = asyncio.get_event_loop()

        self.max_connections = max_connections
        if self.max_connections > 0:
            self.semaphore = asyncio.Semaphore(self.max_connections, loop=self._loop)
        else:
            self.semaphore = None

    async def acquire_connection(self):
        if self.semaphore:
            await self.semaphore.acquire()
        self.conn_count += 1
        if self.max_connections > 0:
            self.logger.info("Listener '%s': %d/%d connections acquired" % (self.listener_name, self.conn_count, self.max_connections))
        else:
            self.logger.info("Listener '%s': %d connections acquired" % (self.listener_name, self.conn_count))

    def release_connection(self):
        if self.semaphore:
            self.semaphore.release()
        self.conn_count -= 1
        if self.max_connections > 0:
            self.logger.info("Listener '%s': %d/%d connections acquired" % (self.listener_name, self.conn_count, self.max_connections))
        else:
            self.logger.info("Listener '%s': %d connections acquired" % (self.listener_name, self.conn_count))

    async def close_instance(self):
        if self.instance:
            self.instance.close()
            await self.instance.wait_closed()


class Broker:
    states = ['new', 'starting', 'started', 'not_started', 'stopping', 'stopped', 'not_stopped', 'stopped']

    def __init__(self, config=None, loop=None):
        self.logger = logging.getLogger(__name__)
        self.config = _defaults
        if config is not None:
            self.config.update(config)
        self._build_listeners_config(self.config)

        if loop is not None:
            self._loop = loop
        else:
            self._loop = asyncio.get_event_loop()

        self._servers = dict()
        self._init_states()
        self._sessions = dict()
        self._subscriptions = dict()
        self._retained_messages = dict()
        self._broadcast_queue = asyncio.Queue(loop=self._loop)

        self._broadcast_task = None

    def _build_listeners_config(self, broker_config):
        self.listeners_config = dict()
        try:
            listeners_config = broker_config['listeners']
            defaults = listeners_config['default']
            for listener in listeners_config:
                config = dict(defaults)
                config.update(listeners_config[listener])
                self.listeners_config[listener] = config
        except KeyError as ke:
            raise BrokerException("Listener config not found invalid: %s" % ke)

    def _init_states(self):
        self.transitions = Machine(states=Broker.states, initial='new')
        self.transitions.add_transition(trigger='start', source='new', dest='starting')
        self.transitions.add_transition(trigger='starting_fail', source='starting', dest='not_started')
        self.transitions.add_transition(trigger='starting_success', source='starting', dest='started')
        self.transitions.add_transition(trigger='shutdown', source='started', dest='stopping')
        self.transitions.add_transition(trigger='stopping_success', source='stopping', dest='stopped')
        self.transitions.add_transition(trigger='stopping_failure', source='stopping', dest='not_stopped')
        self.transitions.add_transition(trigger='start', source='stopped', dest='starting')

    async def start(self):
        try:
            self._sessions = dict()
            self._subscriptions = dict()
            self._retained_messages = dict()
            self.transitions.start()
            self.logger.debug("Broker starting")
        except (MachineError, ValueError) as exc:
            self.logger.warning("[WARN-0001] Invalid method call at this moment: %s" % exc)
            raise BrokerException("Broker instance can't be started: %s" % exc)

        self.logger.info('EVENT_BROKER_PRE_START')
        try:
            for listener_name in self.listeners_config:
                listener = self.listeners_config[listener_name]

                if 'bind' not in listener:
                    self.logger.debug("Listener configuration '%s' is not bound" % listener_name)
                else:
                    try:
                        max_connections = listener['max_connections']
                    except KeyError:
                        max_connections = -1

                    address, s_port = listener['bind'].split(':')

                    try:
                        port = int(s_port)
                    except ValueError:
                        raise BrokerException("Invalid port value in bind value: %s" % listener['bind'])

                    if listener['type'] == 'tcp':
                        cb_partial = partial(self.stream_connected, listener_name=listener_name)
                        instance = await asyncio.start_server(cb_partial, address, port, reuse_address=True, loop=self._loop)
                        self._servers[listener_name] = Server(listener_name, instance, max_connections, self._loop)
                    elif listener['type'] == 'ws':
                        cb_partial = partial(self.ws_connected, listener_name=listener_name)
                        instance = await websockets.serve(cb_partial, address, port, loop=self._loop, subprotocols=['mqtt'])
                        self._servers[listener_name] = Server(listener_name, instance, max_connections, self._loop)

                    self.logger.info("Listener '%s' bind to %s (max_connections=%d)" % (listener_name, listener['bind'], max_connections))

            self.transitions.starting_success()
            self.logger.info('EVENT_BROKER_POST_START')

            self._broadcast_task = asyncio.ensure_future(self._broadcast_loop(), loop=self._loop)

            self.logger.debug("Broker started")
        except Exception as e:
            self.logger.error("Broker startup failed: %s" % e)
            self.transitions.starting_fail()
            raise BrokerException("Broker instance can't be started: %s" % e)

    async def shutdown(self):
        try:
            self._sessions = dict()
            self._subscriptions = dict()
            self._retained_messages = dict()
            self.transitions.shutdown()
        except (MachineError, ValueError) as exc:
            self.logger.debug("Invalid method call at this moment: %s" % exc)
            raise BrokerException("Broker instance can't be stopped: %s" % exc)

        self.logger.info('EVENT_BROKER_PRE_SHUTDOWN')

        if self._broadcast_task:
            self._broadcast_task.cancel()
        if self._broadcast_queue.qsize() > 0:
            self.logger.warning("%d messages not broadcasted" % self._broadcast_queue.qsize())

        for listener_name in self._servers:
            server = self._servers[listener_name]
            await server.close_instance()
        self.logger.debug("Broker closing")
        self.logger.info("Broker closed")
        self.logger.info('EVENT_BROKER_POST_SHUTDOWN')
        self.transitions.stopping_success()

    async def internal_message_broadcast(self, topic, data, qos=None):
        return await self._broadcast_message(None, topic, data)

    async def ws_connected(self, websocket, uri, listener_name):
        await self.client_connected(listener_name, WebSocketsReader(websocket), WebSocketsWriter(websocket))

    async def stream_connected(self, reader, writer, listener_name):
        await self.client_connected(listener_name, StreamReaderAdapter(reader), StreamWriterAdapter(writer))

    async def client_connected(self, listener_name, reader: ReaderAdapter, writer: WriterAdapter):
        print('new client connected')
        server = self._servers.get(listener_name, None)
        if not server:
            raise BrokerException("Invalid listener name '%s'" % listener_name)
        await server.acquire_connection()

        remote_address, remote_port = writer.get_peer_info()
        self.logger.info("Connection from %s:%d on listener '%s'" % (remote_address, remote_port, listener_name))

        try:
            handler, client_session = await BrokerProtocolHandler.init_from_connect(reader, writer, loop=self._loop)
        except HZMQTTException as exc:
            self.logger.warning("[MQTT-3.1.0-1] %s: Can't read first packet an CONNECT: %s" % (format_client_message(address=remote_address, port=remote_port), exc))
            self.logger.debug("Connection closed")
            return
        except MQTTException as me:
            self.logger.error('Invalid connection from %s : %s' % (format_client_message(address=remote_address, port=remote_port), me))
            await writer.close()
            self.logger.debug("Connection closed")
            return

        if client_session.clean_session:
            if client_session.client_id is not None and client_session.client_id != "":
                self.delete_session(client_session.client_id)
            else:
                client_session.client_id = gen_client_id()
            client_session.parent = 0
        else:
            if client_session.client_id in self._sessions:
                self.logger.debug("Found old session %s" % repr(self._sessions[client_session.client_id]))
                (client_session, h) = self._sessions[client_session.client_id]
                client_session.parent = 1
            else:
                client_session.parent = 0
        if client_session.keep_alive > 0:
            client_session.keep_alive += self.config['timeout-disconnect-delay']
        self.logger.debug("Keep-alive timeout=%d" % client_session.keep_alive)

        handler.attach(client_session, reader, writer)
        self._sessions[client_session.client_id] = (client_session, handler)

        authenticated = await self.authenticate(client_session, self.listeners_config[listener_name])
        if not authenticated:
            await writer.close()
            server.release_connection()
            return

        while True:
            try:
                client_session.transitions.connect()
                break
            except (MachineError, ValueError):
                self.logger.warning("Client %s is reconnecting too quickly, make it wait" % client_session.client_id)
                await asyncio.sleep(1, loop=self._loop)
        await handler.mqtt_connack_authorize(authenticated)

        self.logger.info(f'EVENT_BROKER_CLIENT_CONNECTED: {client_session.client_id}')

        self.logger.debug("%s Start messages handling" % client_session.client_id)
        await handler.start()
        self.logger.debug("Retained messages queue size: %d" % client_session.retained_messages.qsize())
        await self.publish_session_retained_messages(client_session)

        disconnect_waiter = asyncio.ensure_future(handler.wait_disconnect(), loop=self._loop)
        subscribe_waiter = asyncio.ensure_future(handler.get_next_pending_subscription(), loop=self._loop)
        unsubscribe_waiter = asyncio.ensure_future(handler.get_next_pending_unsubscription(), loop=self._loop)
        wait_deliver = asyncio.ensure_future(handler.mqtt_deliver_next_message(), loop=self._loop)
        connected = True
        while connected:
            try:
                done, pending = await asyncio.wait(
                    [disconnect_waiter, subscribe_waiter, unsubscribe_waiter, wait_deliver],
                    return_when=asyncio.FIRST_COMPLETED, loop=self._loop
                )
                if disconnect_waiter in done:
                    print('disconnection')
                    result = disconnect_waiter.result()
                    self.logger.debug("%s Result from wait_diconnect: %s" % (client_session.client_id, result))
                    if result is None:
                        self.logger.debug("Will flag: %s" % client_session.will_flag)
                        if client_session.will_flag:
                            self.logger.debug("Client %s disconnected abnormally, sending will message" % format_client_message(client_session))
                            await self._broadcast_message(client_session, client_session.will_topic, client_session.will_message, client_session.will_qos)
                            if client_session.will_retain:
                                self.retain_message(client_session, client_session.will_topic, client_session.will_message, client_session.will_qos)
                    self.logger.debug("%s Disconnecting session" % client_session.client_id)
                    await self._stop_handler(handler)
                    client_session.transitions.disconnect()
                    self.logger.info(f'EVENT_BROKER_CLIENT_DISCONNECTED: {client_session.client_id}')
                    connected = False
                if unsubscribe_waiter in done:
                    print('unsubscribed')
                    self.logger.debug("%s handling unsubscription" % client_session.client_id)
                    unsubscription = unsubscribe_waiter.result()
                    for topic in unsubscription['topics']:
                        self._del_subscription(topic, client_session)
                        self.logger.info(f'EVENT_BROKER_CLIENT_UNSUBSCRIBED: {client_session.client_id}, {topic}')
                    await handler.mqtt_acknowledge_unsubscription(unsubscription['packet_id'])
                    unsubscribe_waiter = asyncio.Task(handler.get_next_pending_unsubscription(), loop=self._loop)
                if subscribe_waiter in done:
                    print('subscribed')
                    self.logger.debug("%s handling subscription" % client_session.client_id)
                    subscriptions = subscribe_waiter.result()
                    return_codes = []
                    for subscription in subscriptions['topics']:
                        result = await self.add_subscription(subscription, client_session)
                        return_codes.append(result)
                    await handler.mqtt_acknowledge_subscription(subscriptions['packet_id'], return_codes)
                    for index, subscription in enumerate(subscriptions['topics']):
                        if return_codes[index] != 0x80:
                            self.logger.info(f'EVENT_BROKER_CLIENT_SUBSCRIBED: {client_session.client_id}, {subscription[0]}, {subscription[1]}')
                            await self.publish_retained_messages_for_subscription(subscription, client_session)
                    subscribe_waiter = asyncio.Task(handler.get_next_pending_subscription(), loop=self._loop)
                    self.logger.debug(repr(self._subscriptions))
                if wait_deliver in done:
                    print('delivered')
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug("%s handling message delivery" % client_session.client_id)
                    app_message = wait_deliver.result()
                    if not app_message.topic:
                        self.logger.warning("[MQTT-4.7.3-1] - %s invalid TOPIC sent in PUBLISH message, closing connection" % client_session.client_id)
                        break
                    if "#" in app_message.topic or "+" in app_message.topic:
                        self.logger.warning("[MQTT-3.3.2-2] - %s invalid TOPIC sent in PUBLISH message, closing connection" % client_session.client_id)
                        break

                    self.logger.info(f'EVENT_BROKER_MESSAGE_RECEIVED: {client_session.client_id}, {app_message}')
                    await self._broadcast_message(client_session, app_message.topic, app_message.data)
                    if app_message.publish_packet.retain_flag:
                        self.retain_message(client_session, app_message.topic, app_message.data, app_message.qos)
                    wait_deliver = asyncio.Task(handler.mqtt_deliver_next_message(), loop=self._loop)
            except asyncio.CancelledError:
                self.logger.debug("Client loop cancelled")
                break
        disconnect_waiter.cancel()
        subscribe_waiter.cancel()
        unsubscribe_waiter.cancel()
        wait_deliver.cancel()

        self.logger.debug("%s Client disconnected" % client_session.client_id)
        server.release_connection()

    def _init_handler(self, session, reader, writer):
        handler = BrokerProtocolHandler(loop=self._loop)
        handler.attach(session, reader, writer)
        return handler

    async def _stop_handler(self, handler):
        try:
            await handler.stop()
        except Exception as e:
            self.logger.error(e)

    async def authenticate(self, session: Session, listener):
        auth_result = True
        return auth_result

    async def topic_filtering(self, session: Session, topic):
        topic_result = True
        return topic_result

    def retain_message(self, source_session, topic_name, data, qos=None):
        if data is not None and data != b'':
            self.logger.debug("Retaining message on topic %s" % topic_name)
            retained_message = RetainedApplicationMessage(source_session, topic_name, data, qos)
            self._retained_messages[topic_name] = retained_message
        else:
            if topic_name in self._retained_messages:
                self.logger.debug("Clear retained messages for topic '%s'" % topic_name)
                del self._retained_messages[topic_name]

    async def add_subscription(self, subscription, session):
        try:
            a_filter = subscription[0]
            if '#' in a_filter and not a_filter.endswith('#'):
                return 0x80
            if a_filter != "+":
                if '+' in a_filter:
                    if "/+" not in a_filter and "+/" not in a_filter:
                        return 0x80
            permitted = await self.topic_filtering(session, topic=a_filter)
            if not permitted:
                return 0x80
            qos = subscription[1]
            if 'max-qos' in self.config and qos > self.config['max-qos']:
                qos = self.config['max-qos']
            if a_filter not in self._subscriptions:
                self._subscriptions[a_filter] = []
            already_subscribed = next(
                (s for (s, qos) in self._subscriptions[a_filter] if s.client_id == session.client_id), None)
            if not already_subscribed:
                self._subscriptions[a_filter].append((session, qos))
            else:
                self.logger.debug("Client %s has already subscribed to %s" % (format_client_message(session=session), a_filter))
            return qos
        except KeyError:
            return 0x80

    def _del_subscription(self, a_filter, session):
        deleted = 0
        try:
            subscriptions = self._subscriptions[a_filter]
            for index, (sub_session, qos) in enumerate(subscriptions):
                if sub_session.client_id == session.client_id:
                    self.logger.debug("Removing subscription on topic '%s' for client %s" % (a_filter, format_client_message(session=session)))
                    subscriptions.pop(index)
                    deleted += 1
                    break
        except KeyError:
            pass
        finally:
            return deleted

    def _del_all_subscriptions(self, session):
        filter_queue = deque()
        for topic in self._subscriptions:
            if self._del_subscription(topic, session):
                filter_queue.append(topic)
        for topic in filter_queue:
            if not self._subscriptions[topic]:
                del self._subscriptions[topic]

    @staticmethod
    def matches(topic, a_filter):
        if "#" not in a_filter and "+" not in a_filter:
            return a_filter == topic
        else:
            match_pattern = re.compile(a_filter.replace('#', '.*').replace('$', '\$').replace('+', '[/\$\s\w\d]+'))
            return match_pattern.match(topic)

    async def _broadcast_loop(self):
        running_tasks = deque()
        try:
            while True:
                while running_tasks and running_tasks[0].done():
                    task = running_tasks.popleft()
                    try:
                        task.result()
                    except Exception:
                        pass
                broadcast = await self._broadcast_queue.get()
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug("broadcasting %r" % broadcast)
                for k_filter in self._subscriptions:
                    print(k_filter, broadcast['topic'])
                    if broadcast['topic'].startswith("$") and (k_filter.startswith("+") or k_filter.startswith("#")):
                        self.logger.debug("[MQTT-4.7.2-1] - ignoring brodcasting $ topic to subscriptions starting with + or #")
                    elif self.matches(broadcast['topic'], k_filter):
                        subscriptions = self._subscriptions[k_filter]
                        for (target_session, qos) in subscriptions:
                            if 'qos' in broadcast:
                                qos = broadcast['qos']
                            if target_session.transitions.state == 'connected':
                                if self.logger.isEnabledFor(logging.DEBUG):
                                    self.logger.debug("broadcasting application message from %s on topic '%s' to %s" % (format_client_message(session=broadcast['session']), broadcast['topic'], format_client_message(session=target_session)))
                                handler = self._get_handler(target_session)
                                task = asyncio.ensure_future(
                                    handler.mqtt_publish(broadcast['topic'], broadcast['data'], qos, retain=False),
                                    loop=self._loop)
                                running_tasks.append(task)
                            elif qos is not None and qos > 0:
                                if self.logger.isEnabledFor(logging.DEBUG):
                                    self.logger.debug("retaining application message from %s on topic '%s' to client '%s'" % (format_client_message(session=broadcast['session']), broadcast['topic'], format_client_message(session=target_session)))
                                retained_message = RetainedApplicationMessage(
                                    broadcast['session'], broadcast['topic'], broadcast['data'], qos)
                                await target_session.retained_messages.put(retained_message)
                                if self.logger.isEnabledFor(logging.DEBUG):
                                    self.logger.debug(f'target_session.retained_messages={target_session.retained_messages.qsize()}')
        except CancelledError:
            if running_tasks:
                await asyncio.wait(running_tasks, loop=self._loop)
            raise

    async def _broadcast_message(self, session, topic, data, force_qos=None):
        broadcast = {'session': session, 'topic': topic, 'data': data}
        if force_qos:
            broadcast['qos'] = force_qos
        await self._broadcast_queue.put(broadcast)

    async def publish_session_retained_messages(self, session):
        self.logger.debug("Publishing %d messages retained for session %s" % (session.retained_messages.qsize(), format_client_message(session=session)))
        publish_tasks = []
        handler = self._get_handler(session)
        while not session.retained_messages.empty():
            retained = await session.retained_messages.get()
            publish_tasks.append(asyncio.ensure_future(
                handler.mqtt_publish(retained.topic, retained.data, retained.qos, True), loop=self._loop))
        if publish_tasks:
            await asyncio.wait(publish_tasks, loop=self._loop)

    async def publish_retained_messages_for_subscription(self, subscription, session):
        self.logger.debug("Begin broadcasting messages retained due to subscription on '%s' from %s" % (subscription[0], format_client_message(session=session)))
        publish_tasks = []
        handler = self._get_handler(session)
        for d_topic in self._retained_messages:
            self.logger.debug("matching : %s %s" % (d_topic, subscription[0]))
            if self.matches(d_topic, subscription[0]):
                self.logger.debug("%s and %s match" % (d_topic, subscription[0]))
                retained = self._retained_messages[d_topic]
                publish_tasks.append(asyncio.Task(handler.mqtt_publish(retained.topic, retained.data, subscription[1], True), loop=self._loop))
        if publish_tasks:
            await asyncio.wait(publish_tasks, loop=self._loop)
        self.logger.debug("End broadcasting messages retained due to subscription on '%s' from %s" % (subscription[0], format_client_message(session=session)))

    def delete_session(self, client_id):
        try:
            session = self._sessions[client_id][0]
        except KeyError:
            session = None
        if session is None:
            self.logger.debug("Delete session : session %s doesn't exist" % client_id)
            return

        self.logger.debug("deleting session %s subscriptions" % repr(session))
        self._del_all_subscriptions(session)

        self.logger.debug("deleting existing session %s" % repr(self._sessions[client_id]))
        del self._sessions[client_id]

    def _get_handler(self, session):
        client_id = session.client_id
        if client_id:
            try:
                return self._sessions[client_id][1]
            except KeyError:
                pass
        return None
