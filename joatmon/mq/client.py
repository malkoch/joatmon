import asyncio
import copy
import logging
import ssl
from collections import deque
from functools import wraps
from urllib.parse import (
    urlparse,
    urlunparse
)

import websockets
from websockets.exceptions import InvalidHandshake
from websockets.uri import InvalidURI

from .adapters import (
    StreamReaderAdapter,
    StreamWriterAdapter,
    WebSocketsReader,
    WebSocketsWriter
)
from .mqtt.constants import (
    QOS_0,
    QOS_1,
    QOS_2
)
from .mqtt.handler import (
    ClientProtocolHandler,
    ProtocolHandlerException
)
from .mqtt.packet import CONNECTION_ACCEPTED
from .session import Session
from .utils import not_in_dict_or_none

_defaults = {
    'keep_alive': 10,
    'ping_delay': 1,
    'default_qos': 0,
    'default_retain': False,
    'auto_reconnect': True,
    'reconnect_max_interval': 10,
    'reconnect_retries': 2,
}


class ClientException(Exception):
    pass


class ConnectException(ClientException):
    pass


base_logger = logging.getLogger(__name__)


def mqtt_connected(func):
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        if not self._connected_state.is_set():
            base_logger.warning("Client not connected, waiting for it")
            _, pending = await asyncio.wait([self._connected_state.wait(), self._no_more_connections.wait()], return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            if self._no_more_connections.is_set():
                raise ClientException("Will not reconnect")
        return await func(self, *args, **kwargs)

    return wrapper


class MQTTClient:
    def __init__(self, client_id=None, config=None, loop=None):
        self.logger = logging.getLogger(__name__)
        self.config = copy.deepcopy(_defaults)
        if config is not None:
            self.config.update(config)
        if client_id is not None:
            self.client_id = client_id
        else:
            from .utils import gen_client_id
            self.client_id = gen_client_id()
            self.logger.debug("Using generated client ID : %s" % self.client_id)

        if loop is not None:
            self._loop = loop
        else:
            self._loop = asyncio.get_event_loop()
        self.session = None
        self._handler = None
        self._disconnect_task = None
        self._connected_state = asyncio.Event(loop=self._loop)
        self._no_more_connections = asyncio.Event(loop=self._loop)
        self.extra_headers = {}

        self.client_tasks = deque()

    async def connect(self, uri=None, cleansession=None, cafile=None, capath=None, cadata=None, extra_headers={}):
        self.session = self._initsession(uri, cleansession, cafile, capath, cadata)
        self.extra_headers = extra_headers
        self.logger.debug("Connect to: %s" % uri)

        try:
            return await self._do_connect()
        except BaseException as be:
            self.logger.warning("Connection failed: %r" % be)
            auto_reconnect = self.config.get('auto_reconnect', False)
            if not auto_reconnect:
                raise
            else:
                return await self.reconnect()

    async def disconnect(self):
        await self.cancel_tasks()
        if self.session.transitions.is_connected():
            if not self._disconnect_task.done():
                self._disconnect_task.cancel()
            await self._handler.mqtt_disconnect()
            self._connected_state.clear()
            await self._handler.stop()
            self.session.transitions.disconnect()
        else:
            self.logger.warning("Client session is not currently connected, ignoring call")

    async def cancel_tasks(self):
        try:
            while self.client_tasks:
                task = self.client_tasks.pop()
                task.cancel()
        except IndexError as err:
            pass

    async def reconnect(self, cleansession=None):
        if self.session.transitions.is_connected():
            self.logger.warning("Client already connected")
            return CONNECTION_ACCEPTED

        if cleansession:
            self.session.clean_session = cleansession
        self.logger.debug("Reconnecting with session parameters: %s" % self.session)
        reconnect_max_interval = self.config.get('reconnect_max_interval', 10)
        reconnect_retries = self.config.get('reconnect_retries', 5)
        nb_attempt = 1
        await asyncio.sleep(1, loop=self._loop)
        while True:
            try:
                self.logger.debug("Reconnect attempt %d ..." % nb_attempt)
                return await self._do_connect()
            except BaseException as e:
                self.logger.warning("Reconnection attempt failed: %r" % e)
                if 0 <= reconnect_retries < nb_attempt:
                    self.logger.error("Maximum number of connection attempts reached. Reconnection aborted")
                    raise ConnectException("Too many connection attempts failed")
                exp = 2 ** nb_attempt
                delay = exp if exp < reconnect_max_interval else reconnect_max_interval
                self.logger.debug("Waiting %d second before next attempt" % delay)
                await asyncio.sleep(delay, loop=self._loop)
                nb_attempt += 1

    async def _do_connect(self):
        return_code = await self._connect_coro()
        self._disconnect_task = asyncio.ensure_future(self.handle_connection_close(), loop=self._loop)
        return return_code

    @mqtt_connected
    async def ping(self):
        if self.session.transitions.is_connected():
            await self._handler.mqtt_ping()
        else:
            self.logger.warning("MQTT PING request incompatible with current session state '%s'" % self.session.transitions.state)

    @mqtt_connected
    async def publish(self, topic, message, qos=None, retain=None, ack_timeout=None):
        def get_retain_and_qos():
            if qos:
                assert qos in (QOS_0, QOS_1, QOS_2)
                _qos = qos
            else:
                _qos = self.config['default_qos']
                try:
                    _qos = self.config['topics'][topic]['qos']
                except KeyError:
                    pass
            if retain:
                _retain = retain
            else:
                _retain = self.config['default_retain']
                try:
                    _retain = self.config['topics'][topic]['retain']
                except KeyError:
                    pass
            return _qos, _retain

        (app_qos, app_retain) = get_retain_and_qos()
        return await self._handler.mqtt_publish(topic, message, app_qos, app_retain, ack_timeout)

    @mqtt_connected
    async def subscribe(self, topics):
        return await self._handler.mqtt_subscribe(topics, self.session.next_packet_id)

    @mqtt_connected
    async def unsubscribe(self, topics):
        await self._handler.mqtt_unsubscribe(topics, self.session.next_packet_id)

    async def deliver_message(self, timeout=None):
        deliver_task = asyncio.ensure_future(self._handler.mqtt_deliver_next_message(), loop=self._loop)
        self.client_tasks.append(deliver_task)
        self.logger.debug("Waiting message delivery")
        done, pending = await asyncio.wait([deliver_task], loop=self._loop, return_when=asyncio.FIRST_EXCEPTION, timeout=timeout)
        if self.client_tasks:
            self.client_tasks.pop()
        if deliver_task in done:
            if deliver_task.exception() is not None:
                raise deliver_task.exception()
            return deliver_task.result()
        else:
            deliver_task.cancel()
            raise asyncio.TimeoutError

    async def _connect_coro(self):
        kwargs = dict()

        uri_attributes = urlparse(self.session.broker_uri)
        scheme = uri_attributes.scheme
        secure = True if scheme in ('mqtts', 'wss') else False
        self.session.username = self.session.username if self.session.username else uri_attributes.username
        self.session.password = self.session.password if self.session.password else uri_attributes.password
        self.session.remote_address = uri_attributes.hostname
        self.session.remote_port = uri_attributes.port
        if scheme in ('mqtt', 'mqtts') and not self.session.remote_port:
            self.session.remote_port = 8883 if scheme == 'mqtts' else 1883
        if scheme in ('ws', 'wss') and not self.session.remote_port:
            self.session.remote_port = 443 if scheme == 'wss' else 80
        if scheme in ('ws', 'wss'):
            uri = (scheme, self.session.remote_address + ":" + str(self.session.remote_port), uri_attributes[2], uri_attributes[3], uri_attributes[4], uri_attributes[5])
            self.session.broker_uri = urlunparse(uri)
        self._handler = ClientProtocolHandler(loop=self._loop)

        if secure:
            sc = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=self.session.cafile, capath=self.session.capath, cadata=self.session.cadata)
            if 'certfile' in self.config and 'keyfile' in self.config:
                sc.load_cert_chain(self.config['certfile'], self.config['keyfile'])
            if 'check_hostname' in self.config and isinstance(self.config['check_hostname'], bool):
                sc.check_hostname = self.config['check_hostname']
            kwargs['ssl'] = sc

        try:
            reader = None
            writer = None
            self._connected_state.clear()

            if scheme in ('mqtt', 'mqtts'):
                conn_reader, conn_writer = await asyncio.open_connection(self.session.remote_address, self.session.remote_port, loop=self._loop, **kwargs)
                reader = StreamReaderAdapter(conn_reader)
                writer = StreamWriterAdapter(conn_writer)
            elif scheme in ('ws', 'wss'):
                websocket = await websockets.connect(self.session.broker_uri, subprotocols=['mqtt'], loop=self._loop, extra_headers=self.extra_headers, **kwargs)
                reader = WebSocketsReader(websocket)
                writer = WebSocketsWriter(websocket)

            self._handler.attach(self.session, reader, writer)
            return_code = await self._handler.mqtt_connect()
            if return_code is not CONNECTION_ACCEPTED:
                self.session.transitions.disconnect()
                self.logger.warning("Connection rejected with code '%s'" % return_code)
                exc = ConnectException("Connection rejected by broker")
                exc.return_code = return_code
                raise exc
            else:
                await self._handler.start()
                self.session.transitions.connect()
                self._connected_state.set()
                self.logger.debug("connected to %s:%s" % (self.session.remote_address, self.session.remote_port))
            return return_code
        except InvalidURI as iuri:
            self.logger.warning("connection failed: invalid URI '%s'" % self.session.broker_uri)
            self.session.transitions.disconnect()
            raise ConnectException("connection failed: invalid URI '%s'" % self.session.broker_uri, iuri)
        except InvalidHandshake as ihs:
            self.logger.warning("connection failed: invalid websocket handshake")
            self.session.transitions.disconnect()
            raise ConnectException("connection failed: invalid websocket handshake", ihs)
        except (ProtocolHandlerException, ConnectionError, OSError) as e:
            self.logger.warning("MQTT connection failed: %r" % e)
            self.session.transitions.disconnect()
            raise ConnectException(e)

    async def handle_connection_close(self):

        def cancel_tasks():
            self._no_more_connections.set()
            while self.client_tasks:
                task = self.client_tasks.popleft()
                if not task.done():
                    task.cancel()

        self.logger.debug("Watch broker disconnection")
        await self._handler.wait_disconnect()
        self.logger.warning("Disconnected from broker")

        self._connected_state.clear()

        self._handler.detach()
        self.session.transitions.disconnect()

        if self.config.get('auto_reconnect', False):
            self.logger.debug("Auto-reconnecting")
            try:
                await self.reconnect()
            except ConnectException:
                cancel_tasks()
        else:
            cancel_tasks()

    def _initsession(self, uri=None, cleansession=None, cafile=None, capath=None, cadata=None) -> Session:
        broker_conf = self.config.get('broker', dict()).copy()
        if uri:
            broker_conf['uri'] = uri
        if cafile:
            broker_conf['cafile'] = cafile
        elif 'cafile' not in broker_conf:
            broker_conf['cafile'] = None
        if capath:
            broker_conf['capath'] = capath
        elif 'capath' not in broker_conf:
            broker_conf['capath'] = None
        if cadata:
            broker_conf['cadata'] = cadata
        elif 'cadata' not in broker_conf:
            broker_conf['cadata'] = None

        if cleansession is not None:
            broker_conf['cleansession'] = cleansession

        for key in ['uri']:
            if not_in_dict_or_none(broker_conf, key):
                raise ClientException("Missing connection parameter '%s'" % key)

        s = Session()
        s.broker_uri = uri
        s.client_id = self.client_id
        s.cafile = broker_conf['cafile']
        s.capath = broker_conf['capath']
        s.cadata = broker_conf['cadata']
        if cleansession is not None:
            s.clean_session = cleansession
        else:
            s.clean_session = self.config.get('cleansession', True)
        s.keep_alive = self.config['keep_alive'] - self.config['ping_delay']
        if 'will' in self.config:
            s.will_flag = True
            s.will_retain = self.config['will']['retain']
            s.will_topic = self.config['will']['topic']
            s.will_message = self.config['will']['message']
            s.will_qos = self.config['will']['qos']
        else:
            s.will_flag = False
            s.will_retain = False
            s.will_topic = None
            s.will_message = None
        print(s.username, s.password, s.client_id)
        return s
