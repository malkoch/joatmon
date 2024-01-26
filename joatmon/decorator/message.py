import json
import threading
import time
from threading import Thread

from joatmon.core import context
from joatmon.core.event import Event
from joatmon.core.utility import JSONEncoder

__all__ = ['producer', 'consumer']


def producer(plugin, topic):
    """
    Decorator for handling exceptions in a function.

    This decorator wraps the function in a try-except block. If the function raises an exception of type `ex`, the exception is caught and its message is printed. The function then returns None.

    Args:
        ex (Exception, optional): The type of exception to catch. If None, all exceptions are caught. Defaults to None.

    Returns:
        function: The decorated function.
    """

    def _decorator(func):
        def _wrapper(*args, **kwargs):
            p = context.get_value(plugin).get_producer(topic)
            message = json.dumps({'args': args, 'kwargs': kwargs}, cls=JSONEncoder)
            p.produce(topic, message)
            return func(*args, **kwargs)

        return _wrapper

    return _decorator


consumers = {}
consumer_threads = {}
consumer_events = {}


def loop(topic, cons):
    """
    Function for consuming messages from a topic in a loop.

    This function retrieves a consumer from the context and uses it to consume messages from a specified topic in a loop. When a message is consumed, it is printed and an event is fired with the arguments and keyword arguments from the message.

    Args:
        topic (str): The topic to consume messages from.
        cons (Consumer): The consumer to use.
    """

    while threading.main_thread().is_alive():
        msg = cons.consume()
        if msg is None:
            continue

        packet = json.loads(msg)
        args = packet['args']
        kwargs = packet['kwargs']

        try:
            consumer_events[topic].fire(*args, **kwargs)
        except Exception as ex:
            print(str(ex))


def consumer_loop_creator():
    """
    Function for creating consumer loops.

    This function creates a consumer loop for each consumer in the context. If a consumer loop for a consumer already exists, it is not created again.
    """
    global consumer_threads
    while threading.main_thread().is_alive():
        consumer_threads = {topic: thread for topic, thread in consumer_threads.items() if thread.is_alive()}

        for topic, cons in consumers.items():
            if topic not in consumer_threads:
                thread = Thread(target=loop, args=(topic, cons))
                consumer_threads[topic] = thread
                thread.start()
        time.sleep(0.1)


def add_consumer(topic, c):
    """
    Function for adding a consumer to the context.

    This function adds a consumer to the context and creates an event for it.

    Args:
        topic (str): The topic the consumer consumes messages from.
        c (Consumer): The consumer to add.
    """
    if topic not in consumers:
        consumers[topic] = c
        consumer_events[topic] = Event()


def consumer(plugin, topic):
    """
    Decorator for consuming messages from a topic.

    This decorator retrieves a consumer from the context and uses it to consume messages from a specified topic. When a message is consumed, an event is fired with the arguments and keyword arguments from the message.

    Args:
        plugin (str): The name of the plugin in the context.
        topic (str): The topic to consume messages from.

    Returns:
        function: The decorated function.
    """

    def _decorator(func):
        c = context.get_value(plugin).get_consumer(topic)

        add_consumer(topic, c)
        consumer_events[topic] += func

        return func

    return _decorator


Thread(target=consumer_loop_creator).start()
