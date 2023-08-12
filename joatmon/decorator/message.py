import json
import threading
import time
from threading import Thread

from joatmon import context
from joatmon.event import Event
from joatmon.utility import JSONEncoder

__all__ = ['producer', 'consumer']


def producer(kafka, topic):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """

    def _decorator(func):
        def _wrapper(*args, **kwargs):
            p = context.get_value(kafka).get_producer(topic)
            message = json.dumps({'args': args, 'kwargs': kwargs}, cls=JSONEncoder).encode('utf-8')
            p.produce(message)
            return func(*args, **kwargs)

        return _wrapper

    return _decorator


consumers = {}
consumer_threads = {}
consumer_events = {}


def loop(topic, cons):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    for message in cons:
        time.sleep(0.1)

        if not threading.main_thread().is_alive():
            break

        if message is None:
            continue

        cons.commit_offsets()

        packet = json.loads(message.value.decode('utf-8'))
        args = packet['args']
        kwargs = packet['kwargs']

        consumer_events[topic].fire(*args, **kwargs)


def consumer_loop_creator():
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
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
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    if topic not in consumers:
        consumers[topic] = c
        consumer_events[topic] = Event()


# need to have another parameter called is_batch or batch_size
# if False or None consumer will be working with one message at a time
# if not False and None consumer will be working in batch mode
def consumer(kafka, topic):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """

    def _decorator(func):
        c = context.get_value(kafka).get_consumer(topic)

        add_consumer(topic, c)
        consumer_events[topic] += func

        return func

    return _decorator


Thread(target=consumer_loop_creator).start()
