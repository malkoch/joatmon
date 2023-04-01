import json
import threading
import time
from threading import Thread

from joatmon import context

__all__ = ['producer', 'consumer']

from joatmon.event import Event


def producer(kafka, topic):
    def _decorator(func):
        p = context.get_value(kafka).get_producer(topic)

        def _wrapper(*args, **kwargs):
            message = json.dumps({'args': args, 'kwargs': kwargs}).encode('utf-8')
            p.produce(message)
            return func(*args, **kwargs)

        return _wrapper

    return _decorator


consumers = {}
consumer_threads = {}
consumer_events = {}


def loop(topic, cons):
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
    if topic not in consumers:
        consumers[topic] = c
        consumer_events[topic] = Event()


def consumer(kafka, topic):
    def _decorator(func):
        c = context.get_value(kafka).get_consumer(topic)

        add_consumer(topic, c)
        consumer_events[topic] += func

        return func

    return _decorator


Thread(target=consumer_loop_creator).start()
