import json
import os
import queue
import threading
import time

from joatmon import context
from joatmon.assistant import (
    service,
    task
)
from joatmon.assistant.intents import GenericAssistant
from joatmon.assistant.service import (
    ServiceInfo,
    ServiceState
)
from joatmon.assistant.task import (
    TaskInfo,
    TaskState
)
from joatmon.hid.microphone import InputDriver
from joatmon.hid.speaker import OutputDevice
from joatmon.system.lock import RWLock


class CTX:
    ...


ctx = CTX()
context.set_ctx(ctx)


class API:
    def __init__(self):
        settings = json.loads(open('iva.json', 'r').read())

        self.work_queue = queue.Queue()
        self.work_stack = queue.Queue()

        self.assistant = GenericAssistant('iva.json', model_name="iva")
        self.assistant.train_model()
        self.assistant.save_model('iva')

        self.parent_os_path = os.path.abspath(os.path.curdir)
        self.os_path = os.sep

        self.output_device = OutputDevice()
        self.input_device = InputDriver(self.output_device)

        self.lock = RWLock()
        self.running_tasks = {}  # running, finished
        self.running_services = {}  # running, enabled, disabled, stopped, finished

        self.event = threading.Event()

        tasks = settings.get('tasks', [])
        for _task in sorted(filter(lambda x: x['status'] and x['on'] == 'startup', tasks), key=lambda x: x['priority']):
            self.run_task(_task['name'])  # need to do them in background
        services = settings.get('services', [])
        for _service in sorted(filter(lambda x: x['status'] and x['mode'] == 'automatic', services), key=lambda x: x['priority']):
            self.start_service(_service['name'])  # need to do them in background

        self.cleaning_thread = threading.Thread(target=self.clean)
        self.cleaning_thread.start()
        self.service_thread = threading.Thread(target=self.run_services)
        self.service_thread.start()

        # self.do_action('ls .')
        # self.do_action('dt')

    def run_services(self):
        settings = json.loads(open('iva.json', 'r').read())

        services = settings.get('services', [])

        # if the service is closed for some reason and is configured as restart automatically, need to restart the service

        while not self.event.is_set():
            for _service in sorted(filter(lambda x: x['status'], services), key=lambda x: x['priority']):
                if _service['name'] not in self.running_services or self.running_services[_service['name']].state == ServiceState.finished:
                    self.start_service(_service['name'])  # need to do them in background
            time.sleep(1)

    def consumer(self):
        while not self.event.is_set() and self.work_stack.qsize():
            time.sleep(0.01)
        while not self.event.is_set() and self.work_queue.qsize():
            time.sleep(0.01)

    def clean(self):
        while not self.event.is_set():
            with self.lock.r_locked():
                task_keys = [key for key in self.running_tasks.keys()]
                service_keys = [key for key in self.running_services.keys()]

            delete_task_keys = []
            for key in task_keys:
                task_info = self.running_tasks[key]
                if not task_info.task.running():
                    delete_task_keys.append(key)
            delete_service_keys = []
            for key in service_keys:
                task_info = self.running_services[key]
                if not task_info.service.running() and task_info.state != ServiceState.stopped:
                    delete_service_keys.append(key)

            for key in delete_task_keys:
                with self.lock.w_locked():
                    del self.running_tasks[key]
            for key in delete_service_keys:
                with self.lock.w_locked():
                    del self.running_services[key]

            time.sleep(1)

    def listen(self, prompt=None):
        # put the input to the queue
        # if the task is called from outside, putting inputs to the queue by hand will make it work

        response = self.input_device.listen(prompt)
        intent, prob = self.intent(response)
        if prob > .9:
            return intent
        else:
            return response

    def say(self, text):
        self.output_device.say(text)

    def intent(self, text):
        return self.assistant.request(text)

    def do_action(self, line):
        try:
            if line is None or line == '':
                return False

            match line.lower():
                case 'enable':
                    return False
                case 'disable':
                    return False
                case 'update':
                    return False
                case 'delete':
                    return False
                case 'configure':
                    return False
                case 'start':
                    return False
                case 'stop':
                    return False
                case 'restart':
                    return False
                case 'skip':
                    return False
                case 'help':
                    return False
                case 'exit':
                    return self.exit()
                case 'activate':
                    action = self.listen('yes sir')
                    return self.run_task(action)
                case _:
                    return self.run_task(line)
        except Exception as ex:
            print(str(ex))  # use stacktrace and write all exception details, line number, function name, file name etc.
            # return self.exit()

    def run_task(self, task_name, kwargs=None):
        _task = task.get(self, task_name, kwargs)
        if _task is None:
            return False

        if _task.background:
            if task_name not in self.running_tasks:
                self.running_tasks[task_name] = TaskInfo(task_name, TaskState.running, _task)
            else:
                self.running_tasks[task_name].state = TaskState.running
                self.running_tasks[task_name].task = _task
        _task.start()

        return False

    def start_service(self, service_name):
        _task = service.get(self, service_name)
        if _task is None:
            return False

        if service_name not in self.running_services:
            self.running_services[service_name] = ServiceInfo(service_name, ServiceState.running, _task)
        else:
            self.running_services[service_name].state = ServiceState.running
            self.running_services[service_name].service = _task
        _task.start()

        return False

    def stop_service(self, service_name):
        self.running_services[service_name].state = ServiceState.stopped
        self.running_services[service_name].service.stop()
        return False

    def restart_service(self, service_name):
        self.stop_service(service_name)
        self.start_service(service_name)
        return False

    def exit(self):
        settings = json.loads(open('iva.json', 'r').read())

        tasks = settings.get('tasks', [])
        for _task in sorted(filter(lambda x: x['status'] and x['on'] == 'shutdown', tasks), key=lambda x: x['priority']):
            self.run_task(_task['name'])

        with self.lock.r_locked():
            task_keys = [key for key in self.running_tasks.keys()]
            service_keys = [key for key in self.running_services.keys()]

        for key in task_keys:
            task_info = self.running_tasks[key]
            task_info.task.stop()
        for key in service_keys:
            self.stop_service(key)

        self.event.set()
        self.input_device.stop()
        return True

    def mainloop(self):
        while not self.event.is_set():
            command = self.listen()
            self.do_action(command)


def main():
    api = API()
    api.mainloop()
