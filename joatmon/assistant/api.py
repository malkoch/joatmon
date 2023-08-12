import datetime
import importlib.util
import json
import os
import threading
import time

import openai

from joatmon import context
from joatmon.assistant import (
    service,
    task
)
from joatmon.assistant.intent import GenericAssistant
from joatmon.assistant.service import (
    ServiceInfo,
    ServiceState
)
from joatmon.assistant.stt import STTAgent
from joatmon.assistant.task import (
    BaseTask,
    TaskInfo,
    TaskState
)
from joatmon.assistant.tts import TTSAgent
from joatmon.system.hid.console import (
    ConsoleReader,
    ConsoleWriter
)
from joatmon.system.hid.microphone import Microphone
from joatmon.system.hid.speaker import Speaker
from joatmon.system.lock import RWLock
from joatmon.utility import get_module_classes


class CTX:
    ...


ctx = CTX()
context.set_ctx(ctx)


class API:
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self):
        settings = json.loads(open('iva/iva.json', 'r').read())

        openai.api_key = settings['config']['openai']['key']

        self.tts = settings.get('config', {}).get('tts', False)
        self.stt = settings.get('config', {}).get('stt', False)

        if self.tts:
            self.tts_agent = TTSAgent()
            self.speaker = Speaker()
        else:
            self.writer = ConsoleWriter()

        if self.stt:
            self.stt_agent = STTAgent()
            self.microphone = Microphone()
        else:
            self.reader = ConsoleReader()

        self.output('input and output devices are initialized')

        self.assistant = GenericAssistant('iva/iva.json')
        self.assistant.train_model()
        self.assistant.save_model('iva')

        self.output('intent assistant is initialized')

        self.parent_os_path = os.path.abspath(os.path.curdir)
        self.os_path = os.sep

        self.lock = RWLock()
        self.running_tasks = {}  # running, finished
        self.running_services = {}  # running, enabled, disabled, stopped, finished

        self.event = threading.Event()

        self.output('running startup tasks')
        tasks = settings.get('tasks', [])
        for _task in sorted(filter(lambda x: x['status'] and x['on'] == 'startup', tasks), key=lambda x: x['priority']):
            self.run_task(_task['name'])  # need to do them in background

        self.output('starting automatic services')
        services = settings.get('services', [])
        for _service in sorted(
                filter(lambda x: x['status'] and x['mode'] == 'automatic', services), key=lambda x: x['priority']
        ):
            self.start_service(_service['name'])  # need to do them in background

        self.output('creating cleanup thread')
        self.cleaning_thread = threading.Thread(target=self.clean)
        self.cleaning_thread.start()
        time.sleep(1)

        self.output('starting service runner thread')
        self.service_thread = threading.Thread(target=self.run_services)
        self.service_thread.start()
        time.sleep(1)

        self.output('creating task scheduler thread')
        self.interval_thread = threading.Thread(target=self.run_interval)
        self.interval_thread.start()
        time.sleep(1)

        # make it async
        # need event viewer

    def run_interval(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        while not self.event.is_set():
            settings = json.loads(open('iva/iva.json', 'r').read())
            tasks = settings.get('tasks', [])
            tasks = filter(lambda x: x['status'], tasks)
            tasks = filter(lambda x: x['on'] == 'interval', tasks)
            tasks = filter(lambda x: x['interval'] > 0, tasks)
            tasks = list(tasks)

            new_tasks = filter(lambda x: x.get('last_run_time', None) is None, tasks)
            new_tasks = list(new_tasks)
            old_tasks = filter(lambda x: x.get('next_run_time', None) is not None, tasks)
            old_tasks = filter(
                lambda x: datetime.datetime.now() > datetime.datetime.fromisoformat(x['next_run_time']), old_tasks
            )
            old_tasks = list(old_tasks)

            for _task in sorted(new_tasks, key=lambda x: x['priority']):
                self.run_task(task_name=_task['name'], kwargs=None, background=True)  # need to do them in background
            for _task in sorted(old_tasks, key=lambda x: x['priority']):
                self.run_task(task_name=_task['name'], kwargs=None, background=True)  # need to do them in background

            # need to run to do as well
            # need to run agenda as well

            time.sleep(1)

    def run_services(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        settings = json.loads(open('iva/iva.json', 'r').read())

        services = settings.get('services', [])

        # if the service is closed for some reason and is configured as restart automatically, need to restart the service

        while not self.event.is_set():
            for _service in sorted(filter(lambda x: x['status'], services), key=lambda x: x['priority']):
                if (
                        _service['name'] not in self.running_services
                        or self.running_services[_service['name']].state == ServiceState.finished
                ):
                    self.start_service(_service['name'])  # need to do them in background
            time.sleep(1)

    def clean(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
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

    def listen_intent(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        response = self.input('what is your intent')
        intent, prob = self.intent(response)
        if prob > 0.9:
            return intent
        else:
            return response

    def listen_command(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        response = self.input('what is your command')

        tasks = []

        settings = json.loads(open('iva/iva.json', 'r').read())
        for scripts_folder in settings.get('scripts', []):
            if os.path.isabs(scripts_folder):
                for script_file in os.listdir(scripts_folder):
                    spec = importlib.util.spec_from_file_location(
                        scripts_folder, os.path.join(scripts_folder, script_file)
                    )
                    action_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(action_module)

                    for class_ in get_module_classes(action_module):
                        if not issubclass(class_[1], BaseTask) or class_[1] is BaseTask:
                            continue

                        tasks.append(class_[1])
            else:
                try:
                    _module = __import__('.'.join(scripts_folder.split('.')), fromlist=[scripts_folder.split('.')[-1]])

                    scripts_folder = _module.__path__[0]

                    for script_file in os.listdir(scripts_folder):
                        if '__' in script_file:
                            continue

                        spec = importlib.util.spec_from_file_location(
                            scripts_folder, os.path.join(scripts_folder, script_file)
                        )
                        action_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(action_module)

                        for class_ in get_module_classes(action_module):
                            if not issubclass(class_[1], BaseTask) or class_[1] is BaseTask:
                                continue

                            tasks.append(class_[1])
                except ModuleNotFoundError:
                    print('module not found')
                    continue

        functions = list(map(lambda x: x.help(), tasks))
        functions = list(filter(lambda x: x, functions))

        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-0613', messages=[{'role': 'user', 'content': response}], functions=functions
        )

        result = response['choices'][0]
        if result['finish_reason'] == 'function_call':
            message = result['message']

            return message['function_call']['name'], json.loads(message['function_call']['arguments'])

    def input(self, prompt=None):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        # put the input to the queue
        # if the task is called from outside, putting inputs to the queue by hand will make it work

        # if not english, translate to english
        # then get the response
        # get the intent
        # run the action
        # translate the response if needed

        if prompt:
            self.output(prompt)

        if self.stt:
            response = self.microphone.listen()
            response = self.stt_agent.transcribe(response)
        else:
            response = self.reader.read()
        return response

    def output(self, text):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if self.tts:
            text = self.tts_agent.convert(text)
            self.speaker.say(text)
        else:
            self.writer.write(text)

    def intent(self, text):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return self.assistant.request(text)

    def do_action(self, line):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
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
                    command, arguments = self.listen_command()
                    return self.run_task(task_name=command, kwargs=arguments)
                case _:
                    raise ValueError(f'wanted command is {line}, default case is not implemented')
        except Exception as ex:
            print(str(ex))  # use stacktrace and write all exception details, line number, function name, file name etc.
            # return self.exit()

    def run_task(self, task_name, kwargs=None, background=False):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        _task = task.get(self, task_name, kwargs, background)
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
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
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
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.running_services[service_name].state = ServiceState.stopped
        self.running_services[service_name].service.stop()
        return False

    def restart_service(self, service_name):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.stop_service(service_name)
        self.start_service(service_name)
        return False

    def exit(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.output('shutting down the system')

        settings = json.loads(open('iva/iva.json', 'r').read())

        self.output('running shutdown tasks')
        tasks = settings.get('tasks', [])
        for _task in sorted(
                filter(lambda x: x['status'] and x['on'] == 'shutdown', tasks), key=lambda x: x['priority']
        ):
            self.run_task(_task['name'])

        self.output('stopping background tasks, jobs and services')
        with self.lock.r_locked():
            task_keys = [key for key in self.running_tasks.keys()]
            service_keys = [key for key in self.running_services.keys()]

        for key in task_keys:
            task_info = self.running_tasks[key]
            task_info.task.stop()
        for key in service_keys:
            self.stop_service(key)

        self.event.set()
        self.output('closing input device')
        if self.stt:
            self.microphone.stop()
        self.output('shutdown')
        return True

    def mainloop(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        while not self.event.is_set():
            command = self.listen_intent()
            self.do_action(command)


def main():
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    api = API()
    api.mainloop()
