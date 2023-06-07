import argparse
import dataclasses
import enum
import importlib.util
import inspect
import json
import os
import re
import threading
import time
from cmd import Cmd

import colorama
import schedule
from colorama import Fore

from joatmon import context
from joatmon.assistant.job import BaseJob
from joatmon.assistant.service import BaseService
from joatmon.assistant.task import BaseTask
from joatmon.hid.microphone import InputDriver
from joatmon.hid.speaker import OutputDevice
from joatmon.system.lock import RWLock
from joatmon.utility import first


class CTX:
    ...


ctx = CTX()
context.set_ctx(ctx)

PROMPT_CHAR = '~>'
COMMA_MATCHER = re.compile(r" (?=(?:[^\"']*[\"'][^\"']*[\"'])*[^\"']*$)")


class TaskState(enum.Enum):
    running = enum.auto()
    finished = enum.auto()


@dataclasses.dataclass
class TaskInfo:
    name: str
    state: TaskState
    task: BaseTask


class JobState(enum.Enum):
    running = enum.auto()
    finished = enum.auto()


@dataclasses.dataclass
class JobInfo:
    name: str
    state: JobState
    job: BaseJob


class ServiceState(enum.Enum):
    running = enum.auto()
    finished = enum.auto()
    stopped = enum.auto()


@dataclasses.dataclass
class ServiceInfo:
    name: str
    state: ServiceState
    service: BaseService


class Interpreter(Cmd):
    first_reaction_text = ""
    # first_reaction_text += Fore.BLUE + 'IVA\'s sound is by default disabled.' + Fore.RESET
    # first_reaction_text += "\n"
    # first_reaction_text += Fore.BLUE + 'In order to let IVA talk out loud type: '
    # first_reaction_text += Fore.RESET + Fore.RED + 'enable sound' + Fore.RESET
    # first_reaction_text += "\n"
    first_reaction_text += Fore.BLUE + "Type 'help' for a list of available actions." + Fore.RESET
    first_reaction_text += "\n"

    prompt = first_reaction_text + Fore.RED + "{} What can I do for you?: ".format(PROMPT_CHAR) + Fore.RESET

    def __init__(self):
        self.parent_os_path = os.path.abspath(os.path.curdir)
        self.os_path = os.sep

        self._tts = False
        self._stt = False

        settings = json.loads(open('iva.json', 'r').read())

        self._tts = settings.get('tts', False)
        self._stt = settings.get('stt', False)

        self.output_device = OutputDevice(tts_enabled=self.tts_enabled)
        self.input_device = InputDriver(stt_enabled=self.stt_enabled)

        super(Interpreter, self).__init__(stdin=self.input_device, stdout=self.output_device)

        if self.tts_enabled:
            first_reaction_text = ""
            first_reaction_text += "Type 'help' for a list of available actions."
            first_reaction_text += "\n"
            self.intro = first_reaction_text
        else:
            first_reaction_text = ""
            first_reaction_text += Fore.BLUE + "Type 'help' for a list of available actions." + Fore.RESET
            first_reaction_text += "\n"
            self.intro = first_reaction_text

        if self.tts_enabled:
            self.prompt = "What can I do for you?: "
        else:
            self.prompt = Fore.RED + "{} What can I do for you?: ".format(PROMPT_CHAR) + Fore.RESET

        self.use_rawinput = False

        self.lock = RWLock()
        self.running_tasks = {}  # running, finished
        self.running_jobs = {}  # running, enabled, disabled, finished
        self.running_services = {}  # running, enabled, disabled, stopped, finished

        self.event = threading.Event()

        settings = json.loads(open('iva.json', 'r').read())

        tasks = settings.get('tasks', [])
        for task in sorted(filter(lambda x: x['status'] and x['on'] == 'startup', tasks), key=lambda x: x['priority']):
            self.run_task(task['name'])  # need to do them in background
        services = settings.get('services', [])
        for service in sorted(filter(lambda x: x['status'] and x['mode'] == 'automatic', services), key=lambda x: x['priority']):
            self.start_service(service['name'])  # need to do them in background

        self.cleaning_thread = threading.Thread(target=self.clean)
        self.cleaning_thread.start()
        self.job_thread = threading.Thread(target=self.run_jobs)
        self.job_thread.start()
        self.service_thread = threading.Thread(target=self.run_services)
        self.service_thread.start()

    @property
    def tts_enabled(self):
        return self._tts

    @tts_enabled.setter
    def tts_enabled(self, value):
        self._tts = value
        self.output_device.tts_enabled = value

    @property
    def stt_enabled(self):
        return self._stt

    @stt_enabled.setter
    def stt_enabled(self, value):
        self._stt = value
        self.input_device.stt_enabled = value

    def run_jobs(self):
        settings = json.loads(open('iva.json', 'r').read())

        jobs = settings.get('jobs', {})
        for job in sorted(filter(lambda x: x['status'], jobs), key=lambda x: x['priority']):
            schedule.every(int(job['every'])).seconds.do(self.run_job, job['name'])  # need to do them in background

        while not self.event.is_set():
            schedule.run_pending()
            time.sleep(0.1)

        schedule.clear()

    def run_services(self):
        settings = json.loads(open('iva.json', 'r').read())

        services = settings.get('services', [])

        # if the service is closed for some reason and it is configured as restart automatically, need to restart the service

        while not self.event.is_set():
            for service in sorted(filter(lambda x: x['status'], services), key=lambda x: x['priority']):
                if service['name'] not in self.running_services or self.running_services[service['name']].state == ServiceState.finished:
                    self.start_service(service['name'])  # need to do them in background
            time.sleep(1)

    def clean(self):
        while not self.event.is_set():
            with self.lock.r_locked():
                task_keys = [key for key in self.running_tasks.keys()]
                job_keys = [key for key in self.running_jobs.keys()]
                service_keys = [key for key in self.running_services.keys()]

            delete_task_keys = []
            for key in task_keys:
                task_info = self.running_tasks[key]
                if not task_info.task.running():
                    delete_task_keys.append(key)
            delete_job_keys = []
            for key in job_keys:
                task_info = self.running_jobs[key]
                if not task_info.job.running():
                    delete_job_keys.append(key)
            delete_service_keys = []
            for key in service_keys:
                task_info = self.running_services[key]
                if not task_info.service.running() and task_info.state != ServiceState.stopped:
                    delete_service_keys.append(key)

            for key in delete_task_keys:
                with self.lock.w_locked():
                    del self.running_tasks[key]
            for key in delete_job_keys:
                with self.lock.w_locked():
                    del self.running_jobs[key]
            for key in delete_service_keys:
                with self.lock.w_locked():
                    del self.running_services[key]

            time.sleep(1)

    def input(self, force=False):
        return self.input_device.input()

    def output(self, text, force=False, order=None):
        order = order or ['terminal']

        for out_type, *extras in order:
            match out_type:
                case 'gui':
                    ...
                case 'pretty':
                    ...
                case 'terminal':
                    ...
                case 'voice':
                    ...
                case _:
                    ...

        self.output_device.output(text)

    def precmd(self, line):
        return 'action ' + line

    def do_action(self, line):
        try:
            if line is None or line == '':
                return False

            action, *args = COMMA_MATCHER.split(line)

            if action is None or action == '':
                return False

            match action.lower():
                case 'enable':
                    return self.enable()
                case 'disable':
                    return self.disable()
                case 'create':
                    return self.create_()
                case 'update':
                    return self.update()
                case 'delete':
                    return self.delete()
                case 'configure':
                    return self.configure()
                case 'run':
                    return self.run()
                case 'start':
                    return self.start()
                case 'stop':
                    return self.stop()
                case 'restart':
                    return self.restart()
                case 'help':
                    return self.help()
                case 'exit':
                    return self.exit()
                case _:
                    parser = argparse.ArgumentParser()
                    _, extras = parser.parse_known_args(args)
                    # for k, v in zip(*(iter(extras),) * 2):

                    return self.run_task(action, [k for k in extras])
        except Exception as ex:
            print(str(ex))  # use stacktrace and write all exception details, line number, function name, file name etc.
            # return self.exit()

    @staticmethod
    def _get_task(script_name):
        task = None

        settings = json.loads(open('iva.json', 'r').read())
        for scripts in settings.get('scripts', []):
            if os.path.isabs(scripts):
                if os.path.exists(scripts) and os.path.exists(os.path.join(scripts, f'{script_name}.py')):
                    spec = importlib.util.spec_from_file_location(scripts, os.path.join(scripts, f'{script_name}.py'))
                    action_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(action_module)
                else:
                    continue
            else:
                try:
                    _module = __import__(scripts, fromlist=[f'{script_name}'])
                except ModuleNotFoundError:
                    continue

                action_module = getattr(_module, script_name, None)

            if action_module is None:
                continue

            task = getattr(action_module, 'Task', None)

        return task

    @staticmethod
    def _get_job(script_name):
        task = None

        settings = json.loads(open('iva.json', 'r').read())
        for scripts in settings.get('scripts', []):
            if os.path.isabs(scripts):
                if os.path.exists(scripts) and os.path.exists(os.path.join(scripts, f'{script_name}.py')):
                    spec = importlib.util.spec_from_file_location(scripts, os.path.join(scripts, f'{script_name}.py'))
                    action_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(action_module)
                else:
                    continue
            else:
                try:
                    _module = __import__(scripts, fromlist=[f'{script_name}'])
                except ModuleNotFoundError:
                    continue

                action_module = getattr(_module, script_name, None)

            if action_module is None:
                continue

            task = getattr(action_module, 'Job', None)

        return task

    @staticmethod
    def _get_service(script_name):
        task = None

        settings = json.loads(open('iva.json', 'r').read())
        for scripts in settings.get('scripts', []):
            if os.path.isabs(scripts):
                if os.path.exists(scripts) and os.path.exists(os.path.join(scripts, f'{script_name}.py')):
                    spec = importlib.util.spec_from_file_location(scripts, os.path.join(scripts, f'{script_name}.py'))
                    action_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(action_module)
                else:
                    continue
            else:
                try:
                    _module = __import__(scripts, fromlist=[f'{script_name}'])
                except ModuleNotFoundError:
                    continue

                action_module = getattr(_module, script_name, None)

            if action_module is None:
                continue

            task = getattr(action_module, 'Service', None)

        return task

    def enable(self, *args):
        for arg in args:
            if arg == 'tts':
                self.tts_enabled = True
                self.output_device.tts_enabled = True
            if arg == 'stt':
                self.stt_enabled = True
                self.input_device.stt_enabled = True

        settings = json.loads(open('iva.json', 'r').read())
        settings['tts'] = self.tts_enabled
        settings['stt'] = self.stt_enabled
        open('iva.json', 'w').write(json.dumps(settings, indent=4))

        return False

    def disable(self, *args):
        for arg in args:
            if arg == 'tts':
                self.tts_enabled = False
                self.output_device.tts_enabled = False
            if arg == 'stt':
                self.stt_enabled = False
                self.input_device.stt_enabled = False

        settings = json.loads(open('iva.json', 'r').read())
        settings['tts'] = self.tts_enabled
        settings['stt'] = self.stt_enabled
        open('iva.json', 'w').write(json.dumps(settings, indent=4))

        return False

    def create_(self):
        self.output('what do you want me to create')
        action_type = self.input()

        if action_type.lower() == 'task':
            self.output('which script do you want me to create as task')
            script = self.input()
            return self.create_task(script)
        if action_type.lower() == 'job':
            self.output('which script do you want me to create as job')
            script = self.input()
            return self.create_job(script)
        if action_type.lower() == 'service':
            self.output('which script do you want me to create as service')
            script = self.input()
            return self.create_service(script)

        return False

    def create_task(self, script):
        task = self._get_task(script)

        if task is None:
            self.output('task is not found')
            return False

        create_args = {}
        self.output(f'what do you want to call this task')
        name = self.input()
        self.output(f'what should be the priority of this task')
        priority = self.input()
        self.output(f'when should this task run')
        on = self.input()

        create_args['name'] = name
        create_args['priority'] = int(priority)
        create_args['on'] = on
        create_args['script'] = script
        create_args['status'] = True
        create_args['args'] = task.create(self)

        settings = json.loads(open('iva.json', 'r').read())
        tasks = settings.get('tasks', [])

        tasks.append(create_args)

        settings['tasks'] = tasks
        open('iva.json', 'w').write(json.dumps(settings, indent=4))

    def create_job(self, script):
        task = self._get_job(script)

        if task is None:
            self.output('job is not found')
            return False

        create_args = {}
        self.output(f'what do you want to call this job')
        name = self.input()
        self.output(f'what should be the priority of this job')
        priority = self.input()
        self.output(f'how often should this job run')
        every = self.input()

        create_args['name'] = name
        create_args['priority'] = int(priority)
        create_args['every'] = int(every)
        create_args['script'] = script
        create_args['status'] = True
        create_args['args'] = task.create(self)

        settings = json.loads(open('iva.json', 'r').read())
        tasks = settings.get('jobs', [])

        tasks.append(create_args)

        settings['jobs'] = tasks
        open('iva.json', 'w').write(json.dumps(settings, indent=4))

    def create_service(self, script):
        task = self._get_service(script)

        if task is None:
            self.output('service is not found')
            return False

        create_args = {}
        self.output(f'what do you want to call this task')
        name = self.input()
        self.output(f'what should be the priority of this task')
        priority = self.input()
        self.output(f'when should this service run')
        mode = self.input()

        create_args['name'] = name
        create_args['priority'] = int(priority)
        create_args['mode'] = mode
        create_args['script'] = script
        create_args['status'] = True
        create_args['args'] = task.create(self)

        settings = json.loads(open('iva.json', 'r').read())
        tasks = settings.get('services', [])

        tasks.append(create_args)

        settings['services'] = tasks
        open('iva.json', 'w').write(json.dumps(settings, indent=4))

    def update(self):
        ...

    def delete(self):
        ...

    def configure(self):
        ...

    def run(self):
        self.output('what do you want me to run')
        action_type = self.input()

        if action_type.lower() == 'task':
            self.output('which task do you want me to run')
            task = self.input()
            return self.run_task(task)
        if action_type.lower() == 'job':
            self.output('which job do you want me to run')
            job = self.input()
            return self.run_job(job)

        return False

    def run_task(self, task_name, args=None):
        args = args or []

        settings = json.loads(open('iva.json', 'r').read())
        task_info = first(filter(lambda x: x['name'] == task_name, settings.get('tasks', [])))

        if task_info is None:
            task_info = {'script': task_name, 'args': {}}
            # return False

        script = task_info['script']

        task = self._get_task(script)

        if task is None:
            self.output('task is not found')
            return False

        kwargs = task_info['args']
        kwargs['parent_os_path'] = self.parent_os_path
        kwargs['os_path'] = self.os_path

        task = task(self, *args, **kwargs)
        if task_name not in self.running_tasks:
            self.running_tasks[task_name] = TaskInfo(task_name, TaskState.running, task)
        else:
            self.running_tasks[task_name].state = TaskState.running
            self.running_tasks[task_name].task = task
        task.start()

        return False

    def run_job(self, job_name):
        settings = json.loads(open('iva.json', 'r').read())
        task_info = first(filter(lambda x: x['name'] == job_name, settings.get('jobs', [])))

        if task_info is None:
            task_info = {'script': job_name, 'args': {}}
            # return False

        script = task_info['script']

        task = self._get_job(script)

        if task is None:
            self.output('job is not found')
            return False

        args = task_info['args']
        args['parent_os_path'] = self.parent_os_path
        args['os_path'] = self.os_path

        task = task(self, *[], **args)
        if job_name not in self.running_jobs:
            self.running_jobs[job_name] = JobInfo(job_name, JobState.running, task)
        else:
            self.running_jobs[job_name].state = JobState.running
            self.running_jobs[job_name].job = task
        task.start()

        return False

    def start(self):
        self.output('what service do you want me to start')
        action = self.input()
        return self.start_service(action)

    def start_service(self, service_name):
        settings = json.loads(open('iva.json', 'r').read())
        task_info = first(filter(lambda x: x['name'] == service_name, settings.get('services', [])))

        if task_info is None:
            task_info = {'script': service_name, 'args': {}}
            # return False

        script = task_info['script']

        task = self._get_service(script)

        if task is None:
            self.output('service is not found')
            return False

        args = task_info['args']
        args['parent_os_path'] = self.parent_os_path
        args['os_path'] = self.os_path

        task = task(self, *[], **args)
        if service_name not in self.running_services:
            self.running_services[service_name] = ServiceInfo(service_name, ServiceState.running, task)
        else:
            self.running_services[service_name].state = ServiceState.running
            self.running_services[service_name].service = task
        task.start()

        return False

    def stop(self):
        self.output('what service do you want me to stop')
        action = self.input()
        return self.stop_service(action)

    def stop_service(self, service_name):
        self.running_services[service_name].state = ServiceState.stopped
        self.running_services[service_name].service.stop()
        return False

    def restart(self):
        self.output('what service do you want me to restart')
        action = self.input()
        return self.restart_service(action)

    def restart_service(self, service_name):
        self.stop_service(service_name)
        self.start_service(service_name)
        return False

    def config(self, *args):
        parser = argparse.ArgumentParser()
        parser.add_argument('--create', type=str)
        parser.add_argument('--update', type=str)
        parser.add_argument('--delete', type=str)
        parser.add_argument('--value', type=str)

        namespace, _ = parser.parse_known_args(args)

        if namespace.create:
            config = {
                'action': 'create',
                'name': namespace.create,
                'value': namespace.value
            }
        elif namespace.update:
            config = {
                'action': 'update',
                'name': namespace.update,
                'value': namespace.value
            }
        elif namespace.delete:
            config = {
                'action': 'delete',
                'name': namespace.delete
            }
        else:
            raise ValueError('')

        def set_config(parent, name, value):
            if name == '':
                return

            names = name.split('.')
            if names[0] not in parent:
                if len(names) == 1:
                    if value is not None:
                        parent[names[0]] = value
                    else:
                        del parent[names[0]]
                else:
                    parent[names[0]] = {}
            else:
                if len(names) == 1:
                    if value is not None:
                        parent[names[0]] = value
                    else:
                        del parent[names[0]]
            set_config(parent[names[0]], '.'.join(names[1:]), value)

        settings = json.loads(open('iva.json', 'r').read())
        configs = settings.get('configs', {})

        if config['action'] == 'create':
            set_config(configs, config['name'], config['value'])
            settings['configs'] = configs
            open('iva.json', 'w').write(json.dumps(settings, indent=4))
        elif config['action'] == 'update':
            set_config(configs, config['name'], config['value'])
            settings['configs'] = configs
            open('iva.json', 'w').write(json.dumps(settings, indent=4))
        elif config['action'] == 'delete':
            set_config(configs, config['name'], None)
            settings['configs'] = configs
            open('iva.json', 'w').write(json.dumps(settings, indent=4))
        else:
            raise ValueError(f'arguments are not recognized')

    def help(self):
        settings = json.loads(open('iva.json', 'r').read())
        for scripts in settings.get('scripts', []):
            if os.path.isabs(scripts) and os.path.exists(scripts):
                for module in list(filter(lambda x: '__' not in x, map(lambda x: x.replace('.py', ''), os.listdir(scripts.replace('.', '/'))))):
                    spec = importlib.util.spec_from_file_location(module, os.path.join(scripts, f'{module}.py'))
                    action_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(action_module)

                    task = getattr(action_module, 'Task', None)
                    if task is None:
                        continue
                    task.help(self)
            else:
                _module = __import__(scripts, fromlist=[''])

                for module in inspect.getmembers(_module, predicate=inspect.ismodule):
                    action_module = getattr(_module, module[0])

                    task = getattr(action_module, 'Task', None)
                    if task is None:
                        continue
                    task.help(self)

        return False

    def exit(self):
        settings = json.loads(open('iva.json', 'r').read())

        tasks = settings.get('tasks', [])
        for task in sorted(filter(lambda x: x['status'] and x['on'] == 'shutdown', tasks), key=lambda x: x['priority']):
            self.run_task(task['name'])

        with self.lock.r_locked():
            task_keys = [key for key in self.running_tasks.keys()]
            job_keys = [key for key in self.running_jobs.keys()]
            service_keys = [key for key in self.running_services.keys()]

        for key in task_keys:
            task_info = self.running_tasks[key]
            task_info.task.stop()
        for key in job_keys:
            task_info = self.running_jobs[key]
            task_info.job.stop()
        for key in service_keys:
            # task = self.running_services[key]
            self.stop_service(key)
            # task.stop()

        self.event.set()
        self.input_device.stop()
        self.output_device.stop()
        return True


def main():
    colorama.init()
    interpreter = Interpreter()
    interpreter.cmdloop()
