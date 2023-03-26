import argparse
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
from joatmon.system.core import RWLock
from joatmon.system.microphone import InputDriver
from joatmon.system.speaker import OutputDevice

PROMPT_CHAR = '~>'
COMMA_MATCHER = re.compile(r" (?=(?:[^\"']*[\"'][^\"']*[\"'])*[^\"']*$)")


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
        self._tts = False
        self._stt = False

        settings = json.loads(open('iva.json', 'r').read())

        self._tts = settings.get('tts', False) and False
        self._stt = settings.get('stt', False) and False

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
        self.running_tasks = {}
        self.running_jobs = {}
        self.running_services = {}

        self.event = threading.Event()
        self.cleaning_thread = threading.Thread(target=self.clean)
        self.cleaning_thread.start()
        self.job_thread = threading.Thread(target=self.run_jobs)
        self.job_thread.start()
        self.service_thread = threading.Thread(target=self.run_services)
        self.service_thread.start()

        settings = json.loads(open('iva.json', 'r').read())

        class CTX:
            ...

        ctx = CTX()
        context.set_ctx(ctx)

        tasks = settings.get('tasks', {})
        for task in sorted(filter(lambda x: x['status'] and x['on'] == 'startup', tasks.values()), key=lambda x: x['priority']):
            self.do_action(task['command'], name=task['name'], task_type='task')  # need to do them in background
        services = settings.get('services', {})
        for service in sorted(filter(lambda x: x['status'] and x['mode'] == 'automatic', services.values()), key=lambda x: x['priority']):
            self.do_action(service['command'], name=service['name'], task_type='service')  # need to do them in background

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

    def output(self, text):
        self.output_device.output(text)

    def precmd(self, line):
        return 'action ' + line

    def do_action(self, line, name=None, task_type=None):
        # need to be able to process | character as well
        try:
            if line is None or line == '':
                return False

            action, *args = COMMA_MATCHER.split(line)

            if action is None or action == '':
                return False

            if action.lower() == 'enable':
                return self.enable(*args)
            if action.lower() == 'disable':
                return self.disable(*args)
            if action.lower() == 'task':
                return self.task(*args)
            if action.lower() == 'job':
                return self.job(*args)
            if action.lower() == 'service':
                return self.service(*args)
            if action.lower() == 'config':
                return self.config(*args)
            if action.lower() == 'help':
                return self.help()
            if action.lower() == 'exit':
                return self.exit()

            action_found = False

            # it could be external scripts
            settings = json.loads(open('iva.json', 'r').read())
            for scripts in settings.get('scripts', []):
                if os.path.isabs(scripts) and os.path.exists(scripts):
                    spec = importlib.util.spec_from_file_location(action, os.path.join(scripts, f'{action}.py'))
                    action_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(action_module)
                else:
                    try:
                        _module = __import__(scripts, fromlist=[f'{action}'])
                    except ModuleNotFoundError:
                        continue

                    action_module = getattr(_module, action, None)

                if action_module is None:
                    continue
                action_module.sys.argv = [action] + list(args)

                # module might not have function named task
                if task_type is None or task_type == 'task':
                    task = getattr(action_module, 'Task', None)
                else:
                    task = getattr(action_module, task_type.title(), None)

                if task is None:
                    continue

                if task_type is None or task_type == 'task':
                    t = task(self)
                else:
                    if name is None:
                        raise ValueError('job or service has to have name')
                    t = task()

                t.name = name

                action_found = True

                # function might not have any argument

                # token = secrets.token_hex()
                # self.running_tasks[token] = t
                if isinstance(t, BaseTask):
                    if hash(t) in self.running_tasks:
                        continue
                        # raise ValueError(f'{hash(t)} already is in tasks you need to wait for the task to finish')
                    self.running_tasks[hash(t)] = t
                    t.start()
                elif isinstance(t, BaseJob):
                    if name in self.running_jobs:
                        continue
                        # raise ValueError(f'{hash(t)} already is in tasks you need to wait for the task to finish')
                    self.running_jobs[name] = t
                    t.start()
                elif isinstance(t, BaseService):
                    if name in self.running_services:
                        continue
                        # raise ValueError(f'{hash(t)} already is in tasks you need to wait for the task to finish')
                    self.running_services[name] = t
                    t.start()
                else:
                    raise ValueError(f'type {type(t)} is not recognized')

            if not action_found:
                self.output_device.output('action is not found')
            return False
        except:
            # print(str(ex))  # use stacktrace and write all exception details, line number, function name, file name etc.
            # return self.exit()
            ...

    def run_jobs(self):
        settings = json.loads(open('iva.json', 'r').read())

        jobs = settings.get('jobs', {})
        for job in sorted(filter(lambda x: x['status'], jobs.values()), key=lambda x: x['priority']):
            schedule.every(int(job['schedule'])).seconds.do(self.do_action, job['command'].replace('"', ''), name=job['name'], task_type='job')  # need to do them in background

        while not self.event.is_set():
            schedule.run_pending()
            time.sleep(0.1)

    def run_services(self):
        settings = json.loads(open('iva.json', 'r').read())

        services = settings.get('services', {})

        while not self.event.is_set():
            for service in sorted(filter(lambda x: x['status'], services.values()), key=lambda x: x['priority']):
                if service['name'] in self.running_services:
                    continue
                self.do_action(service['command'].replace('"', ''), name=service['name'], task_type='service')  # need to do them in background

            time.sleep(0.1)

    def clean(self):
        while not self.event.is_set():
            with self.lock.r_locked():
                task_keys = [key for key in self.running_tasks.keys()]
                job_keys = [key for key in self.running_jobs.keys()]
                service_keys = [key for key in self.running_services.keys()]

            delete_task_keys = []
            for key in task_keys:
                task = self.running_tasks[key]
                if not task.running():
                    delete_task_keys.append(key)
            delete_job_keys = []
            for key in job_keys:
                task = self.running_jobs[key]
                if not task.running():
                    delete_job_keys.append(key)
            delete_service_keys = []
            for key in service_keys:
                task = self.running_services[key]
                if not task.running():
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

            time.sleep(0.1)

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

    def task(self, *args):
        parser = argparse.ArgumentParser()
        parser.add_argument('--create', dest='create', action='store_true')
        parser.add_argument('--update', dest='update', action='store_true')
        parser.add_argument('--delete', dest='delete', action='store_true')
        parser.add_argument('--list', dest='list', action='store_true')
        parser.add_argument('--run', dest='run', action='store_true')
        parser.add_argument('--name', type=str, default='')
        parser.add_argument('--script', type=str, default='')
        parser.add_argument('--priority', type=int, default=1)
        parser.add_argument('--on', type=str, default='startup')
        parser.add_argument('--status', type=bool, default=True)
        parser.set_defaults(create=False)
        parser.set_defaults(update=False)
        parser.set_defaults(delete=False)
        parser.set_defaults(list=False)
        parser.set_defaults(run=False)

        namespace, args = parser.parse_known_args(args)

        settings = json.loads(open('iva.json', 'r').read())
        tasks = settings.get('tasks', {})

        if namespace.create:
            if tasks.get(namespace.name, None) is not None:
                raise ValueError('task is already exists')
            tasks[namespace.name] = {
                'name': namespace.name,
                'command': ' '.join([namespace.script] + args),
                'priority': namespace.priority or 1,
                'on': namespace.on or 'startup',
                'status': True
            }
        elif namespace.update:
            if tasks.get(namespace.name, None) is None:
                raise ValueError('task does not exists')
            tasks[namespace.name] = {
                'name': namespace.name,
                'command': ' '.join([namespace.script] + args),
                'priority': namespace.priority or 1,
                'on': namespace.on or 'startup',
                'status': True
            }
        elif namespace.delete:
            if tasks.get(namespace.name, None) is None:
                raise ValueError('task does not exists')
            del tasks[namespace.name]
        elif namespace.list:
            for task_hash, task in self.running_tasks.items():
                print(task)
        elif namespace.run:
            ...

        settings['tasks'] = tasks
        open('iva.json', 'w').write(json.dumps(settings, indent=4))

    def job(self, *args):
        parser = argparse.ArgumentParser()
        parser.add_argument('--create', dest='create', action='store_true')
        parser.add_argument('--update', dest='update', action='store_true')
        parser.add_argument('--delete', dest='delete', action='store_true')
        parser.add_argument('--list', dest='list', action='store_true')
        parser.add_argument('--run', dest='run', action='store_true')
        parser.add_argument('--name', type=str, default='')
        parser.add_argument('--script', type=str, default='')
        parser.add_argument('--priority', type=int, default=1)
        parser.add_argument('--schedule', type=int, default=24 * 60 * 60)
        parser.add_argument('--status', type=bool, default=True)
        parser.set_defaults(create=False)
        parser.set_defaults(update=False)
        parser.set_defaults(delete=False)
        parser.set_defaults(list=False)
        parser.set_defaults(run=False)

        namespace, args = parser.parse_known_args(args)

        settings = json.loads(open('iva.json', 'r').read())
        jobs = settings.get('jobs', {})

        if namespace.create:
            if jobs.get(namespace.name, None) is not None:
                raise ValueError('task is already exists')
            jobs[namespace.name] = {
                'name': namespace.name,
                'command': ' '.join([namespace.script] + args),
                'priority': namespace.priority or 1,
                'schedule': namespace.schedule or 24 * 60 * 60,
                'status': True
            }
        elif namespace.update:
            if jobs.get(namespace.name, None) is None:
                raise ValueError('task does not exists')
            jobs[namespace.name] = {
                'name': namespace.name,
                'command': ' '.join([namespace.script] + args),
                'priority': namespace.priority or 1,
                'schedule': namespace.schedule or 24 * 60 * 60,
                'status': True
            }
        elif namespace.delete:
            if jobs.get(namespace.name, None) is None:
                raise ValueError('task does not exists')
            del jobs[namespace.name]
        elif namespace.list:
            for task_hash, task in self.running_jobs.items():
                print(task['name'], task['name'] in self.running_jobs)
        elif namespace.run:
            ...

        settings['jobs'] = jobs
        open('iva.json', 'w').write(json.dumps(settings, indent=4))

    def service(self, *args):
        # need to add depends on
        parser = argparse.ArgumentParser()
        parser.add_argument('--create', dest='create', action='store_true')
        parser.add_argument('--update', dest='update', action='store_true')
        parser.add_argument('--delete', dest='delete', action='store_true')
        parser.add_argument('--list', dest='list', action='store_true')
        parser.add_argument('--start', dest='start', action='store_true')
        parser.add_argument('--stop', dest='stop', action='store_true')
        parser.add_argument('--restart', dest='restart', action='store_true')
        parser.add_argument('--name', type=str, default='')
        parser.add_argument('--script', type=str, default='')
        parser.add_argument('--priority', type=int, default=1)
        parser.add_argument('--mode', type=str, default='automatic')
        parser.add_argument('--status', type=bool, default=True)
        parser.set_defaults(create=False)
        parser.set_defaults(update=False)
        parser.set_defaults(delete=False)
        parser.set_defaults(list=False)
        parser.set_defaults(start=False)
        parser.set_defaults(stop=False)
        parser.set_defaults(restart=False)

        namespace, args = parser.parse_known_args(args)

        settings = json.loads(open('iva.json', 'r').read())
        services = settings.get('services', {})

        if namespace.create:
            if services.get(namespace.name, None) is not None:
                raise ValueError('task is already exists')
            services[namespace.name] = {
                'name': namespace.name,
                'command': ' '.join([namespace.script] + args),
                'priority': namespace.priority or 1,
                'mode': namespace.mode or 'automatic',
                'status': True
            }
        elif namespace.update:
            if services.get(namespace.name, None) is None:
                raise ValueError('task does not exists')
            services[namespace.name] = {
                'name': namespace.name,
                'command': ' '.join([namespace.script] + args),
                'priority': namespace.priority or 1,
                'mode': namespace.mode or 'automatic',
                'status': True
            }
        elif namespace.delete:
            if services.get(namespace.name, None) is None:
                raise ValueError('task does not exists')
            del services[namespace.name]
        elif namespace.list:
            for name, service in services.items():
                print(service['name'], service['name'] in self.running_services)
        elif namespace.start:
            service = list(filter(lambda x: x['name'] == namespace.name, services.values()))
            if len(service) == 1:
                service = service[0]
            else:
                service = None

            if service is not None:
                self.do_action(service['command'], name=service['name'], task_type='service')  # need to do them in background
        elif namespace.stop:
            ...
        elif namespace.restart:
            ...

        settings['services'] = services
        open('iva.json', 'w').write(json.dumps(settings, indent=4))

    def config(self, *args):
        parser = argparse.ArgumentParser()
        parser.add_argument('--create', type=str)
        parser.add_argument('--update', type=str)
        parser.add_argument('--delete', type=str)
        parser.add_argument('--value', type=str)

        namespace, _ = parser.parse_known_args(args)

        config = None
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

        tasks = settings.get('tasks', {})
        for task in sorted(filter(lambda x: x['status'] and x['on'] == 'shutdown', tasks.values()), key=lambda x: x['priority']):
            self.do_action(task['command'], name=task['name'], task_type='task')

        with self.lock.r_locked():
            task_keys = [key for key in self.running_tasks.keys()]
            job_keys = [key for key in self.running_jobs.keys()]
            service_keys = [key for key in self.running_services.keys()]

        for key in task_keys:
            task = self.running_tasks[key]
            task.stop()
        for key in job_keys:
            task = self.running_jobs[key]
            task.stop()
        for key in service_keys:
            task = self.running_services[key]
            task.stop()

        self.event.set()
        self.input_device.stop()
        self.output_device.stop()
        return True


def main():
    colorama.init()
    interpreter = Interpreter()
    interpreter.cmdloop()
