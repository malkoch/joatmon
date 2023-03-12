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
# import schedule
from colorama import Fore

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
        settings = json.loads(open('iva.json', 'r').read())

        self.tts_enabled = settings.get('tts', False)
        self.stt_enabled = settings.get('stt', False)

        if self.tts_enabled:
            first_reaction_text = ""
            # first_reaction_text += 'IVA\'s sound is by default disabled.'
            # first_reaction_text += "\n"
            # first_reaction_text += 'In order to let IVA talk out loud type: '
            # first_reaction_text += 'enable sound'
            # first_reaction_text += "\n"
            first_reaction_text += "Type 'help' for a list of available actions."
            first_reaction_text += "\n"

            self.prompt = first_reaction_text + "What can I do for you?: "

        self.output_device = OutputDevice(tts_enabled=self.tts_enabled)
        self.input_device = InputDriver(self.output_device, stt_enabled=self.stt_enabled)

        super(Interpreter, self).__init__(stdin=self.input_device, stdout=self.output_device)

        self.use_rawinput = False

        self.lock = RWLock()
        self.running_tasks = {}

        self.event = threading.Event()
        self.cleaning_thread = threading.Thread(target=self.clean)
        self.cleaning_thread.start()
        self.task_thread = threading.Thread(target=self.run_tasks)
        self.task_thread.start()

        settings = json.loads(open('iva.json', 'r').read())

        tasks = settings.get('tasks', {})
        for task in sorted(filter(lambda x: x['status'] and x['schedule'] == '@startup', tasks.values()), key=lambda x: x['priority']):
            self.do_action(task['command'].replace('"', ''))  # need to do them in background
        services = settings.get('services', {})
        for service in sorted(filter(lambda x: x['status'] and x['mode'] == 'automatic', services.values()), key=lambda x: x['priority']):
            self.do_action(service['command'].replace('"', ''))  # need to do them in background

    def input(self, prompt=None):
        return self.input_device.input(prompt)

    def output(self, text):
        self.output_device.output(text)

    def precmd(self, line):
        return 'action ' + line

    def postcmd(self, stop, line):
        while self.output_device.speaking_event.is_set():
            time.sleep(0.1)
            continue

        if self.tts_enabled:
            self.prompt = "What can I do for you?: "
        else:
            self.prompt = Fore.RED + "{} What can I do for you?: ".format(PROMPT_CHAR) + Fore.RESET
        return stop

    def do_action(self, line):
        # need to be able to process | character as well
        try:
            if line is None or line == '':
                return False

            action, *args = COMMA_MATCHER.split(line)

            if action is None or action == '':
                return False

            if action == 'enable':
                return self.enable(*args)
            if action == 'disable':
                return self.disable(*args)
            if action == 'task':
                return self.task(*args)
            if action == 'service':
                return self.service(*args)
            if action == 'config':
                return self.config(*args)
            if action == 'help':
                return self.help()
            if action == 'exit':
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
                task = getattr(action_module, 'Task', None)

                if task is None:
                    continue

                action_found = True

                # function might not have any argument
                t = task(self)
                if hash(t) in self.running_tasks:
                    raise ValueError(f'{hash(t)} already is in tasks you need to wait for the task to finish')
                self.running_tasks[hash(t)] = t
                t.start()

            if not action_found:
                self.output_device.output('action is not found')
            return False
        except Exception as ex:
            print(str(ex))  # use stacktrace and write all exception details, line number, function name, file name etc.
            return self.exit()

    def run_tasks(self):
        settings = json.loads(open('iva.json', 'r').read())

        tasks = settings.get('tasks', {})
        for task in sorted(filter(lambda x: x['status'] and x['schedule'] != '@startup', tasks.values()), key=lambda x: x['priority']):
            schedule.every(int(task['schedule'])).seconds.do(self.do_action, task['command'].replace('"', ''))  # need to do them in background

        while not self.event.is_set():
            schedule.run_pending()
            time.sleep(0.1)

    def clean(self):
        while not self.event.is_set():
            with self.lock.r_locked():
                keys = [key for key in self.running_tasks.keys()]

            delete_keys = []
            for key in keys:
                task = self.running_tasks[key]
                if not task.running():
                    delete_keys.append(key)

            for key in delete_keys:
                with self.lock.w_locked():
                    del self.running_tasks[key]

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
        parser.add_argument('--create', type=str)
        parser.add_argument('--update', type=str)
        parser.add_argument('--delete', type=str)
        parser.add_argument('--command', type=str)
        parser.add_argument('--priority', type=int)
        parser.add_argument('--schedule', type=str)
        parser.add_argument('--status', type=bool)

        namespace, _ = parser.parse_known_args(args)

        action = None
        if namespace.create:
            action = {
                'action': 'create',
                'name': namespace.create,
                'command': namespace.command.replace('"', ''),
                'priority': namespace.priority,
                'schedule': namespace.schedule
            }
        elif namespace.update:
            action = {
                'action': 'update',
                'name': namespace.update,
                'command': namespace.command.replace('"', ''),
                'priority': namespace.priority,
                'schedule': namespace.schedule,
                'status': namespace.status
            }
        elif namespace.delete:
            action = {
                'action': 'delete',
                'name': namespace.delete
            }

        settings = json.loads(open('iva.json', 'r').read())
        tasks = settings.get('tasks', {})

        if action['action'] == 'create':
            task = tasks.get(action['name'], None)
            if task:
                raise ValueError('task is already exists')
            task = {
                'command': action['command'],
                'priority': action['priority'] or 1,
                'schedule': action['schedule'] or '* * * * *',
                'status': True
            }
            tasks[action['name']] = task
            settings['tasks'] = tasks
            open('iva.json', 'w').write(json.dumps(settings, indent=4))
        elif action['action'] == 'update':
            task = tasks.get(action['name'], None)
            if task is None:
                raise ValueError('task does not exists')
            task = {
                'command': action['command'],
                'priority': action['priority'] or 1,
                'schedule': action['schedule'] or '* * * * *',
                'status': action['status'] or False,
            }
            tasks[action['name']] = task
            settings['tasks'] = tasks
            open('iva.json', 'w').write(json.dumps(settings, indent=4))
        elif action['action'] == 'delete':
            task = tasks.get(action['name'], None)
            if task is None:
                raise ValueError('task does not exists')
            del tasks[action['name']]
            settings['tasks'] = tasks
            open('iva.json', 'w').write(json.dumps(settings, indent=4))
        else:
            raise ValueError(f'arguments are not recognized')

    def service(self, *args):
        parser = argparse.ArgumentParser()
        parser.add_argument('--create', type=str)
        parser.add_argument('--update', type=str)
        parser.add_argument('--delete', type=str)
        parser.add_argument('--command', type=str)
        parser.add_argument('--priority', type=int)

        namespace, _ = parser.parse_known_args(args)

        action = None
        if namespace.create:
            action = ['create', namespace.create, namespace.command]
        elif namespace.update:
            action = ['update', namespace.update, namespace.command]
        elif namespace.delete:
            action = ['delete', namespace.delete, namespace.command]

        if action[0] == 'create':
            ...
        elif action[0] == 'update':
            ...
        elif action[0] == 'delete':
            ...
        else:
            raise ValueError(f'arguments are not recognized')

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
        self.output_device.output('shutting down')
        time.sleep(1)

        settings = json.loads(open('iva.json', 'r').read())

        tasks = settings.get('tasks', {})
        for task in sorted(filter(lambda x: x['status'] and x['schedule'] == '@shutdown', tasks.values()), key=lambda x: x['priority']):
            self.do_action(task['command'].replace('"', ''))

        with self.lock.r_locked():
            keys = [key for key in self.running_tasks.keys()]

        for key in keys:
            task = self.running_tasks[key]
            task.stop()

        self.output_device.output('good bye')
        time.sleep(1)

        self.event.set()
        self.input_device.stop()
        self.output_device.stop()
        return True


def main():
    colorama.init()
    interpreter = Interpreter()
    interpreter.cmdloop()
