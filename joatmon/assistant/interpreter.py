import importlib.util
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
            if action == 'help':
                return self.help()
            if action == 'exit':
                return self.exit()

            action_found = False

            # it could be external scripts
            settings = json.loads(open('iva.json', 'r').read())
            for scripts in set(settings.get('scripts', []) + ['joatmon.assistant.scripts']):
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

    def help(self):
        settings = json.loads(open('iva.json', 'r').read())
        for scripts in set(settings.get('scripts', []) + ['joatmon.assistant.scripts']):
            for module in list(filter(lambda x: '__' not in x, map(lambda x: x.replace('.py', ''), os.listdir(scripts.replace('.', '/'))))):
                print(module)
                _module = __import__(settings['scripts'][0], fromlist=[module])

                action_module = getattr(_module, module)
                task = getattr(action_module, 'Task')
                task.help()

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
