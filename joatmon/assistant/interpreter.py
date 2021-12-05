import json
import os
import re
import threading
import time
from cmd import Cmd

import colorama
import schedule
from colorama import Fore

from joatmon.context import (
    initialize_context,
    teardown_context
)
from joatmon.system.core import RWLock
from joatmon.system.microphone import InputDriver
from joatmon.system.speaker import OutputDevice

PROMPT_CHAR = '~>'
COMMA_MATCHER = re.compile(r" (?=(?:[^\"']*[\"'][^\"']*[\"'])*[^\"']*$)")


class Interpreter(Cmd):
    first_reaction_text = ""
    first_reaction_text += Fore.BLUE + 'IVA\'s sound is by default disabled.' + Fore.RESET
    first_reaction_text += "\n"
    first_reaction_text += Fore.BLUE + 'In order to let IVA talk out loud type: '
    first_reaction_text += Fore.RESET + Fore.RED + 'enable sound' + Fore.RESET
    first_reaction_text += "\n"
    first_reaction_text += Fore.BLUE + "Type 'help' for a list of available actions." + Fore.RESET
    first_reaction_text += "\n"

    prompt = first_reaction_text + Fore.RED + "{} What can I do for you?: ".format(PROMPT_CHAR) + Fore.RESET

    def __init__(self):
        self.tts_enabled = False
        self.stt_enabled = False

        if self.tts_enabled:
            first_reaction_text = ""
            first_reaction_text += 'IVA\'s sound is by default disabled.'
            first_reaction_text += "\n"
            first_reaction_text += 'In order to let IVA talk out loud type: '
            first_reaction_text += 'enable sound'
            first_reaction_text += "\n"
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

        initialize_context()

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
            action, *args = COMMA_MATCHER.split(line)

            if action == 'enable':
                return self.enable(*args)
            if action == 'disable':
                return self.disable(*args)
            if action == 'help':
                return self.help()
            if action == 'exit':
                return self.exit()

            # it could be external scripts
            settings = json.loads(open('iva.json', 'r').read())
            for scripts in settings['scripts']:
                _module = __import__(scripts, fromlist=[f'{action}'])
                action_module = getattr(_module, action)
                action_module.sys.argv = [action] + list(args)

                # module might not have function named task
                task = getattr(action_module, 'Task')

                # function might not have any argument
                t = task(self)
                if hash(t) in self.running_tasks:
                    raise ValueError(f'{hash(t)} already is in tasks you need to wait for the task to finish')
                self.running_tasks[hash(t)] = t
                t.start()
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
        return False

    def disable(self, *args):
        for arg in args:
            if arg == 'tts':
                self.tts_enabled = False
                self.output_device.tts_enabled = False
            if arg == 'stt':
                self.stt_enabled = False
                self.input_device.stt_enabled = False
        return False

    def help(self):
        settings = json.loads(open('iva.json', 'r').read())
        for scripts in settings['scripts']:
            for module in list(filter(lambda x: '__' not in x, map(lambda x: x.replace('.py', ''), os.listdir(scripts.replace('.', '/'))))):
                print(module)
                _module = __import__(settings['scripts'][0], fromlist=[module])

                action_module = getattr(_module, module)
                task = getattr(action_module, 'Task')
                task.help()

        return False

    def exit(self):
        settings = json.loads(open('iva.json', 'r').read())

        tasks = settings.get('tasks', {})
        for task in sorted(filter(lambda x: x['status'] and x['schedule'] == '@shutdown', tasks.values()), key=lambda x: x['priority']):
            self.do_action(task['command'].replace('"', ''))

        teardown_context()

        with self.lock.r_locked():
            keys = [key for key in self.running_tasks.keys()]

        for key in keys:
            task = self.running_tasks[key]
            task.stop()

        self.event.set()
        self.input_device.stop()
        self.output_device.stop()
        return True


def main():
    colorama.init()
    interpreter = Interpreter()
    interpreter.cmdloop()
