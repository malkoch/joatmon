import importlib.util
import inspect
import json
import os

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return ''

    @staticmethod
    def params():
        return []

    def run(self):
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

        if not self.event.is_set():
            self.event.set()