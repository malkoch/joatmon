import importlib.util
import json
import os

from joatmon.assistant.task import BaseTask
from joatmon.core.utility import JSONEncoder, get_module_classes


class Task(BaseTask):
    """
    Task class for providing help about a function.

    This class provides a way to learn about a function by returning its name, description, and parameters.

    Attributes:
        task (str): The task name.
        api (object): The API object.
        kwargs (dict): Additional keyword arguments.

    Args:
        task (str): The task name.
        api (object): The API object.
        kwargs (dict): Additional keyword arguments.
    """

    def __init__(self, task, api, **kwargs):
        """
        Initialize the Task.

        Args:
            task (str): The task name.
            api (object): The API object.
            kwargs (dict): Additional keyword arguments.
        """
        super(Task, self).__init__(task, api, **kwargs)

    @staticmethod
    def help():
        """
        Provide help about the 'help' function.

        Returns:
            dict: A dictionary containing the name, description, and parameters of the 'help' function.
        """
        return {
            'name': 'help',
            'description': 'a function for user to learn about a function',
            'parameters': {
                'type': 'object',
                'properties': {
                    'function': {
                        'type': 'string',
                        'description': 'name of the function that the user want to learn about',
                    }
                },
                'required': ['name'],
            },
        }

    async def run(self):
        """
        Run the task.

        This method runs the task by loading the scripts from the API folders, importing the modules, and getting the help about the functions in the modules.
        """
        script = self.kwargs.get('script', None)

        tasks = []

        for scripts in self.api.folders:
            if os.path.isabs(scripts) and os.path.exists(scripts):
                for module in list(
                        filter(
                            lambda x: '__' not in x,
                            map(lambda x: x.replace('.py', ''), os.listdir(scripts.replace('.', '/'))),
                        )
                ):
                    spec = importlib.util.spec_from_file_location(module, os.path.join(scripts, f'{module}.py'))
                    action_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(action_module)

                    if script is not None and action_module != script:
                        continue

                    task = getattr(action_module, 'Task', None)
                    if task is None:
                        continue
                    tasks.append(task)
            else:
                try:
                    _module = __import__('.'.join(scripts.split('.')), fromlist=[scripts.split('.')[-1]])

                    scripts = _module.__path__[0]

                    for script_file in os.listdir(scripts):
                        if '__' in script_file:
                            continue

                        if script is not None and script_file.replace('.py', '') != script:
                            continue

                        spec = importlib.util.spec_from_file_location(
                            scripts, os.path.join(scripts, script_file)
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
        print(functions)
        _ = json.dumps(functions, indent=4, cls=JSONEncoder)
