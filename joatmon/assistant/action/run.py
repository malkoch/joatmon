from __future__ import print_function

import os.path
import subprocess

from joatmon.assistant.runnable import ActionException
from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    """
    Task class for running an executable.

    This class provides a way to run an executable with specified arguments.

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
        Provide help about the 'run' function.

        Returns:
            dict: A dictionary containing the name, description, and parameters of the 'run' function.
        """
        return {
            'name': 'run',
            'description': 'a function for user to run an executable',
            'parameters': {
                'type': 'object',
                'properties': {'executable': {'type': 'string', 'description': 'executable to run'}},
                'required': ['executable'],
            },
        }

    async def run(self):
        """
        Run the task.

        This method runs the task by executing the specified executable with the provided arguments.

        Args:
            executable (str): The executable to run.
            args (str): The arguments to pass to the executable.
        """
        executable = self.kwargs.get('executable', '')
        args = self.kwargs.get('args', '')

        # send the os path to all executables
        # send the parent os path to all executables

        executable_path = os.path.join(self.api.cwd, 'executables', executable)

        process = subprocess.run(['python', executable_path] + args.split(' '))
        if process.returncode:
            raise ActionException(process.returncode)
