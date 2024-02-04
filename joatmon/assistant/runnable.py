import asyncio
import uuid

from transitions import Machine


class ActionException(Exception):
    def __init__(self, code: int):
        self.code = code

    def __str__(self):
        return f'({self.code})'

    def __repr__(self):
        return str(self)


class Runnable:
    """
    Runnable class for managing asynchronous tasks.

    This class provides a way to manage asynchronous tasks, including starting, stopping, and checking the running status of the tasks.

    Attributes:
        machine (Machine): The state machine for managing the state of the task.
        process_id (uuid.UUID): The unique identifier for the task.
        info (Union[Task, Service]): The information about the task.
        api (API): The API object.
        kwargs (dict): Additional keyword arguments.
        event (asyncio.Event): An event for signaling the termination of the task.
        task (asyncio.Task): The task that is being run.

    Args:
        info ([Task, Service]): The information about the task.
        api (API): The API object.
        kwargs (dict): Additional keyword arguments.
    """

    def __init__(self, info, api, **kwargs):  # another parameter called cache output
        """
        Initialize the Runnable.

        Args:
            info ([Task, Service]): The information about the task.
            api (API): The API object.
            kwargs (dict): Additional keyword arguments.
        """
        states = ['none', 'started', 'stopped', 'running', 'exception', 'starting', 'stopping']
        self.machine = Machine(model=self, states=states, initial='none')

        self.process_id = uuid.uuid4()
        self.info = info
        self.api = api
        self.kwargs = kwargs
        self.event = asyncio.Event()
        self.task = None

    def __str__(self):
        return f'{self.process_id} {self.info} {self.machine.state}'

    def __repr__(self):
        return str(self)

    @staticmethod
    def help():
        """
        Provide help about the Runnable.

        Returns:
            dict: An empty dictionary as this method needs to be implemented in subclasses.
        """
        return {}

    async def run(self):
        """
        Run the task.

        This method needs to be implemented in subclasses.
        """
        raise NotImplementedError

    def running(self):
        """
        Check if the task is running.

        Returns:
            bool: True if the task is running, False otherwise.
        """
        return not self.task.done()

    def start(self):
        """
        Start the task.

        This method starts the task by setting the state to 'starting', firing the 'begin' event, and then setting the state to 'started'.
        """
        self.machine.set_state('starting')
        self.task = asyncio.ensure_future(self.run(), loop=self.api.loop)
        self.machine.set_state('started')

    def stop(self):
        """
        Stop the task.

        This method stops the task by setting the state to 'stopping', firing the 'end' event, setting the event, setting the state to 'stopped', and then cancelling the task.
        """
        self.machine.set_state('stopping')
        self.event.set()
        if self.task and not self.task.done():
            self.task.cancel()
        self.machine.set_state('stopped')
