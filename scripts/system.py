from __future__ import print_function

import psutil

from joatmon.assistant.task import BaseTask
from joatmon.utility import convert_size


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def params():
        return ['action']

    def run(self):
        action = self.kwargs.get('action', '') or self.api.listen('what do you want the action to be')

        if action == 'memory':
            self.api.say(
                f'Memory Total: {convert_size(psutil.virtual_memory().total)}, '
                f'Memory Used: {convert_size(psutil.virtual_memory().used)}, '
                f'Memory Free: {convert_size(psutil.virtual_memory().free)}, '
                f'Memory Percent: {psutil.virtual_memory().percent}, '
                f'Swap Total: {convert_size(psutil.swap_memory().total)}, '
                f'Swap Used: {convert_size(psutil.swap_memory().used)}, '
                f'Swap Free: {convert_size(psutil.swap_memory().free)}, '
                f'Swap Percent: {psutil.swap_memory().percent}'
            )
        if action == 'cpu':
            self.api.say(
                f'CPU Total Usage: {psutil.cpu_percent(percpu=False)}, '
                f'CPU Per Usage: {psutil.cpu_percent(percpu=True)}, '
                f'CPU Count: {psutil.cpu_count(logical=False)}, '
                f'CPU Count Logical: {psutil.cpu_count(logical=True)}'
            )
        if action == 'disk':
            for d in psutil.disk_partitions():
                self.api.say(
                    f'Disk Device: {d.device}, '
                    f'Disk Mount: {d.mountpoint}, '
                    f'Disk File System Type: {d.fstype}'
                )

                self.api.say(
                    f'Disk Total: {convert_size(psutil.disk_usage(d.device).total)}, '
                    f'Disk Used: {convert_size(psutil.disk_usage(d.device).used)}, '
                    f'Disk Free: {convert_size(psutil.disk_usage(d.device).free)}, '
                    f'Disk Percent: {psutil.disk_usage(d.device).percent}'
                )

        if action == 'battery':
            self.api.say(
                f'Battery Percent: {psutil.sensors_battery().percent}, '
                f'Battery Plugged: {psutil.sensors_battery().power_plugged}, '
                f'Batter Left: {psutil.sensors_battery().secsleft}'
            )
        # if action == 'process':
        #     for p in psutil.pids():
        #         process = psutil.Process(p)
        #         print(process)

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
