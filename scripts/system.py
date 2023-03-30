from __future__ import print_function

import psutil

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    def run(self):
        try:
            if self.action[0] == 'memory':
                import math

                def convert_size(size_bytes):
                    if size_bytes == 0:
                        return "0B"
                    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
                    i = int(math.floor(math.log(size_bytes, 1024)))
                    p = math.pow(1024, i)
                    s = round(size_bytes / p, 2)
                    return "%s %s" % (s, size_name[i])

                self.api.output(
                    f'Memory Total: {convert_size(psutil.virtual_memory().total)}, '
                    f'Memory Used: {convert_size(psutil.virtual_memory().used)}, '
                    f'Memory Free: {convert_size(psutil.virtual_memory().free)}, '
                    f'Memory Percent: {psutil.virtual_memory().percent}, '
                    f'Swap Total: {convert_size(psutil.swap_memory().total)}, '
                    f'Swap Used: {convert_size(psutil.swap_memory().used)}, '
                    f'Swap Free: {convert_size(psutil.swap_memory().free)}, '
                    f'Swap Percent: {psutil.swap_memory().percent}'
                )
            elif self.action[0] == 'cpu':
                self.api.output(
                    f'CPU Total Usage: {psutil.cpu_percent(percpu=False)}, '
                    f'CPU Per Usage: {psutil.cpu_percent(percpu=True)}, '
                    f'CPU Count: {psutil.cpu_count(logical=False)}, '
                    f'CPU Count Logical: {psutil.cpu_count(logical=True)}'
                )
            elif self.action[0] == 'disk':
                import math

                def convert_size(size_bytes):
                    if size_bytes == 0:
                        return "0B"
                    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
                    i = int(math.floor(math.log(size_bytes, 1024)))
                    p = math.pow(1024, i)
                    s = round(size_bytes / p, 2)
                    return "%s %s" % (s, size_name[i])

                for d in psutil.disk_partitions():
                    self.api.output(
                        f'Disk Device: {d.device}, '
                        f'Disk Mount: {d.mountpoint}, '
                        f'Disk File System Type: {d.fstype}'
                    )

                    self.api.output(
                        f'Disk Total: {convert_size(psutil.disk_usage(d.device).total)}, '
                        f'Disk Used: {convert_size(psutil.disk_usage(d.device).used)}, '
                        f'Disk Free: {convert_size(psutil.disk_usage(d.device).free)}, '
                        f'Disk Percent: {psutil.disk_usage(d.device).percent}'
                    )

            elif self.action[0] == 'battery':
                self.api.output(
                    f'Battery Percent: {psutil.sensors_battery().percent}, '
                    f'Battery Plugged: {psutil.sensors_battery().power_plugged}, '
                    f'Batter Left: {psutil.sensors_battery().secsleft}'
                )
            elif self.action[0] == 'process':
                for p in psutil.pids():
                    process = psutil.Process(p)
                    print(process)
            else:
                raise ValueError(f'arguments are not recognized')

            if not self.event.is_set():
                self.event.set()
        except:
            ...


if __name__ == '__main__':
    Task(None).run()
