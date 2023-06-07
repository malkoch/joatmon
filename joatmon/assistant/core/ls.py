from __future__ import print_function

import datetime
import os

from joatmon.assistant.task import BaseTask
from joatmon.utility import (
    convert_size,
    pretty_printer
)


class Task(BaseTask):
    def __init__(self, api, *args, **kwargs):
        super(Task, self).__init__(api, *args, **kwargs)

    @staticmethod
    def params():
        return []

    def run(self):
        path = self.args[0]

        parent_os_path = self.kwargs.get('parent_os_path', '')
        os_path = self.kwargs.get('os_path', '')

        ui = []

        text, printer = pretty_printer([('NAME', 2), ('SIZE', 2), ('CREATE TIME', 4), ('MODIFIED TIME', 4)], m=100)
        ui.append(text)

        ui.append(
            printer(
                [
                    '.',
                    convert_size(os.path.getsize(os.path.join(parent_os_path, os_path[1:], path))),
                    datetime.datetime.fromtimestamp(os.path.getctime(os.path.join(parent_os_path, os_path[1:], path))).isoformat(),
                    datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(parent_os_path, os_path[1:], path))).isoformat()
                ]
            )
        )
        ui.append(
            printer(
                [
                    '..',
                    convert_size(os.path.getsize(os.path.join(parent_os_path, os_path[1:]))),
                    datetime.datetime.fromtimestamp(os.path.getctime(os.path.join(parent_os_path, os_path[1:]))).isoformat(),
                    datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(parent_os_path, os_path[1:]))).isoformat()
                ]
            )
        )

        for item in os.listdir(os.path.join(parent_os_path, os_path[1:], path)):
            item_path = os.path.join(parent_os_path, os_path[1:], path, item)

            ui.append(
                printer(
                    [
                        item,
                        convert_size(os.path.getsize(item_path)),
                        datetime.datetime.fromtimestamp(os.path.getctime(item_path)).isoformat(),
                        datetime.datetime.fromtimestamp(os.path.getmtime(item_path)).isoformat()
                    ]
                )
            )
        self.api.show_('m1', os_path, ui)

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
