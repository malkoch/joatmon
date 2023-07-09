from __future__ import print_function

import webbrowser

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return {
            "name": "browser",
            "description": "a function for user to open a browser using a link",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "url to open in the browser"
                    }
                },
                "required": ["url"]
            }
        }

    def run(self):
        webbrowser.register('firefox', None, webbrowser.BackgroundBrowser(r'C:\Program Files\Mozilla Firefox\firefox.exe'))
        webbrowser.get('firefox').open_new_tab(self.kwargs.get('url', None))

        if not self.stop_event.is_set():
            self.stop_event.set()
