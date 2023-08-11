from __future__ import print_function

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, True, **kwargs)

    @staticmethod
    def help():
        return {
            "name": "download",
            "description": "a function for user to download a file from given url to given path",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "url of the file to be downloaded"
                    },
                    "path": {
                        "type": "string",
                        "description": "path of the file to be saved"
                    }
                },
                "required": ["url", "path"]
            }
        }

    def run(self):
        # need to do this in background
        # after it is done, need to notify user and prompt action to continue
        # it should not interfere with the current task that the user running
        if not self.stop_event.is_set():
            self.stop_event.set()
