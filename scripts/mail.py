from __future__ import print_function

import argparse
import datetime
import json
import smtplib
import sys
import time
from email.mime.text import MIMEText

from joatmon.assistant.service import BaseService
from joatmon.assistant.task import BaseTask
from joatmon.decorator.message import consumer


class Task(BaseTask):
    def __init__(self, api):
        parser = argparse.ArgumentParser()
        parser.add_argument('--background', dest='background', action='store_true')
        parser.set_defaults(background=False)

        namespace, _ = parser.parse_known_args(sys.argv)

        super(Task, self).__init__(api, namespace.background, 1, 100)

    @staticmethod
    def help(api):
        ...

    def run(self):
        config = json.loads(open('iva.json', 'r').read())['configs']['mail']

        text_subtype = 'plain'
        content = """Test message"""
        subject = "Sent from Python"

        smtp_server = config['smtp_server']
        port = config['smtp_port']
        sender_email = config['sender_mail']
        password = config['sender_password']
        receivers = config['receivers']

        msg = MIMEText(content, text_subtype)
        msg['Subject'] = subject
        msg['From'] = sender_email

        conn = smtplib.SMTP(smtp_server, port)
        conn.login(sender_email, password)
        conn.sendmail(sender_email, receivers, msg.as_string())
        conn.quit()

        if not self.event.is_set():
            self.event.set()


class Service(BaseService):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--kafka', type=str)
        parser.add_argument('--topic', type=str)

        namespace, _ = parser.parse_known_args(sys.argv)

        super(Service, self).__init__(1, 100)

        if namespace.topic == 'arxiv':
            self.new_pdf = consumer(namespace.kafka, namespace.topic)(self.new_pdf)

        self.buffer = []
        self.last_mail_time = datetime.datetime.now()

    @staticmethod
    def help(api):
        ...

    def new_pdf(self, pdf):
        self.buffer.append(pdf)

    def send_mail(self):
        if len(self.buffer) == 0:
            return

        config = json.loads(open('iva.json', 'r').read())['configs']['mail']

        text_subtype = 'html'
        content = """    
        <html>
            <head></head>
            <body>
        """ + '<br><br>'.join(
            [
                f'<b>Title:</b> {source["title"]}<br>'
                f'<b>Summary:</b> {source["summary"]}<br>'
                f'<b>Link:</b> {source["id"]}<br>'
                f'<b>Publish Date:</b> {source["published"]}' for source in self.buffer]
        ) + """
            </body>
        </html>
        """
        subject = "Sent from Python"

        smtp_server = config['smtp_server']
        port = config['smtp_port']
        sender_email = config['sender_mail']
        password = config['sender_password']
        receivers = config['receivers']

        msg = MIMEText(content, text_subtype)
        msg['Subject'] = subject
        msg['From'] = sender_email

        conn = smtplib.SMTP(smtp_server, port)
        conn.login(sender_email, password)
        conn.sendmail(sender_email, receivers, msg.as_string())
        conn.quit()

        self.buffer = []

    def run(self):
        try:
            while True:
                if self.event.is_set():
                    break
                time.sleep(1)

                if datetime.datetime.now() - self.last_mail_time > datetime.timedelta(hours=1):
                    self.last_mail_time = datetime.datetime.now()
                    self.send_mail()
        except:
            ...

    def stop(self):
        super(Service, self).stop()


if __name__ == '__main__':
    Task(None).run()
