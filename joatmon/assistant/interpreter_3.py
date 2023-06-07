import argparse
import dataclasses
import datetime
import enum
import functools
import importlib.util
import inspect
import json
import os
import queue
import re
import sys
import threading
import time

import schedule
from PyQt5.QtCore import (
    Qt,
    QTimer
)
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStyleFactory,
    QTabWidget,
    QVBoxLayout,
    QWidget
)

from joatmon import context
from joatmon.assistant.job import BaseJob
from joatmon.assistant.service import BaseService
from joatmon.assistant.task import BaseTask
from joatmon.system.lock import RWLock
from joatmon.utility import first


class CTX:
    ...


ctx = CTX()
context.set_ctx(ctx)

PROMPT_CHAR = '~>'
COMMA_MATCHER = re.compile(r" (?=(?:[^\"']*[\"'][^\"']*[\"'])*[^\"']*$)")


def _get_task(script_name):
    task = None

    settings = json.loads(open('iva.json', 'r').read())
    for scripts in settings.get('scripts', []):
        if os.path.isabs(scripts):
            if os.path.exists(scripts) and os.path.exists(os.path.join(scripts, f'{script_name}.py')):
                spec = importlib.util.spec_from_file_location(scripts, os.path.join(scripts, f'{script_name}.py'))
                action_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(action_module)
            else:
                continue
        else:
            try:
                _module = __import__(scripts, fromlist=[f'{script_name}'])
            except ModuleNotFoundError:
                continue

            action_module = getattr(_module, script_name, None)

        if action_module is None:
            continue

        task = getattr(action_module, 'Task', None)

    return task


def _get_job(script_name):
    task = None

    settings = json.loads(open('iva.json', 'r').read())
    for scripts in settings.get('scripts', []):
        if os.path.isabs(scripts):
            if os.path.exists(scripts) and os.path.exists(os.path.join(scripts, f'{script_name}.py')):
                spec = importlib.util.spec_from_file_location(scripts, os.path.join(scripts, f'{script_name}.py'))
                action_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(action_module)
            else:
                continue
        else:
            try:
                _module = __import__(scripts, fromlist=[f'{script_name}'])
            except ModuleNotFoundError:
                continue

            action_module = getattr(_module, script_name, None)

        if action_module is None:
            continue

        task = getattr(action_module, 'Job', None)

    return task


def _get_service(script_name):
    task = None

    settings = json.loads(open('iva.json', 'r').read())
    for scripts in settings.get('scripts', []):
        if os.path.isabs(scripts):
            if os.path.exists(scripts) and os.path.exists(os.path.join(scripts, f'{script_name}.py')):
                spec = importlib.util.spec_from_file_location(scripts, os.path.join(scripts, f'{script_name}.py'))
                action_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(action_module)
            else:
                continue
        else:
            try:
                _module = __import__(scripts, fromlist=[f'{script_name}'])
            except ModuleNotFoundError:
                continue

            action_module = getattr(_module, script_name, None)

        if action_module is None:
            continue

        task = getattr(action_module, 'Service', None)

    return task


def create_task(task):
    create_args = {
        'name': task.pop('name', ''),
        'priority': int(task.pop('priority', '')),
        'on': task.pop('on', ''),
        'script': task.pop('script', ''),
        'status': True,
        'position': task.pop('position', ''),
        'args': task
    }

    settings = json.loads(open('iva.json', 'r').read())
    tasks = settings.get('tasks', [])

    tasks.append(create_args)

    settings['tasks'] = tasks
    open('iva.json', 'w').write(json.dumps(settings, indent=4))


def create_job(job):
    create_args = {
        'name': job.pop('name', ''),
        'priority': int(job.pop('priority', '')),
        'every': int(job.pop('every', '')),
        'script': job.pop('script', ''),
        'status': True,
        'position': job.pop('position', ''),
        'args': job
    }

    settings = json.loads(open('iva.json', 'r').read())
    tasks = settings.get('jobs', [])

    tasks.append(create_args)

    settings['jobs'] = tasks
    open('iva.json', 'w').write(json.dumps(settings, indent=4))


def create_service(service):
    create_args = {
        'name': service.pop('name', ''),
        'priority': int(service.pop('priority', '')),
        'mode': service.pop('mode', ''),
        'script': service.pop('script', ''),
        'status': True,
        'position': service.pop('position', ''),
        'args': service
    }

    settings = json.loads(open('iva.json', 'r').read())
    tasks = settings.get('services', [])

    tasks.append(create_args)

    settings['services'] = tasks
    open('iva.json', 'w').write(json.dumps(settings, indent=4))


class TaskState(enum.Enum):
    running = enum.auto()
    finished = enum.auto()


@dataclasses.dataclass
class TaskInfo:
    name: str
    state: TaskState
    task: BaseTask


class JobState(enum.Enum):
    running = enum.auto()
    finished = enum.auto()


@dataclasses.dataclass
class JobInfo:
    name: str
    state: JobState
    job: BaseJob


class ServiceState(enum.Enum):
    running = enum.auto()
    finished = enum.auto()
    stopped = enum.auto()


@dataclasses.dataclass
class ServiceInfo:
    name: str
    state: ServiceState
    service: BaseService


class LabelAndEdit(QWidget):
    def __init__(self, label, placeholder, parent, change=None):
        super(LabelAndEdit, self).__init__(parent=parent)

        layout = QHBoxLayout()

        self.label = QLabel(label)
        layout.addWidget(self.label, 1)

        self.line_edit = QLineEdit()
        self.line_edit.setText(placeholder)
        if change is not None:
            self.line_edit.textChanged[str].connect(functools.partial(change, label))
        layout.addWidget(self.line_edit, 2)

        self.setLayout(layout)

    def get_value(self):
        return {self.label.text(): self.line_edit.text()}


class CreateWindow(QWidget):
    def __init__(self):
        super(CreateWindow, self).__init__(None)

        layout = QVBoxLayout()

        self.dropdown = QComboBox()
        self.dropdown.addItem('task')
        self.dropdown.addItem('job')
        self.dropdown.addItem('service')
        self.dropdown.currentIndexChanged.connect(self.create_type_changed)
        layout.addWidget(self.dropdown)

        self.bottom_layout = QVBoxLayout()

        self.create_params = {}

        self.script_row = LabelAndEdit('script', self.create_params.get('script', ''), self, self.script_name_changed)
        self.bottom_layout.addWidget(self.script_row)

        self.name_edit = LabelAndEdit('name', self.create_params.get('name', ''), self, self.line_edit_changed)
        self.bottom_layout.addWidget(self.name_edit)

        self.priority_edit = LabelAndEdit('priority', self.create_params.get('priority', ''), self, self.line_edit_changed)
        self.bottom_layout.addWidget(self.priority_edit)

        self.on_edit = LabelAndEdit('on', self.create_params.get('on', ''), self, self.line_edit_changed)
        self.bottom_layout.addWidget(self.on_edit)

        self.position_edit = LabelAndEdit('position', self.create_params.get('position', ''), self, self.line_edit_changed)
        self.bottom_layout.addWidget(self.position_edit)

        layout.addLayout(self.bottom_layout)

        self.args_layout = QVBoxLayout()
        layout.addLayout(self.args_layout)

        create_button = QPushButton('Create')
        create_button.clicked.connect(self.create_pressed)
        layout.addWidget(create_button)

        self.setLayout(layout)

    def create_pressed(self):
        match self.dropdown.currentIndex():
            case 0:
                if 'every' in self.create_params:
                    self.create_params.pop('every')
                if 'mode' in self.create_params:
                    self.create_params.pop('mode')

                create_task(self.create_params)
            case 1:
                if 'on' in self.create_params:
                    self.create_params.pop('on')
                if 'mode' in self.create_params:
                    self.create_params.pop('mode')
                create_job(self.create_params)
            case 2:
                if 'on' in self.create_params:
                    self.create_params.pop('on')
                if 'every' in self.create_params:
                    self.create_params.pop('every')
                create_service(self.create_params)
            case _:
                ...

        self.close()

    def script_name_changed(self, k, v):
        self.create_params = {}

        for i in reversed(range(self.args_layout.count())):
            self.args_layout.itemAt(i).widget().deleteLater()

        task = None
        match self.dropdown.currentIndex():
            case 0:
                task = _get_task(v)
            case 1:
                task = _get_job(v)
            case 2:
                task = _get_service(v)
            case _:
                ...

        if task is None:
            return

        self.create_params[k] = v

        for k in task.params():
            row = LabelAndEdit(k, self.create_params.get(k, ''), self, self.line_edit_changed)
            self.args_layout.addWidget(row)

    def line_edit_changed(self, k, v):
        self.create_params[k] = v

    def create_type_changed(self, idx):
        for i in reversed(range(self.bottom_layout.count())):
            self.bottom_layout.itemAt(i).widget().deleteLater()

        match idx:
            case 0:
                script_row = LabelAndEdit('script', self.create_params.get('script', ''), self, self.script_name_changed)
                self.bottom_layout.addWidget(script_row)

                name_edit = LabelAndEdit('name', self.create_params.get('name', ''), self, self.line_edit_changed)
                self.bottom_layout.addWidget(name_edit)

                priority_edit = LabelAndEdit('priority', self.create_params.get('priority', ''), self, self.line_edit_changed)
                self.bottom_layout.addWidget(priority_edit)

                on_edit = LabelAndEdit('on', self.create_params.get('on', ''), self, self.line_edit_changed)
                self.bottom_layout.addWidget(on_edit)

                position_edit = LabelAndEdit('position', self.create_params.get('position', ''), self, self.line_edit_changed)
                self.bottom_layout.addWidget(position_edit)
            case 1:
                script_row = LabelAndEdit('script', self.create_params.get('script', ''), self, self.script_name_changed)
                self.bottom_layout.addWidget(script_row)

                name_edit = LabelAndEdit('name', self.create_params.get('name', ''), self, self.line_edit_changed)
                self.bottom_layout.addWidget(name_edit)

                priority_edit = LabelAndEdit('priority', self.create_params.get('priority', ''), self, self.line_edit_changed)
                self.bottom_layout.addWidget(priority_edit)

                every_edit = LabelAndEdit('every', self.create_params.get('every', ''), self, self.line_edit_changed)
                self.bottom_layout.addWidget(every_edit)

                position_edit = LabelAndEdit('position', self.create_params.get('position', ''), self, self.line_edit_changed)
                self.bottom_layout.addWidget(position_edit)
            case 2:
                script_row = LabelAndEdit('script', self.create_params.get('script', ''), self, self.script_name_changed)
                self.bottom_layout.addWidget(script_row)

                name_edit = LabelAndEdit('name', self.create_params.get('name', ''), self, self.line_edit_changed)
                self.bottom_layout.addWidget(name_edit)

                priority_edit = LabelAndEdit('priority', self.create_params.get('priority', ''), self, self.line_edit_changed)
                self.bottom_layout.addWidget(priority_edit)

                mode_edit = LabelAndEdit('mode', self.create_params.get('mode', ''), self, self.line_edit_changed)
                self.bottom_layout.addWidget(mode_edit)

                position_edit = LabelAndEdit('position', self.create_params.get('position', ''), self, self.line_edit_changed)
                self.bottom_layout.addWidget(position_edit)
            case _:
                ...


class RunWindow(QWidget):
    def __init__(self, api):
        super(RunWindow, self).__init__(None)

        self.api = api

        layout = QVBoxLayout()

        self.dropdown = QComboBox()
        self.dropdown.addItem('task')
        self.dropdown.addItem('job')
        self.dropdown.addItem('service')
        self.dropdown.currentIndexChanged.connect(self.create_type_changed)
        layout.addWidget(self.dropdown)

        self.bottom_layout = QVBoxLayout()

        # settings = json.loads(open('iva.json', 'r').read())
        # for task in settings.get('tasks', []):
        #     widget = QWidget()
        #     layout = QHBoxLayout()
        #     name = QLabel(task.get('name', ''))
        #     run_button = QPushButton('run')
        #     run_button.setEnabled(len(list(filter(lambda x: x == name, self.api.running_tasks.keys()))) == 0)
        #     run_button.clicked.connect(functools.partial(self.run, name.text()))
        #     layout.addWidget(name)
        #     layout.addWidget(run_button)
        #     widget.setLayout(layout)
        #     self.bottom_layout.addWidget(widget)

        layout.addLayout(self.bottom_layout)

        self.setLayout(layout)

    def create_type_changed(self, idx):
        for i in reversed(range(self.bottom_layout.count())):
            self.bottom_layout.itemAt(i).widget().deleteLater()

        match idx:
            case 0:
                settings = json.loads(open('iva.json', 'r').read())
                for task in settings.get('tasks', []):
                    widget = QWidget()
                    layout = QHBoxLayout()
                    name = QLabel(task.get('name', ''))
                    run_button = QPushButton('run')
                    run_button.setEnabled(len(list(filter(lambda x: x == name, self.api.running_tasks.keys()))) == 0)
                    run_button.clicked.connect(functools.partial(self.run, name.text()))
                    layout.addWidget(name)
                    layout.addWidget(run_button)
                    widget.setLayout(layout)
                    self.bottom_layout.addWidget(widget)
            case 1:
                settings = json.loads(open('iva.json', 'r').read())
                for job in settings.get('jobs', []):
                    widget = QWidget()
                    layout = QHBoxLayout()
                    name = QLabel(job.get('name', ''))
                    run_button = QPushButton('run')
                    run_button.setEnabled(len(list(filter(lambda x: x == name, self.api.running_jobs.keys()))) == 0)
                    run_button.clicked.connect(functools.partial(self.run, name.text()))
                    layout.addWidget(name)
                    layout.addWidget(run_button)
                    widget.setLayout(layout)
                    self.bottom_layout.addWidget(widget)
            case 2:
                settings = json.loads(open('iva.json', 'r').read())
                for service in settings.get('services', []):
                    widget = QWidget()
                    layout = QHBoxLayout()
                    name = QLabel(service.get('name', ''))
                    run_button = QPushButton('run')
                    run_button.setEnabled(len(list(filter(lambda x: x == name, self.api.running_services.keys()))) == 0)
                    run_button.clicked.connect(functools.partial(self.run, name.text()))
                    layout.addWidget(name)
                    layout.addWidget(run_button)
                    widget.setLayout(layout)
                    self.bottom_layout.addWidget(widget)
            case _:
                ...

    def run(self, name):
        match self.dropdown.currentIndex():
            case 0:
                self.api.run_task(name)
            case 1:
                self.api.run_job(name)
            case 2:
                self.api.start_service(name)
            case _:
                ...


class SettingsWindow(QWidget):
    ...


class Interpreter(QDialog):
    def __init__(self):
        settings = json.loads(open('iva.json', 'r').read())

        self.parent_os_path = os.path.abspath(os.path.curdir)
        self.os_path = os.sep

        # self.output_device = OutputDevice()
        # self.input_device = InputDriver()

        super(Interpreter, self).__init__(None)

        self.lock = RWLock()
        self.running_tasks = {}  # running, finished
        self.running_jobs = {}  # running, enabled, disabled, finished
        self.running_services = {}  # running, enabled, disabled, stopped, finished

        self.event = threading.Event()

        self.setFixedSize(1366, 768)
        self.move(-1643, 156)
        self.setFocus()

        # current datetime, create task/job/service button, run task/job/service button, settings button
        top_bar = QWidget()
        top_bar_layout = QHBoxLayout()

        create_button = QPushButton('Create')
        create_button.clicked.connect(self.show_create_window)
        top_bar_layout.addWidget(create_button)

        run_button = QPushButton('Run')
        run_button.clicked.connect(self.show_run_window)
        top_bar_layout.addWidget(run_button)

        settings_button = QPushButton('Settings')
        settings_button.clicked.connect(self.show_settings_window)
        top_bar_layout.addWidget(settings_button)

        dt = QLabel(datetime.datetime.now().isoformat())
        top_bar_layout.addWidget(dt)

        top_bar.setLayout(top_bar_layout)

        content_left_panel_layout = QVBoxLayout()

        content_left_panel_tab_widget_1 = QTabWidget()
        content_left_panel_layout.addWidget(content_left_panel_tab_widget_1)

        content_left_panel_tab_widget_2 = QTabWidget()
        content_left_panel_layout.addWidget(content_left_panel_tab_widget_2)

        content_left_panel_tab_widget_3 = QTabWidget()
        content_left_panel_layout.addWidget(content_left_panel_tab_widget_3)

        content_left_panel = QWidget()
        content_left_panel.setLayout(content_left_panel_layout)

        content_content_panel = QWidget()
        content_content_panel_layout = QVBoxLayout()
        content_content_panel_layout.addWidget(QLabel('2'))
        content_content_panel.setLayout(content_content_panel_layout)

        content_right_panel_layout = QVBoxLayout()

        content_right_panel_tab_widget_1 = QTabWidget()
        content_right_panel_layout.addWidget(content_right_panel_tab_widget_1)

        content_right_panel_tab_widget_2 = QTabWidget()
        content_right_panel_layout.addWidget(content_right_panel_tab_widget_2)

        content_right_panel_tab_widget_3 = QTabWidget()
        content_right_panel_layout.addWidget(content_right_panel_tab_widget_3)

        content_right_panel = QWidget()
        content_right_panel.setLayout(content_right_panel_layout)

        content = QWidget()
        content_layout = QHBoxLayout()
        content.setLayout(content_layout)

        content_layout.addWidget(content_left_panel, 1)
        content_layout.addWidget(content_content_panel, 4)
        content_layout.addWidget(content_right_panel, 1)

        # tooltip, current job running progress, notification
        bottom_bar = QWidget()
        bottom_bar_layout = QHBoxLayout()

        bottom_progress_bar = QProgressBar()
        bottom_progress_bar.setRange(0, 10000)
        bottom_progress_bar.setValue(0)
        bottom_bar_layout.addWidget(bottom_progress_bar)
        bottom_progress_bar = QProgressBar()
        bottom_progress_bar.setRange(0, 10000)
        bottom_progress_bar.setValue(0)
        bottom_bar_layout.addWidget(bottom_progress_bar)
        bottom_progress_bar = QProgressBar()
        bottom_progress_bar.setRange(0, 10000)
        bottom_progress_bar.setValue(0)
        bottom_bar_layout.addWidget(bottom_progress_bar)

        bottom_bar.setLayout(bottom_bar_layout)

        main_layout = QVBoxLayout()

        main_layout.addWidget(top_bar)
        main_layout.addWidget(content)
        main_layout.addWidget(bottom_bar)
        self.setLayout(main_layout)

        self.setWindowTitle("IVA")

        QApplication.setStyle(QStyleFactory.create('Fusion'))
        QApplication.setPalette(QApplication.palette())

        self.panels = {
            'l': {
                '1': content_left_panel_tab_widget_1,
                '2': content_left_panel_tab_widget_2,
                '3': content_left_panel_tab_widget_3,
            },
            'r': {
                '1': content_right_panel_tab_widget_1,
                '2': content_right_panel_tab_widget_2,
                '3': content_right_panel_tab_widget_3,
            },
            'm': {
                '1': content_content_panel_layout,
            },
            't': {
                '1': dt
            },
            'b': {
                '1': bottom_progress_bar,
            },
        }
        self.display_queue = queue.Queue()

        self.timer = QTimer()
        self.timer.start(1000)
        self.timer.timeout.connect(self.display)

        tasks = settings.get('tasks', [])
        for task in sorted(filter(lambda x: x['status'] and x['on'] == 'startup', tasks), key=lambda x: x['priority']):
            self.run_task(task['name'])  # need to do them in background
        jobs = settings.get('jobs', [])
        for job in sorted(filter(lambda x: x['status'] and x['every'] > 0, jobs), key=lambda x: x['priority']):
            self.run_job(job['name'])  # need to do them in background
        services = settings.get('services', [])
        for service in sorted(filter(lambda x: x['status'] and x['mode'] == 'automatic', services), key=lambda x: x['priority']):
            self.start_service(service['name'])  # need to do them in background

        self.cleaning_thread = threading.Thread(target=self.clean)
        self.cleaning_thread.start()
        self.job_thread = threading.Thread(target=self.run_jobs)
        self.job_thread.start()
        self.service_thread = threading.Thread(target=self.run_services)
        self.service_thread.start()

        self.do_action('ls .')
        self.do_action('dt')

    def show_create_window(self):
        self.create_window = CreateWindow()
        self.create_window.show()

    def show_run_window(self):
        self.run_window = RunWindow(self)
        self.run_window.show()

    def show_settings_window(self):
        self.settings_window = SettingsWindow()
        self.settings_window.show()

    def run_jobs(self):
        schedule.every(10).seconds.do(self.do_action, 'ls .')  # need to do them in background
        schedule.every(1).seconds.do(self.do_action, 'dt')  # need to do them in background

        settings = json.loads(open('iva.json', 'r').read())

        jobs = settings.get('jobs', {})
        for job in sorted(filter(lambda x: x['status'] and x['every'] > 0, jobs), key=lambda x: x['priority']):
            schedule.every(int(job['every'])).seconds.do(self.run_job, job['name'])  # need to do them in background

        while not self.event.is_set():
            schedule.run_pending()
            time.sleep(0.1)

        schedule.clear()

    def run_services(self):
        settings = json.loads(open('iva.json', 'r').read())

        services = settings.get('services', [])

        # if the service is closed for some reason and it is configured as restart automatically, need to restart the service

        while not self.event.is_set():
            for service in sorted(filter(lambda x: x['status'], services), key=lambda x: x['priority']):
                if service['name'] not in self.running_services or self.running_services[service['name']].state == ServiceState.finished:
                    self.start_service(service['name'])  # need to do them in background
            time.sleep(1)

    def clean(self):
        while not self.event.is_set():
            with self.lock.r_locked():
                task_keys = [key for key in self.running_tasks.keys()]
                job_keys = [key for key in self.running_jobs.keys()]
                service_keys = [key for key in self.running_services.keys()]

            delete_task_keys = []
            for key in task_keys:
                task_info = self.running_tasks[key]
                if not task_info.task.running():
                    delete_task_keys.append(key)
            delete_job_keys = []
            for key in job_keys:
                task_info = self.running_jobs[key]
                if not task_info.job.running():
                    delete_job_keys.append(key)
            delete_service_keys = []
            for key in service_keys:
                task_info = self.running_services[key]
                if not task_info.service.running() and task_info.state != ServiceState.stopped:
                    delete_service_keys.append(key)

            for key in delete_task_keys:
                with self.lock.w_locked():
                    del self.running_tasks[key]
            for key in delete_job_keys:
                with self.lock.w_locked():
                    del self.running_jobs[key]
            for key in delete_service_keys:
                with self.lock.w_locked():
                    del self.running_services[key]

            time.sleep(1)

    def display(self):
        while not self.display_queue.empty():
            location, name, text = self.display_queue.get_nowait()

            if location[0] in ('l', 'r'):
                tab = self.panels[location[0]][location[1]]

                index = tab.currentIndex()

                for idx in range(tab.count()):
                    if tab.tabText(idx) == name:
                        tab.removeTab(idx)

                        scroll = QScrollArea()

                        layout = QVBoxLayout()
                        for t in text:
                            layout.addWidget(QLabel(t))
                        content = QWidget()
                        content.setLayout(layout)

                        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
                        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                        # scroll.setWidgetResizable(True)
                        scroll.setWidget(content)

                        tab.insertTab(idx, scroll, name)
                        tab.setCurrentIndex(index)
                        break
                else:
                    scroll = QScrollArea()

                    layout = QVBoxLayout()
                    for t in text:
                        layout.addWidget(QLabel(t))
                    content = QWidget()
                    content.setLayout(layout)

                    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
                    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                    # scroll.setWidgetResizable(True)
                    scroll.setWidget(content)

                    tab.addTab(scroll, name)
            if location[0] in ('m',):
                layout = self.panels['m']['1']
                for i in reversed(range(layout.count())):
                    layout.itemAt(i).widget().deleteLater()
                for t in text:
                    layout.addWidget(QLabel(t))

            if location[0] in ('t',):
                label = self.panels['t']['1']
                t = ' '.join(text)
                label.setText(t)
            if location[0] in ('b',):
                ...

            time.sleep(0.1)

    def show_(self, location, name, text):
        # tab = self.panels[location[0]][location[1]]
        # tab.addTab(QLabel(text), name)
        self.display_queue.put_nowait((location, name, text))

    def listen(self):
        return self.input_device.listen()

    def say(self, text):
        self.output_device.say(text)

    def do_action(self, line):
        try:
            if line is None or line == '':
                return False

            action, *args = COMMA_MATCHER.split(line)

            if action is None or action == '':
                return False

            match action.lower():
                case 'enable':
                    return self.enable()
                case 'disable':
                    return self.disable()
                case 'create':
                    return self.create_()
                case 'update':
                    return self.update()
                case 'delete':
                    return self.delete()
                case 'configure':
                    return self.configure()
                case 'run':
                    return self.run()
                case 'start':
                    return self.start()
                case 'stop':
                    return self.stop()
                case 'restart':
                    return self.restart()
                case 'help':
                    return self.help()
                case 'exit':
                    return self.exit()
                case _:
                    parser = argparse.ArgumentParser()
                    _, extras = parser.parse_known_args(args)
                    # for k, v in zip(*(iter(extras),) * 2):

                    return self.run_task(action, [k for k in extras])
        except Exception as ex:
            print(str(ex))  # use stacktrace and write all exception details, line number, function name, file name etc.
            # return self.exit()

    def enable(self, *args):
        for arg in args:
            ...

        settings = json.loads(open('iva.json', 'r').read())
        open('iva.json', 'w').write(json.dumps(settings, indent=4))

        return False

    def disable(self, *args):
        for arg in args:
            ...

        settings = json.loads(open('iva.json', 'r').read())
        open('iva.json', 'w').write(json.dumps(settings, indent=4))

        return False

    def update(self):
        ...

    def delete(self):
        ...

    def configure(self):
        ...

    def run(self):
        self.output('what do you want me to run')
        action_type = self.input()

        if action_type.lower() == 'task':
            self.output('which task do you want me to run')
            task = self.input()
            return self.run_task(task)
        if action_type.lower() == 'job':
            self.output('which job do you want me to run')
            job = self.input()
            return self.run_job(job)

        return False

    def run_task(self, task_name, args=None):
        args = args or []

        settings = json.loads(open('iva.json', 'r').read())
        task_info = first(filter(lambda x: x['name'] == task_name, settings.get('tasks', [])))

        if task_info is None:
            task_info = {'script': task_name, 'args': {}}
            # return False

        script = task_info['script']

        task = _get_task(script)

        if task is None:
            self.output('task is not found')
            return False

        kwargs = task_info['args']
        kwargs['parent_os_path'] = self.parent_os_path
        kwargs['os_path'] = self.os_path

        task = task(self, *args, **kwargs)
        if task_name not in self.running_tasks:
            self.running_tasks[task_name] = TaskInfo(task_name, TaskState.running, task)
        else:
            self.running_tasks[task_name].state = TaskState.running
            self.running_tasks[task_name].task = task
        task.start()

        return False

    def run_job(self, job_name):
        settings = json.loads(open('iva.json', 'r').read())
        task_info = first(filter(lambda x: x['name'] == job_name, settings.get('jobs', [])))

        if task_info is None:
            task_info = {'script': job_name, 'args': {}}
            # return False

        script = task_info['script']

        task = _get_job(script)

        if task is None:
            self.output('job is not found')
            return False

        args = task_info['args']
        args['parent_os_path'] = self.parent_os_path
        args['os_path'] = self.os_path

        task = task(self, *[], **args)
        if job_name not in self.running_jobs:
            self.running_jobs[job_name] = JobInfo(job_name, JobState.running, task)
        else:
            self.running_jobs[job_name].state = JobState.running
            self.running_jobs[job_name].job = task
        task.start()

        return False

    def start(self):
        self.output('what service do you want me to start')
        action = self.input()
        return self.start_service(action)

    def start_service(self, service_name):
        settings = json.loads(open('iva.json', 'r').read())
        task_info = first(filter(lambda x: x['name'] == service_name, settings.get('services', [])))

        if task_info is None:
            task_info = {'script': service_name, 'args': {}}
            # return False

        script = task_info['script']

        task = _get_service(script)

        if task is None:
            self.output('service is not found')
            return False

        args = task_info['args']
        args['parent_os_path'] = self.parent_os_path
        args['os_path'] = self.os_path

        task = task(self, *[], **args)
        if service_name not in self.running_services:
            self.running_services[service_name] = ServiceInfo(service_name, ServiceState.running, task)
        else:
            self.running_services[service_name].state = ServiceState.running
            self.running_services[service_name].service = task
        task.start()

        return False

    def stop(self):
        self.output('what service do you want me to stop')
        action = self.input()
        return self.stop_service(action)

    def stop_service(self, service_name):
        self.running_services[service_name].state = ServiceState.stopped
        self.running_services[service_name].service.stop()
        return False

    def restart(self):
        self.output('what service do you want me to restart')
        action = self.input()
        return self.restart_service(action)

    def restart_service(self, service_name):
        self.stop_service(service_name)
        self.start_service(service_name)
        return False

    def config(self, *args):
        parser = argparse.ArgumentParser()
        parser.add_argument('--create', type=str)
        parser.add_argument('--update', type=str)
        parser.add_argument('--delete', type=str)
        parser.add_argument('--value', type=str)

        namespace, _ = parser.parse_known_args(args)

        if namespace.create:
            config = {
                'action': 'create',
                'name': namespace.create,
                'value': namespace.value
            }
        elif namespace.update:
            config = {
                'action': 'update',
                'name': namespace.update,
                'value': namespace.value
            }
        elif namespace.delete:
            config = {
                'action': 'delete',
                'name': namespace.delete
            }
        else:
            raise ValueError('')

        def set_config(parent, name, value):
            if name == '':
                return

            names = name.split('.')
            if names[0] not in parent:
                if len(names) == 1:
                    if value is not None:
                        parent[names[0]] = value
                    else:
                        del parent[names[0]]
                else:
                    parent[names[0]] = {}
            else:
                if len(names) == 1:
                    if value is not None:
                        parent[names[0]] = value
                    else:
                        del parent[names[0]]
            set_config(parent[names[0]], '.'.join(names[1:]), value)

        settings = json.loads(open('iva.json', 'r').read())
        configs = settings.get('configs', {})

        if config['action'] == 'create':
            set_config(configs, config['name'], config['value'])
            settings['configs'] = configs
            open('iva.json', 'w').write(json.dumps(settings, indent=4))
        elif config['action'] == 'update':
            set_config(configs, config['name'], config['value'])
            settings['configs'] = configs
            open('iva.json', 'w').write(json.dumps(settings, indent=4))
        elif config['action'] == 'delete':
            set_config(configs, config['name'], None)
            settings['configs'] = configs
            open('iva.json', 'w').write(json.dumps(settings, indent=4))
        else:
            raise ValueError(f'arguments are not recognized')

    def help(self):
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

        return False

    def exit(self):
        settings = json.loads(open('iva.json', 'r').read())

        tasks = settings.get('tasks', [])
        for task in sorted(filter(lambda x: x['status'] and x['on'] == 'shutdown', tasks), key=lambda x: x['priority']):
            self.run_task(task['name'])

        with self.lock.r_locked():
            task_keys = [key for key in self.running_tasks.keys()]
            job_keys = [key for key in self.running_jobs.keys()]
            service_keys = [key for key in self.running_services.keys()]

        for key in task_keys:
            task_info = self.running_tasks[key]
            task_info.task.stop()
        for key in job_keys:
            task_info = self.running_jobs[key]
            task_info.job.stop()
        for key in service_keys:
            self.stop_service(key)

        self.event.set()
        self.input_device.stop()
        return True


class App(QApplication):
    def __init__(self):
        super(App, self).__init__(sys.argv)

        self.setQuitOnLastWindowClosed(False)
        self.lastWindowClosed.connect(self.on_last_closed)

        self.mainwins = []
        win = Interpreter()
        win.show()
        self.mainwins.append(win)

    def on_last_closed(self):
        self.mainwins[0].exit()
        self.exit()


def main():
    app = App()
    # gallery = Interpreter()
    # gallery.show()
    sys.exit(app.exec())
