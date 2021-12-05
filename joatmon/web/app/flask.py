import flask

from joatmon.context import (
    current,
    initialize_context,
    teardown_context
)
from joatmon.utility import (
    JSONDecoder,
    JSONEncoder
)
from joatmon.web.app import core
from joatmon.web.core import WebException


class ApplicationError(WebException):
    pass


class AppWrapper(object):
    def __init__(self, application, prefix=''):
        self.app = application
        self.prefix = prefix

    def __call__(self, environ, start_response):
        if environ['PATH_INFO'].lower().startswith(self.prefix.lower()):
            environ['PATH_INFO'] = environ['PATH_INFO'][len(self.prefix):].lower()
            environ['SCRIPT_NAME'] = self.prefix.lower()
            return self.app(environ, start_response)
        else:
            start_response('404', [('Content-Type', 'text/plain')])
            return ["This url does not belong to the app.".encode()]


class APIApplication(core.APIApplication):
    def __init__(self, name):
        self.app = flask.Flask(name.replace('.', '_'))
        self.app.json_encoder = JSONEncoder
        self.app.json_decoder = JSONDecoder

        self.app.wsgi_app = AppWrapper(self.app.wsgi_app, prefix=f'/{name}.Api')

        self.app.before_request(self.before_request)
        self.app.after_request(self.after_request)
        self.app.route('/')(self.index)

    def add_controller(self, api):
        for blueprint in api.blueprints:
            self.app.register_blueprint(blueprint)

    def before_request(self):
        initialize_context()
        current['token'] = flask.request.headers.get('Authorization', None)
        current['issuer'] = flask.request.headers.get('Issuer', None)
        current['ip'] = flask.request.environ.get('HTTP_X_FORWARDED_FOR', None) or flask.request.environ.get('REMOTE_ADDR', None)
        current['cookies'] = flask.request.cookies or {}
        current['headers'] = flask.request.headers or {}
        current['json'] = flask.request.json or {}
        current['args'] = flask.request.args or {}

    def after_request(self, response):
        teardown_context()

        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response

    def index(self):
        return f'Welcome to {self.app.name}'

    def __call__(self, environ, start_response):
        return self.app.wsgi_app(environ, start_response)


class WebApplication(core.WebApplication):
    def __init__(self, name):
        self.app = flask.Flask(name)
        self.app.json_encoder = JSONEncoder
        self.app.json_decoder = JSONDecoder

        self.app.wsgi_app = AppWrapper(self.app.wsgi_app, prefix=f'/{name}.Web')

        self.app.before_request(self.before_request)
        self.app.after_request(self.after_request)

    def add_controller(self, api):
        self.app.register_blueprint(api.blueprint)

    def before_request(self):
        initialize_context()
        current['ip'] = flask.request.environ.get('HTTP_X_FORWARDED_FOR') or flask.request.environ.get('REMOTE_ADDR')
        current['headers'] = flask.request.headers or {}
        current['json'] = flask.request.json or {}
        current['args'] = flask.request.args or {}
        current['form'] = flask.request.form or {}
        current['cookies'] = flask.request.cookies or {}

    def after_request(self, response):
        teardown_context()

        return response

    def __call__(self, environ, start_response):
        return self.app.wsgi_app(environ, start_response)
