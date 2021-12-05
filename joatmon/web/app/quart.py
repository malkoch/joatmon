import quart

from joatmon.context import (
    current,
    initialize_context,
    teardown_context
)
from joatmon.web.core import WebException
from joatmon.web.app import core


class ApplicationError(WebException):
    pass


class AppWrapper(object):
    def __init__(self, application, prefix=''):
        self.app = application
        self.prefix = prefix

    async def __call__(self, scope, receive, send):
        if scope['PATH_INFO'].lower().startswith(self.prefix.lower()):
            scope['PATH_INFO'] = scope['PATH_INFO'][len(self.prefix):].lower()
            scope['SCRIPT_NAME'] = self.prefix.lower()
            return await self.app(scope, receive, send)
        else:
            send('404', [('Content-Type', 'text/plain')])
            return ["This url does not belong to the app.".encode()]


class APIApplication(core.APIApplication):
    def __init__(self, name):
        self.app = quart.Quart(name)

        self.app.asgi_app = AppWrapper(self.app.asgi_app, prefix=f'/{name}.Api')

        self.app.before_request(self.before_request)
        self.app.after_request(self.after_request)
        self.app.route('/')(self.index)

    def add_controller(self, api):
        for blueprint in api.blueprints:
            self.app.register_blueprint(blueprint)

    async def before_request(self):
        initialize_context()
        current['ip'] = quart.request.scope.get('HTTP_X_FORWARDED_FOR') or quart.request.scope.get('REMOTE_ADDR')
        current['headers'] = quart.request.headers
        current['json'] = quart.request.json
        current['args'] = quart.request.args

    async def after_request(self, response):
        teardown_context()

        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response

    async def index(self):
        return f'Welcome to {self.app.name}'

    async def __call__(self, scope, receive, send):
        return await self.app.asgi_app(scope, receive, send)


class WebApplication(core.WebApplication):
    def __init__(self, name):
        self.app = quart.Quart(name)

        self.app.asgi_app = AppWrapper(self.app.asgi_app, prefix=f'/{name}.Api')

        self.app.before_request(self.before_request)
        self.app.after_request(self.after_request)
        self.app.route('/')(self.index)

    def add_controller(self, api):
        self.app.register_blueprint(api.blueprint)

    async def before_request(self):
        initialize_context()
        current['ip'] = quart.request.scope.get('HTTP_X_FORWARDED_FOR') or quart.request.scope.get('REMOTE_ADDR')
        current['headers'] = quart.request.headers
        current['json'] = quart.request.json
        current['args'] = quart.request.args

    async def after_request(self, response):
        teardown_context()

        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response

    async def index(self):
        return f'Welcome to {self.app.name}'

    async def __call__(self, scope, receive, send):
        return await self.app.asgi_app(scope, receive, send)
