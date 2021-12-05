import inspect

from flask import (
    Blueprint,
    jsonify,
    make_response,
    redirect,
    render_template
)

from joatmon.web.core import WebException


class ControllerException(WebException):
    pass


class APIControllerException(ControllerException):
    pass


class WebControllerException(ControllerException):
    pass


class Controller:
    ...


class APIController(Controller):  # we can get rid of class attributes now, they can be parameter
    def __init__(self, name, version, latest=False):
        self.blueprints = []
        if latest:
            self.blueprints.append(Blueprint(f'{name}', name, url_prefix=f'/api/{name}'))
        self.blueprints.append(Blueprint(f'{version}/{name}', name, url_prefix=f'/api/{version}/{name}'))

        functions = inspect.getmembers(type(self), inspect.isfunction)
        functions = list(filter(lambda x: not x[0].startswith('_'), functions))

        for function in functions:
            current_function = getattr(self, function[0])

            name = function[0].replace('_', '-')
            access = function[1].access  # access might not exist
            method = function[1].method  # method might not exist
            if access in ['private', 'protected']:
                continue

            for blueprint in self.blueprints:
                blueprint.route('/')

                if method.lower() in ['put', 'delete']:  # maybe remove first parameter
                    first_parameter_name = list(filter(lambda x: x != 'self', inspect.getfullargspec(function[1])[0]))[0]
                    blueprint.route(f'/{name}/<{first_parameter_name}>', methods=[method])(current_function)
                else:
                    blueprint.route(f'/{name}', methods=[method])(current_function)


class WebController(Controller):
    def __init__(self, name):
        self.blueprint = Blueprint(f'{name}', name, url_prefix=f'/{name}')

        functions = inspect.getmembers(type(self), inspect.isfunction)
        functions = list(filter(lambda x: not x[0].startswith('_'), functions))

        for function in functions:
            if function[0] in ['render', 'response', 'json', 'redirect']:
                continue

            current_function = getattr(self, function[0])

            name = function[0].replace('_', '-')
            access = function[1].access  # access might not exist
            method = function[1].method  # method might not exist
            if access in ['private', 'protected']:
                continue

            if method.lower() in ['put', 'delete']:  # maybe remove first parameter
                first_parameter_name = list(filter(lambda x: x != 'self', inspect.getfullargspec(function[1])[0]))[0]
                self.blueprint.route(f'/{name}/<{first_parameter_name}>', methods=[method])(current_function)
            else:
                self.blueprint.route(f'/{name}', methods=[method])(current_function)

    def render(self, template, **model):
        return render_template(template, **model)

    def response(self, string):
        return make_response(string)

    def json(self, data):
        return jsonify(data)

    def redirect(self, url):
        return redirect(url)
