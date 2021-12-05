from joatmon.core import CoreException
from joatmon.plugin.core import Plugin


class AuthorizerException(CoreException):
    ...


class Auth(Plugin):
    def __init__(self, alias: str):
        super(Auth, self).__init__(alias)

        self.alias = alias

    def authenticate(self, username, password, issuer, audience, expire_at):
        raise NotImplementedError

    def authorize(self, token, issuer, audience, min_role):
        raise NotImplementedError
