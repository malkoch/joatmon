from joatmon.plugin.core import Plugin


class Auth(Plugin):
    async def authenticate(self, issuer, audience, expire_at):
        raise NotImplementedError

    async def authorize(self, token, issuer, audience, min_role):
        raise NotImplementedError
