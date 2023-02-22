import jwt

from joatmon.plugin.auth.core import Auth, AuthStatusCode
from joatmon.plugin.core import PluginResponse


class JWTAuth(Auth):
    def __init__(self, secret: str):
        self.secret = secret

    async def authenticate(self, role, issuer, audience, expire_at):
        token = jwt.encode(
            payload={
                'role': role,
                'exp': expire_at,
                'iss': issuer,
                'aud': audience
            },
            key=self.secret,
            algorithm='HS256',
        )
        return token

    async def authorize(self, token, issuer, audience, min_role):
        try:
            decoded = jwt.decode(token, self.secret, issuer=issuer, audience=audience, algorithms='HS256')
        except jwt.DecodeError:
            return PluginResponse(data=None, code=AuthStatusCode.TOKEN_DECODE_ERROR)
        except jwt.ExpiredSignatureError:
            return PluginResponse(data=None, code=AuthStatusCode.TOKEN_EXPIRED)
        except ValueError:
            return PluginResponse(data=None, code=AuthStatusCode.TOKEN_DECODE_ERROR)

        role = decoded.get('role', None)

        if role < min_role:
            return PluginResponse(data=None, code=AuthStatusCode.TOKEN_NOT_ALLOWED)

        return PluginResponse(data=True, code=AuthStatusCode.OK)  # can be used on context.user
