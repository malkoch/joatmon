import jwt

from joatmon.plugin.auth.core import Auth


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
            raise ValueError('token_decode_error')
        except jwt.ExpiredSignatureError:
            raise ValueError('token_expired')
        except ValueError:
            raise ValueError('token_decode_error')

        role = decoded.get('role', None)

        if role < min_role:
            raise ValueError('token_not_allowed')

        return True
