import jwt

from joatmon.plugin.auth.core import Auth


class JWTAuth(Auth):
    def __init__(self, secret: str):
        self.secret = secret

    async def authenticate(self, issuer, audience, expire_at, **kwargs):
        kwargs.update({'exp': expire_at, 'iss': issuer, 'aud': audience})

        token = jwt.encode(
            payload=kwargs,
            key=self.secret,
            algorithm='HS256',
        )
        return token

    async def authorize(self, token, issuer, audience):
        try:
            decoded = jwt.decode(token, self.secret, issuer=issuer, audience=audience, algorithms='HS256')
        except jwt.DecodeError:
            raise ValueError('token_decode_error')
        except jwt.ExpiredSignatureError:
            raise ValueError('token_expired')
        except ValueError:
            raise ValueError('token_decode_error')

        return decoded
