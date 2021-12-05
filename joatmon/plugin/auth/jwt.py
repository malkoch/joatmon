import jwt

from joatmon.core import CoreException
from joatmon.database.model.user import OUser, OUserLoginHistory, DUserType, EUserType
from joatmon.plugin.auth.core import Auth
from joatmon.plugin.core import create
from joatmon.utility import first


class JWTAuth(Auth):
    def __init__(self, alias: str, secret: str, database: str):
        super(JWTAuth, self).__init__(alias)

        self.secret = secret
        self.database = database

    def authenticate(self, username, password, issuer, audience, expire_at):
        token = jwt.encode(
            payload={
                'nickname': username,
                'password': password,
                'exp': expire_at,
                'iss': issuer,
                'aud': audience
            },
            key=self.secret,
            algorithm='HS256',
        )
        return token

    def authorize(self, token, issuer, audience, min_role):
        try:
            decoded = jwt.decode(token, self.secret, issuer=issuer, audience=audience, algorithms='HS256')
        except jwt.DecodeError:
            raise CoreException('token_decode_error')
        except jwt.ExpiredSignatureError:
            raise CoreException('token_expired')
        except ValueError:
            raise CoreException('token_decode_error')

        nickname = decoded.get('nickname', None)
        password = decoded.get('password', None)

        if nickname is None or password is None:
            raise CoreException('authorization_header_not_valid')

        with create(self.database) as db:
            db_user = first(db.read(OUser, nickname=nickname, password=password, is_deleted=False))

            if db_user is None:
                raise CoreException('user_information_not_valid')

            # make sure user is logged in
            if first(db.read(OUserLoginHistory, user_id=db_user.object_id, logout_at=None)) is None:
                raise CoreException('user_not_logged_in')

            if min_role is None:
                raise CoreException('min_role_none')

            type_code = first(db.read(DUserType, object_id=db_user.type_id))
            if EUserType(type_code.code) < min_role:
                raise CoreException('not_allowed')

            return db_user  # can be used on context.user
