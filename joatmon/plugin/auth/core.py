from joatmon.orm.enum import Enum
from joatmon.plugin.core import Plugin


class AuthStatusCode(Enum):
    OK = 0

    USER_ALREADY_EXISTS = 1
    USER_DOES_NOT_EXISTS = 2
    USER_ALREADY_LOGGED_IN = 3
    USER_TPA_REQUIRED = 4
    USER_ALREADY_LOGGED_OUT = 5

    TOKEN_DECODE_ERROR = 6
    TOKEN_EXPIRED = 7
    TOKEN_NOT_VALID = 8
    TOKEN_USER_NOT_VALID = 9
    TOKEN_USER_NOT_LOGGED_IN = 10
    TOKEN_MIN_ROLE_NONE = 11
    TOKEN_NOT_ALLOWED = 12

    AUTH_NOT_ALLOWED = 13

    LOG_LEVEL_NOT_FOUND = 14
    TOO_MANY_REQUEST = 15

    UNKNOWN_ERROR = 1 << 31


class Auth(Plugin):
    async def authenticate(self, user_id, issuer, audience, expire_at):
        raise NotImplementedError

    async def authorize(self, token, issuer, audience, min_role):
        raise NotImplementedError
