import jwt
from joatmon.core.exception import CoreException

from joatmon.plugin.auth.core import Auth


class JWTAuth(Auth):
    """
    JWTAuth class that inherits from the Auth class. It implements the abstract methods of the Auth class
    using JSON Web Tokens (JWT) for authentication and authorization.

    Attributes:
        secret (str): The secret key used for encoding and decoding the JWT.

    """

    def __init__(self, secret: str):
        """
        Initialize JWTAuth with the given secret key.

        Args:
            secret (str): The secret key used for encoding and decoding the JWT.
        """
        self.secret = secret

    async def authenticate(self, issuer, audience, expire_at, **kwargs):
        """
        Authenticate a user by encoding a JWT with the given parameters.

        Args:
            issuer (str): The issuer of the authentication.
            audience (str): The audience of the authentication.
            expire_at (datetime): The expiration date of the authentication.
            **kwargs: Additional claims to be included in the JWT.

        Returns:
            str: The encoded JWT.
        """
        kwargs.update({'exp': expire_at, 'iss': issuer, 'aud': audience})

        token = jwt.encode(
            payload=kwargs,
            key=self.secret,
            algorithm='HS256',
        )
        return token

    async def authorize(self, token, issuer, audience):
        """
        Authorize a user by decoding the given JWT and verifying the issuer and audience.

        Args:
            token (str): The JWT to be decoded.
            issuer (str): The expected issuer of the JWT.
            audience (str): The expected audience of the JWT.

        Returns:
            dict: The decoded JWT claims.

        Raises:
            CoreException: If the JWT cannot be decoded or the issuer and audience do not match the expected values.
        """
        try:
            decoded = jwt.decode(token, self.secret, issuer=issuer, audience=audience, algorithms='HS256')
        except (jwt.PyJWTError, ValueError):
            raise CoreException('not_authorized')

        return decoded
