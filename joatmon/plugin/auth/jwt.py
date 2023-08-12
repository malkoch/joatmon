import jwt

from joatmon.plugin.auth.core import Auth


class JWTAuth(Auth):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, secret: str):
        self.secret = secret

    async def authenticate(self, issuer, audience, expire_at, **kwargs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
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
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        try:
            decoded = jwt.decode(token, self.secret, issuer=issuer, audience=audience, algorithms='HS256')
        except jwt.DecodeError:
            raise ValueError('token_decode_error')
        except jwt.ExpiredSignatureError:
            raise ValueError('token_expired')
        except ValueError:
            raise ValueError('token_decode_error')

        return decoded
