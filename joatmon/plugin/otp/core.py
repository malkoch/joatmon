from joatmon.plugin.core import Plugin


class OTP(Plugin):
    async def get_qr(self, secret, name, issuer):
        raise NotImplementedError

    async def verify(self, secret, otp):
        raise NotImplementedError
