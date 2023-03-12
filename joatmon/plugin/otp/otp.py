import pyotp

from joatmon.plugin.otp.core import OTP


class TOTP(OTP):
    async def get_qr(self, secret, name, issuer):
        return pyotp.totp.TOTP(secret).provisioning_uri(name=name, issuer_name=issuer)

    async def verify(self, secret, otp):
        return pyotp.TOTP(secret).verify(otp)
