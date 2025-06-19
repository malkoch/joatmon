import base64
from io import BytesIO

import pyotp
import qrcode

from joatmon.plugin.otp.core import OTP


class TOTP(OTP):
    """
    TOTP class that inherits from the OTP class. It provides the functionality for generating and verifying Time-based One-Time Passwords (TOTP).

    Methods:
        get_qr: Generates a QR code for the TOTP.
        verify: Verifies the TOTP.
    """

    async def get_qr(self, secret, name, issuer):
        """
        Generates a QR code for the TOTP.

        This method uses the pyotp library to generate a provisioning URI for the TOTP. This URI can be converted into a QR code.

        Args:
            secret (str): The secret key for the TOTP.
            name (str): The name of the TOTP.
            issuer (str): The issuer of the TOTP.

        Returns:
            str: The provisioning URI for the TOTP.
        """
        uri = pyotp.totp.TOTP(secret).provisioning_uri(name=name, issuer_name=issuer)
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(uri)
        qr.make(fit=True)
        img = qr.make_image(fill_color='black', back_color='white')
        buffered = BytesIO()
        img.save(buffered)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    async def verify(self, secret, otp):
        """
        Verifies the TOTP.

        This method uses the pyotp library to verify the TOTP.

        Args:
            secret (str): The secret key for the TOTP.
            otp (str): The TOTP to be verified.

        Returns:
            bool: True if the TOTP is valid, False otherwise.
        """
        return pyotp.TOTP(secret).verify(otp)
