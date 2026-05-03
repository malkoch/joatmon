import io

from minio import Minio

from joatmon.plugin.storage.core import ObjectStorage


class Mio(ObjectStorage):
    """
    Mio class that inherits from the ObjectStorage class.

    Methods:
        get_qr: Generates a QR code for the Mio.
        verify: Verifies the Mio.
    """

    def __init__(self, host, port, access_key, secret_key, bucket):
        self.host = host
        self.port = port
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket = bucket

        self.connection = Minio(f'{self.host}:{self.port}', self.access_key, self.secret_key, secure=False)

    async def put_object(self, name, content, type_):
        """
        Generates a QR code for the TOTP.

        This method uses the pyotp library to generate a provisioning URI for the TOTP. This URI can be converted into a QR code.

        Args:
            name (str): The name of the object.
            content (bytes): The content to be stored in the object.
        Returns:
            str: The provisioning URI for the TOTP.
        """
        stream = io.BytesIO(content)

        self.connection.put_object(
            self.bucket,
            name,
            stream,
            length=len(content),
            content_type=type_
        )

    async def get_object(self, name):
        """
        Verifies the TOTP.

        This method uses the pyotp library to verify the TOTP.

        Args:
            name (str): The name of the object.
        Returns:
            bool: True if the TOTP is valid, False otherwise.
        """
        obj = self.connection.get_object(self.bucket, name)
        return obj.read()
