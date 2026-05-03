from joatmon.plugin.core import Plugin


class ObjectStorage(Plugin):
    """
    ObjectStorage class that inherits from the Plugin class. It provides the functionality for generating and verifying ObjectStorages.

    Methods:
        get_qr: Generates a QR code for the OTP.
        verify: Verifies the OTP.
    """

    async def put_object(self, name, content, type_):
        """
        Generates a QR code for the OTP.

        This method should be implemented in the child classes.

        Args:
            name (str): The name of the OTP.
            content (bytes): The content of the object.
            type_ (str): The content type of the object.
        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """
        raise NotImplementedError

    async def get_object(self, name):
        """
        Verifies the OTP.

        This method should be implemented in the child classes.

        Args:
            name (str): The name of the OTP.
        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """
        raise NotImplementedError
