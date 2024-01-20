from joatmon.core import context
from joatmon.plugin.localizer.core import Localizer
from joatmon.core.utility import (
    new_object_id,
    to_list_async
)


class FileLocalizer(Localizer):
    """
    FileLocalizer class that inherits from the Localizer class. It implements the abstract methods of the Localizer class
    using a file for localization operations.

    Attributes:
        database (str): The name of the database to be used for localization.
        cls (str): The class of the documents to be localized.
    """

    def __init__(self, database, cls):
        """
        Initialize FileLocalizer with the given database and class.

        Args:
            database (str): The name of the database to be used for localization.
            cls (str): The class of the documents to be localized.
        """
        self.database = database
        self.cls = cls

    async def localize(self, language, keys):
        """
        Localize a set of keys to a specified language using a file.

        This method reads the keys from the file and localizes them to the specified language. If a key is not found in the file,
        it is added with its localized value being the same as the key. If a key is found but does not have a localized value for the
        specified language, its localized value is set to the key.

        Args:
            language (str): The language to which the keys should be localized.
            keys (list): The keys to be localized.

        Returns:
            list: The localized keys.
        """
        database = context.get_value(self.database)

        db_resources = await to_list_async(database.read(self.cls, {'key': {'$in': keys}}))
        found_keys = list(map(lambda x: x.key, db_resources))
        not_found_keys = list(filter(lambda x: x not in found_keys, keys))

        for not_found_key in not_found_keys:
            r = {'object_id': new_object_id(), 'key': not_found_key, language: not_found_key}

            await database.insert(self.cls, r)

        for found_key in found_keys:
            db_resource = list(filter(lambda x: x.key == found_key, db_resources))[0]
            if getattr(db_resource, language, None) is None:
                setattr(db_resource, language, db_resource.key)
                await database.update(self.cls, {'object_id': db_resource.object_id}, db_resource)

        db_resources = await to_list_async(database.read(self.cls, {'key': {'$in': keys}}))
        return db_resources
        # return list(map(lambda x: x[language], db_resources))
