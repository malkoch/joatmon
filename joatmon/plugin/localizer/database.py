from joatmon.core import context
from joatmon.plugin.localizer.core import Localizer
from joatmon.core.utility import (
    new_object_id,
    to_list_async
)


class DatabaseLocalizer(Localizer):
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

    def __init__(self, database, cls):
        self.database = database
        self.cls = cls

    async def localize(self, language, keys):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
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
