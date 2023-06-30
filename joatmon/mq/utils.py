import logging

import yaml

logger = logging.getLogger(__name__)


def not_in_dict_or_none(dictionary, key):
    if key not in dictionary or dictionary[key] is None:
        return True
    else:
        return False


def format_client_message(session=None, address=None, port=None):
    if session:
        return "(client id=%s)" % session.client_id
    elif address is not None and port is not None:
        return "(client @=%s:%d)" % (address, port)
    else:
        return "(unknown client)"


def gen_client_id():
    import random
    gen_id = 'hzmqtt/'

    for i in range(7, 23):
        gen_id += chr(random.randint(0, 74) + 48)
    return gen_id


def read_yaml_config(config_file):
    config = None
    try:
        with open(config_file, 'r') as stream:
            config = yaml.full_load(stream) if hasattr(yaml, 'full_load') else yaml.load(stream)
    except yaml.YAMLError as exc:
        logger.error("Invalid config_file %s: %s" % (config_file, exc))
    return config
