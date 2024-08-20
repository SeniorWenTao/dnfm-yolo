import os

_current = os.path.abspath(__file__)
_BASE_DIR = os.path.dirname(os.path.dirname(_current))
_config_path = _BASE_DIR + os.sep + "config"


def get_base_path():
    return _BASE_DIR


def get_config_path():
    return _config_path
