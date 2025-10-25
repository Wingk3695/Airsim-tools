import iniconfig
def load_config(config_path):
    config = iniconfig.IniConfig(config_path)
    return config

