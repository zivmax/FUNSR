from pyhocon import ConfigFactory
import os


class Config:
    def __init__(self, conf_path, data_name=None):
        self.conf_path = conf_path
        self.conf = ConfigFactory.parse_file(self.conf_path)
        if data_name:
            self.conf["dataset.np_data_name"] = data_name + ".pt"

    def get_string(self, path):
        return self.conf.get_string(path)

    def get_int(self, path):
        return self.conf.get_int(path)

    def get_float(self, path):
        return self.conf.get_float(path)

    def get_bool(self, path):
        return self.conf.get_bool(path)

    def get_list(self, path):
        return self.conf.get_list(path)

    def get_config(self, path):
        return self.conf.get_config(path)
