import hashlib
import math
import os
import shutil

import config


class DataSelector:
    @classmethod
    def select_data(cls):
        cls._remove_data_folder()

        map_painters_paintings = cls._get_painters_map()
        map_pp_reduced = cls._reduce_data(map_painters_paintings)
        data_train, data_val = cls._split_data(map_pp_reduced)

        cls._move_to_data(config.FOLDER_NAME_TRAINING, data_train)
        cls._move_to_data(config.FOLDER_NAME_VALIDATION, data_val)

    @classmethod
    def _remove_data_folder(cls):
        if os.path.exists(config.URL_DATA):
            shutil.rmtree(config.URL_DATA)

    @classmethod
    def _get_painters_map(cls):
        painters = cls._get_painters()
        map_pp = [(p, cls._get_paintings(p)) for p in painters]
        map_pp_sorted = sorted(map_pp, key=lambda pair: len(pair[1]), reverse=True)
        return map_pp_sorted

    @classmethod
    def _get_painters(cls):
        return os.listdir(config.URL_PAINTINGS)

    @classmethod
    def _get_paintings(cls, painter):
        base_url_rel = os.path.join(config.URL_PAINTINGS, painter)
        base_url_abs = os.path.abspath(base_url_rel)
        return [os.path.join(base_url_abs, file) for file in os.listdir(base_url_abs)]

    @classmethod
    def _reduce_data(cls, data):
        return [(painter, paintings[:config.NUM_PAINTINGS])
                for (painter, paintings)
                in data[:config.NUM_PAINTERS]]

    @classmethod
    def _split_data(cls, data):
        num_train = math.ceil(config.NUM_PAINTINGS * config.PCT_TRAINING)
        num_val = math.floor(config.NUM_PAINTINGS * (1 - config.PCT_TRAINING))

        data_train = [(painter, paintings[:num_train]) for (painter, paintings) in data]
        data_val = [(painter, paintings[num_val:]) for (painter, paintings) in data]

        return data_train, data_val

    @classmethod
    def _move_to_data(cls, folder_name, data):
        path_base = os.path.join(config.URL_DATA, folder_name)

        for (painter, paintings) in data:
            unique_name = hashlib.sha1(str.encode(painter)).hexdigest()
            path_painter = os.path.join(path_base, unique_name)
            cls._create_folder(path_painter)
            cls._move_to_(path_painter, paintings)

    @classmethod
    def _create_folder(cls, path):
        if not os.path.exists(path):
            os.makedirs(path)

    @classmethod
    def _move_to_(cls, dst_path, data):
        [shutil.copy(path, dst_path) for path in data]


if __name__ == '__main__':
    DataSelector.select_data()
