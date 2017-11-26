import os

URL_PAINTINGS = './paintings'

NUM_PAINTERS = 2


class DataProvider:
    @classmethod
    def get_data(cls):
        data = cls._map_painters_paintings()
        return data[:NUM_PAINTERS]

    @classmethod
    def _map_painters_paintings(cls):
        painters = cls._get_painters()
        map_pp = [cls._get_paintings(painter) for painter in painters]
        return sorted(map_pp, key=lambda pair: len(pair[1]), reverse=True)

    @classmethod
    def _get_painters(cls):
        return os.listdir(URL_PAINTINGS)

    @classmethod
    def _get_paintings(cls, painter):
        base_url_rel = os.path.join(URL_PAINTINGS, painter)
        base_url_abs = os.path.abspath(base_url_rel)
        return [os.path.join(base_url_abs, file) for file in os.listdir(base_url_abs)]
