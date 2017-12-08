import os

from PIL import Image

from config import IMAGE_DIM, NUM_PAINTERS, URL_PAINTINGS


class DataProvider:
    @classmethod
    def get_data(cls):
        return cls._map_painters_paintings()

    @classmethod
    def _map_painters_paintings(cls):
        painters = cls._get_painters()
        painters_set = painters[:NUM_PAINTERS]
        map_pp = [(painter, cls._get_paintings(painter)) for painter in painters_set]
        return sorted(map_pp, key=lambda pair: len(pair[1]), reverse=True)

    @classmethod
    def _get_painters(cls):
        return os.listdir(URL_PAINTINGS)

    @classmethod
    def _get_paintings(cls, painter):
        filepaths = cls._get_filepaths(painter)
        images = cls._get_images_cropped(filepaths)
        return images

    @classmethod
    def _get_filepaths(cls, painter):
        base_url_rel = os.path.join(URL_PAINTINGS, painter)
        base_url_abs = os.path.abspath(base_url_rel)
        return [os.path.join(base_url_abs, file) for file in os.listdir(base_url_abs)]

    @classmethod
    def _get_images_cropped(cls, filepaths):
        try:
            images = [Image.open(path) for path in filepaths]
            [img.thumbnail(IMAGE_DIM) for img in images]
            return images
        except IOError as e:
            print(f'Could not load or crop image. Error: {e}')
