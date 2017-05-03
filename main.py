import glob
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from PIL import Image

from models.generativeAdversarial import GAN_ISR_Model

def get_inputs(size_x=(256,256), size_y=(64,64), limit=None):
    file_path = '../data/flowers/*.jpg'
    files = glob.glob(file_path) if limit is None else glob.glob(file_path)[:limit]
    def channels_last_data_format(img):
        return np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)
    def process_img_file(file_name):
        image = Image.open(file_name).resize(size_x, Image.BICUBIC)
        image_lowres = (image.resize(size_y)).resize(size_x, Image.BICUBIC)
        return np.array(image), np.array(image_lowres)
    return zip(*map(process_img_file, files))

orig, small = get_inputs()
model = GAN_ISR_Model(orig)
model.train()
