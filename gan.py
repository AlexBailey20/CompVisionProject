import glob, sys
import numpy as np
from argparse import ArgumentParser
from scipy.ndimage.filters import gaussian_filter
from PIL import Image

from models.generativeAdversarial import GAN_Model

def get_inputs(size_x=(256,256), size_y=(64,64), limit=None, resize=False):
    file_path='../data/flowers/*.jpg'
    files = glob.glob(file_path)
    if limit is not None: files = files[:limit]
    def channels_last_data_format(img):
        return np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)
    def process_img_file(file_name):
        image, image_lowres = Image.open(file_name).resize(size_x, Image.BICUBIC), None
        if resize:
            image_lowres = (image.resize(size_y)).resize(size_x, Image.BICUBIC)
        return np.array(image), np.array(image_lowres) if image_lowres is not None else None
    return zip(*map(process_img_file, files))

def main(num_samples=None, epochs=None, skip=False, resume=False, limit=None):
    if skip:
        model = GAN_Model()
    else:
        images, _ = get_inputs(limit=limit)
        model = GAN_Model(images)
    if epochs is not None:
        model.train(epochs, resume)
    if num_samples is not None:
        model.generate(num_samples)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("-f", "--skipimages", action="store_true")
    parser.add_argument("-s", "--sample", type=int, default=None)
    parser.add_argument("-t", "--train", type=int, default=None)
    args = parser.parse_args()
    main(num_samples=args.sample, limit=args.limit, epochs=args.train, skip=args.skipimages, resume=args.resume)
