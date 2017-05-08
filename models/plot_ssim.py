"""
modified code from SSI
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import skimage

#img = img_as_float(data.camera())
img_original = img_as_float(skimage.io.imread('/Users/ouchouyang/Desktop/CompVisionProject/image.bmp'))
img_ours = img_as_float(skimage.io.imread('/Users/ouchouyang/Desktop/CompVisionProject/inter_image.bmp'))
#img_SRGAM ... from other state-of-art algorithms
#img_xxx
#rows, cols, channels = img.shape

#noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
#noise[np.random.random(size=noise.shape) > 0.5] *= -1

def mse(x, y):
    return np.linalg.norm(x - y)

#img_noise = img + noise
#img_const = img + abs(noise)


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4),
                         sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})
ax = axes.ravel()

mse_none = mse(img_original, img_original)
ssim_none = ssim(img_original, img_original, data_range=img_original.max() - img_original.min(),multichannel=True)
psnr_none = psnr(img_original, img_original, data_range=img_original.max() - img_original.min())

mse_2 = mse(img_original, img_ours)
ssim_2 = ssim(img_original, img_ours,
                  data_range=img_ours.max() - img_ours.min(),multichannel=True)
psnr_2 = psnr(img_original, img_ours, data_range=img_ours.max() - img_ours.min())

"""
mse_2 = mse(img_original, img_xxx)
ssim_2 = ssim(img_original, img_xxx,
                  data_range=img_xxx.max() - img_xxx.min(),multichannel=True)

"""
label = 'MSE: {:.2f}, SSIM: {:.2f}, PSNR: {:.2f}'

ax[0].imshow(img_original, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[0].set_xlabel(label.format(mse_none, ssim_none, psnr_none))
ax[0].set_title('original')

ax[1].imshow(img_ours, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[1].set_xlabel(label.format(mse_2, ssim_2, psnr_2))
ax[1].set_title('ours')

"""
ax[2].imshow(img_xxx, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[2].set_xlabel(label.format(mse_3, ssim_3))
ax[2].set_title('xxx')
"""


plt.tight_layout()
plt.savefig('comparsion.png',bbox_inches='tight')

plt.show()
