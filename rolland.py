import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure

from skimage.morphology import disk
from skimage.filters import rank
from skimage.color import rgb2gray
from skimage.util import img_as_uint
from skimage.exposure import equalize_hist, histogram, cumulative_distribution
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.feature import canny





def get_file_list(directory, extensions=['.jpg','.jpeg','.png','.tif'], max_files=0):
    file_list = []
    for f in os.listdir(directory):
        name, file_ext = os.path.splitext(f)
        if file_ext in extensions and name[0]!='.':
            file_list.append(os.path.join(directory, name + file_ext))
    
    file_list = sorted(file_list)
    return file_list if max_files==0 else file_list[:max_files]


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=160)
    print "Rolland"

    file_list = get_file_list("images", max_files=0)

    for n,f in enumerate(file_list):
        print "%d/%d %s" % (n+1,len(file_list),f)

        imi = io.imread(f, as_grey=True)
        imi = imi[:,imi.shape[1]*0.6:]
        imi = canny(imi)
        h, theta, d = hough_line(imi)

        fig, axes = plt.subplots(1, 3, figsize=(15, 6), subplot_kw={'adjustable': 'box-forced'})
        ax = axes.ravel()

        ax[0].imshow(imi, cmap="gray")
        ax[0].set_title('Input image')
        ax[0].set_axis_off()

        ax[1].imshow(np.log(1 + h), extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                                          cmap="gray", aspect=1/1.5)
        ax[1].set_title('Hough transform')
        ax[1].set_xlabel('Angles (degrees)')
        ax[1].set_ylabel('Distance (pixels)')
        ax[1].axis('image')

        ax[2].imshow(imi, cmap="gray")
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold=0.0*np.max(h), min_distance=300)):
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - imi.shape[1] * np.cos(angle)) / np.sin(angle)
            ax[2].plot((0, imi.shape[1]), (y0, y1), '-r')
            ax[2].set_xlim((0, imi.shape[1]))
            ax[2].set_ylim((imi.shape[0], 0))
            ax[2].set_axis_off()
            ax[2].set_title('Detected lines')

        plt.show()

        if True:
            plt.figure("image in"), plt.imshow(imi, cmap="gray")
            plt.show()
