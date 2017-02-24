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

from skimage.feature import corner_harris, corner_subpix, corner_peaks




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
        image = imi[:,imi.shape[1]*0.6:]
        
        coords = corner_peaks(corner_harris(image), min_distance=5)
        coords_subpix = corner_subpix(image, coords, window_size=13)

        fig, ax = plt.subplots()
        ax.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
        ax.plot(coords[:, 1], coords[:, 0], '.b', markersize=3)
        ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
        plt.show()
