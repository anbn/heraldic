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

from scipy.ndimage import convolve
from scipy.ndimage.interpolation import rotate

from scipy.optimize import leastsq
from scipy.stats import binned_statistic



def get_file_list(directory, extensions=['.jpg','.jpeg','.png','.tif'], max_files=0):
    file_list = []
    for f in os.listdir(directory):
        name, file_ext = os.path.splitext(f)
        if file_ext in extensions and name[0]!='.':
            file_list.append(os.path.join(directory, name + file_ext))
    
    file_list = sorted(file_list)
    return file_list if max_files==0 else file_list[:max_files]


def fit_and_predict(fx,fy, predict):
    # we do a linear fit
    func_linear=lambda params,x : params[0]*x+params[1]
    error_func=lambda params,fx,fy: func_linear(params,fx)-fy
    final_params,success=leastsq(error_func,(1.0,2.0),args=(fx,fy))
    return func_linear(final_params,predict)


def bin_and_predict(vals, bins, verbose=False):
    hist_stats_x, _, _, = binned_statistic(vals,vals, statistic='median', bins=bins)

    fy = hist_stats_x
    fx = np.arange(fy.shape[0])
    
    fxx = np.arange(bins)
    fyy = fit_and_predict(fx,fy,fxx)
    if verbose:
        plt.figure("fit"),
        plt.scatter(xrange(vals.shape[0]), np.sort(vals))
        plt.plot(fx,fy,'go',fxx,fyy,'r-')
        plt.show()
    #return np.hstack((fyy[0],hist_stats_x,fyy[-1]))
    return fyy


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=160)
    print "Rolland"

    file_list = get_file_list("images", max_files=0)

    for n,f in enumerate(file_list):
        print "%d/%d %s" % (n+1,len(file_list),f)

        kernel = np.ones((12,12))
        kernel[:,4:8] = -1
        kernel[4:8,:] = -1
        
        imi = io.imread(f, as_grey=True)
        #imi = rotate(imi, 5)
        image = imi[:,int(imi.shape[1]*0.6):].astype("float")
        filtered = horizontal_edge_response = convolve(image, kernel)
        
        fig, (ax0,ax1) = plt.subplots(1,2)
        ax0.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
        ax1.imshow(filtered, interpolation='nearest', cmap=plt.cm.gray)

        markers = np.dstack(np.unravel_index(np.argsort(filtered.ravel()), image.shape))
        markers_y = markers[0,-128:,0]
        markers_x = markers[0,-128:,1]
        plt.scatter(markers_x,markers_y)

        fitx = bin_and_predict(markers_x,6)
        fity = bin_and_predict(markers_y,7)

        plt.scatter(np.tile(fitx,7), np.tile(fity,6), color='yellow')
        #print np.vstack((np.tile(hist_stats_x,7), np.tile(hist_stats_y,6)))


        print fitx.shape, fity.shape
        #fit = np.vstack((np.tile(fitx,7), np.tile(fity,6)))
        
        for ix,x in enumerate(fitx):
            for iy,y in enumerate(fity):
                print "%d %d: %4d %4d     " % (ix,iy,x,y),
                idx = [np.linalg.norm([x-markers_x, y-markers_y], axis=0)<155]
                correct_x, correct_y = np.median(markers_x[idx]), np.median(markers_y[idx])
                plt.scatter(correct_x, correct_y, color='green')

        plt.show()

