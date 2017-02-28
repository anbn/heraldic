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


options = { 'verbose' : True }


def get_file_list(directory, extensions=['.jpg','.jpeg','.png','.tif'], max_files=0):
    file_list = []
    for f in os.listdir(directory):
        name, file_ext = os.path.splitext(f)
        if file_ext in extensions and name[0]!='.':
            file_list.append(os.path.join(directory, name + file_ext))
    
    file_list = sorted(file_list)
    return file_list if max_files==0 else file_list[:max_files]


def fit_and_predict(fx,fy, predict):
    func_linear = lambda params,x: params[0]*x+params[1]
    error_func  = lambda params,fx,fy: func_linear(params,fx)-fy
    final_params,success = leastsq(error_func,(1.0,2.0),args=(np.asarray(fx),np.asarray(fy)))
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


def build_kernel(size, band):
    kernel = np.ones((size,size))
    kernel[:,size/2-band/2:size/2+band/2] = -1
    kernel[size/2-band/2:size/2+band/2,:] = -1
    return kernel


def compute_grid(filtered, (cross_h, cross_w), num=128, dist_thres=25):
    """ fits a grid over the given responses
        num: take num highest responses
        dist_thres: discard points more than dist_thres pixels away from grid crossings
    """
    markers = np.dstack(np.unravel_index(np.argsort(filtered.ravel()), filtered.shape))
    markers_y = markers[0,-num:,0]
    markers_x = markers[0,-num:,1]

    fitx = bin_and_predict(markers_x,cross_w)
    fity = bin_and_predict(markers_y,cross_h)

    if options["verbose"]:
        plt.scatter(markers_x,markers_y)
        plt.scatter(np.tile(fitx,cross_h), np.tile(fity,cross_w), color='yellow')

    grid = np.zeros((cross_h+2,cross_w+2), dtype='int, int')
    valid = np.zeros((cross_h+2,cross_w+2), dtype=bool)
    for ix,x in enumerate(fitx):
        for iy,y in enumerate(fity):
            idx = [np.linalg.norm([x-markers_x, y-markers_y], ord=2, axis=0)<dist_thres]
            if np.sum(idx) > 0:
                grid[iy+1,ix+1] = np.median(markers_x[idx]), np.median(markers_y[idx])
                valid[iy+1,ix+1] = True
            else:
                valid[iy+1,ix+1] = False

    left  = grid[cross_h/2+1,cross_w/2-2][0],grid[cross_h/2+1,cross_w/2-2][1]
    right = grid[cross_h/2+1,cross_w/2+3][0],grid[cross_h/2+1,cross_w/2+3][1]
    mid = (np.asarray(left) + np.asarray(right)) / 2
    rotation = np.arctan2(*(right-mid)[::-1])/np.pi*180
    #plt.scatter(*left, color="black")
    #plt.scatter(*mid, color="black")
    #plt.scatter(*right, color="black")


    #for i in range(1,cross_h+1):
    #    grid[i,0][0] = grid[i,1][0]-(grid[i,2][0]-grid[i,1][0])
    #    grid[i,0][1] = grid[i,1][1]-(grid[i,2][1]-grid[i,1][1])

    #    grid[i,cross_w+1][0] = grid[i,cross_w][0]+(grid[i,cross_w][0]-grid[i,cross_w-1][0])
    #    grid[i,cross_w+1][1] = grid[i,cross_w][1]+(grid[i,cross_w][1]-grid[i,cross_w-1][1])

    #for i in range(0,cross_w+2):
    #    grid[0,i][0] = grid[1,i][0]-(grid[2,i][0]-grid[1,i][0])
    #    grid[0,i][1] = grid[1,i][1]-(grid[2,i][1]-grid[1,i][1])

    #    grid[cross_h+1,i][0] = grid[cross_h,i][0]+(grid[cross_h,i][0]-grid[cross_h-1,i][0])
    #    grid[cross_h+1,i][1] = grid[cross_h,i][1]+(grid[cross_h,i][1]-grid[cross_h-1,i][1])


    print
    for i in range(1,cross_h+1):
        fx = [x             for x,v in zip(np.arange(cross_w+2), valid[i]) if v]
        fy = [grid[i,j][0]  for j,v in zip(np.arange(cross_w+2), valid[i]) if v]
        fit1 = fit_and_predict(fx, fy, np.arange(0,cross_w+2))
        fy = [grid[i,j][1]  for j,v in zip(np.arange(cross_w+2), valid[i]) if v]
        fit2 = fit_and_predict(fx, fy, np.arange(0,cross_w+2))

        for j in range(cross_w+2):
            if not valid[i,j]:
                grid[i,j] = (int(fit1[j]),  int(fit2[j]))

    for i in range(0,cross_w+2):
        fx = [x             for x in np.arange(1,cross_h+1)]
        fy = [grid[j,i][0]  for j in np.arange(1,cross_h+1)]
        fit1 = fit_and_predict(fx, fy, np.arange(0,cross_h+2))
        fy = [grid[j,i][1]  for j in np.arange(1,cross_h+1)]
        fit2 = fit_and_predict(fx, fy, np.arange(0,cross_h+2))

        grid[0,i]         = (int(fit1[0]),  int(fit2[0]))
        grid[cross_h+1,i] = (int(fit1[-1]), int(fit2[-1]))

    return grid, rotation


def process_plate(image, kernel, (cross_h,cross_w)):
    filtered = convolve(image, kernel)
    
    if options["verbose"]:
        fig, (ax0,ax1) = plt.subplots(1,2)
        ax0.imshow(image, interpolation='none', cmap=plt.cm.gray)
        ax1.imshow(filtered, interpolation='none', cmap=plt.cm.gray)

    grid, rotation = compute_grid(filtered, (cross_h,cross_w))
    print "(rotation: %f)" % rotation

    if options["verbose"]:
        plt.figure("extract")
        plt.imshow(image, cmap="gray")

        plt.scatter([p[0] for p in grid.flat], [p[1] for p in grid.flat], color='red')
        plt.show()

    for i in range(cross_h+1):
        for j in range(cross_w+1):
            xb, xe = np.min((grid[i,j][0], grid[i+1,j][0])), np.max((grid[i,j+1][0], grid[i+1,j+1][0])) 
            yb, ye = np.min((grid[i,j][1], grid[i,j+1][1])), np.max((grid[i+1,j][1], grid[i+1,j+1][1])) 
            
            plt.figure("%d,%d"%(i,j))
            plt.imshow(image[yb:ye,xb:xe], cmap='gray')
    plt.show()


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=160)

    cross_h, cross_w = 7,6
    #cross_h, cross_w = 5,6
    kernel = build_kernel(32,4)

    file_list = get_file_list("images", max_files=0)

    for n,f in enumerate(file_list[-1:]):
        print "%d/%d %s" % (n+1,len(file_list),f),

        imi = io.imread(f)
        plate_left  = imi[:, int(0.6*imi.shape[1]):].astype('float')
        plate_right = imi[:,:int(0.4*imi.shape[1]) ].astype('float')

        process_plate(plate_left,  kernel, (cross_h, cross_w))
        process_plate(plate_right, kernel, (cross_h, cross_w))

