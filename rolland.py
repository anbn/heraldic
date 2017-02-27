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
from scipy.ndimage import convolve

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


def draw_grid(x,y,dx,dy):
    for i in range(6):
        for j in range(7):
            plt.scatter(x+i*dx, y+j*dy, color="red")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=160)
    print "Rolland"

    file_list = get_file_list("images", max_files=0)

    for n,f in enumerate(file_list):
        print "%d/%d %s" % (n+1,len(file_list),f)

        #kernel = np.ones((12,12))
        #kernel[:,4:8] = -1
        #kernel[4:8,:] = -1
        kernel = np.ones((12,12))
        kernel[:,4:8] = -1
        kernel[4:8,:] = -1
        
        imi = io.imread(f, as_grey=True)
        image = imi[:,imi.shape[1]*0.6:].astype("float")
        filtered = horizontal_edge_response = convolve(image, kernel)
        
        fig, (ax0,ax1,ax2) = plt.subplots(1,3)
        ax0.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
        ax1.imshow(kernel, interpolation='nearest', cmap=plt.cm.gray)
        ax2.imshow(filtered, interpolation='nearest', cmap=plt.cm.gray)

        markers = np.dstack(np.unravel_index(np.argsort(filtered.ravel()), image.shape))
        y = markers[0,-128:,0]
        x = markers[0,-128:,1]
        plt.scatter(x,y)

        hist_stats, _, _, = binned_statistic(x,x, statistic='median', bins=6)
        for i in hist_stats:
            plt.axvline(x=i)

        hist_stats, _, _, = binned_statistic(y,y, statistic='median', bins=7)
        for i in hist_stats:
            plt.axhline(y=i)

        plt.show()

        sys.exit(1)


        fy = hist_bounds[hist_num.argsort()[-6:]]
        fx = np.arange(fy.shape[0])

        print "FY", fy

        
        #draw_grid(200,300, 300,400)
        plt.figure("fit"), plt.scatter(xrange(x.shape[0]), np.sort(x))

        plt.figure("fit")
        ax2.scatter(fx,fy)
        print markers[0,:,0], markers[0,:,1]

        # here, create lambda functions for Line, Quadratic fit
        # tpl is a tuple that contains the parameters of the fit
        funcLine=lambda tpl,x : tpl[0]*x+tpl[1]
        # func is going to be a placeholder for funcLine,funcQuad or whatever 
        # function we would like to fit
        func=funcLine
        # ErrorFunc is the diference between the func and the fy "experimental" data
        ErrorFunc=lambda tpl,fx,fy: func(tpl,fx)-fy
        #tplInitial contains the "first guess" of the parameters 
        tplInitial1=(1.0,2.0)
        # leastsq finds the set of parameters in the tuple tpl that minimizes
        # ErrorFunc=yfit-yExperimental
        tplFinal1,success=leastsq(ErrorFunc,tplInitial1[:],args=(fx,fy))
        print " linear fit ",tplFinal1
        xx1=np.linspace(fx.min(),fx.max(),50)
        yy1=func(tplFinal1,xx1)
        plt.plot(xx1,yy1,'r-',fx,fy,'bo')



        #for i in range(-1,6):
        #    print tplFinal1[0] + i*tplFinal1[1]

        plt.show()
