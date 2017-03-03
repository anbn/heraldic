#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os, sys, time, getopt
import numpy as np
import matplotlib.pyplot as plt

from skimage import io

from scipy.ndimage import convolve
from scipy.ndimage.interpolation import rotate
from scipy.optimize import leastsq
from scipy.stats import binned_statistic

options = { 'verbose' : False }

def fit_and_predict(fx,fy, predict):
    func_linear = lambda params,x: params[0]*x+params[1]
    error_func  = lambda params,fx,fy: func_linear(params,fx)-fy
    final_params,success = leastsq(error_func,(1.0,2.0),args=(np.asarray(fx),np.asarray(fy)))
    return func_linear(final_params,predict)


def bin_and_predict(vals, bins):
    vals = np.sort(vals)
    distances = np.abs((vals.reshape(1,-1) - vals.reshape(-1,1)))
    neighbours = np.zeros_like(vals)
    hist_stats_x = np.zeros(bins)
    for i in range(vals.shape[0]):
        neighbours[i] = np.sum(distances[i]<50)
    for i in range(bins):
        most_popular = np.argmax(neighbours)
        neighbours[distances[most_popular,:]<50]=0
        hist_stats_x[i] = vals[most_popular]
    hist_stats_x = np.sort(hist_stats_x)

    fy = hist_stats_x
    fx = np.arange(fy.shape[0])
    fxx = np.arange(bins)
    fyy = fit_and_predict(fx,fy,fxx)
    if False and options["verbose"]:
        plt.figure("fit"),
        plt.scatter(xrange(vals.shape[0]), np.sort(vals))
        plt.plot(fx,fy,'go',fxx,fyy,'r-')
        plt.show()
    return fyy


def build_kernel(size, band):
    kernel = np.ones((size,size))
    kernel[:,size/2-band/2:size/2+band/2] = -1
    kernel[size/2-band/2:size/2+band/2,:] = -1
    return kernel


def compute_grid(filtered, (cross_h, cross_w), num=128, dist_thres=50):
    """
        fits a grid over the given responses
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
        plt.scatter(np.repeat(fitx,cross_h), np.tile(fity,cross_w), color='yellow')
        plt.show()

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

    return grid


def process_plate(image, kernel, (cross_h,cross_w), (out_folder, out_prefix)):
    filtered = convolve(image, kernel)
    
    if options["verbose"]:
        fig, (ax0,ax1) = plt.subplots(1,2)
        ax0.imshow(image, interpolation='none', cmap=plt.cm.gray)
        ax1.imshow(filtered, interpolation='none', cmap=plt.cm.gray)

    grid = compute_grid(filtered, (cross_h,cross_w))

    if options["verbose"]:
        plt.figure("extract")
        plt.imshow(image, cmap="gray")

        plt.scatter([p[0] for p in grid.flat], [p[1] for p in grid.flat], color='red')
        plt.show()

    for i in range(cross_h+1):
        for j in range(cross_w+1):
            xb, xe = np.min((grid[i,j][0], grid[i+1,j][0])), np.max((grid[i,j+1][0], grid[i+1,j+1][0])) 
            yb, ye = np.min((grid[i,j][1], grid[i,j+1][1])), np.max((grid[i+1,j][1], grid[i+1,j+1][1])) 

            imo = image[yb:ye,xb:xe]
            
            plt.imsave(os.path.join(out_folder, "%s-%d%d" % (out_prefix,i,j)), imo, cmap="gray")

            if options["verbose"]:
                plt.figure("%d,%d"%(i,j))
                plt.imshow(imo, cmap='gray')

    if options["verbose"]:
        plt.show()

def parse_size(str):
    if str=="big":     return (7,6)
    elif str=="small": return (5,6)
    elif ":" in str:
        a,b = str.split(':')
        return (int(a), int(b))
    else:
        print "unable to parse grid size"
        sys.exit(1)


def usage():
    print "python %s image [options]" % sys.argv[0]
    sys.exit(0)


if __name__ == "__main__":
    np.set_printoptions(precision=4,  suppress=True,  linewidth=160)

    kernel = build_kernel(32, 4)

    left, right, single = None, None, None
    in_file, out_folder = None, None

    try:
        opts, args = getopt.getopt(sys.argv[1:],"hl:r:s:i:o:v",["help","left=","right=","single=","in=","out=","verbose"])
    except getopt.GetoptError as err:
        print str(err)  # will print something like "option -a not recognized"
        sys.exit(2)
    for o,  a in opts:
        if o in ("-v", "--verbose"):
            options["verbose"] = True
        elif o in ("-l","--left"):
            left = a
        elif o in ("-r","--right"):
            right = a
        elif o in ("-s","--single"):
            single = a
        elif o in ("-i","--in"):
            in_file = a
        elif o in ("-o","--out"):
            out_folder = a
        elif o in ("-h","--help"):
            usage()
            sys.exit(0)
        else:
            assert False,  "unhandled option"

    if (right is None) != (left is None):
        print "you have to specify both pages, right and left"
        sys.exit(1)
    if right is None and single is None:
        print "choose either single or double page processing"
        sys.exit(1)
    if in_file is None or out_folder is None:
        print "please specify input file and output folder"
        sys.exit(1)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    out_prefix = "".join(os.path.basename(in_file).split('.')[:-1])

    imi = io.imread(in_file)
    print "Processing %s..." % in_file,

    if single is None:
        plate_left  = imi[:, :int(0.4*imi.shape[1]) ].astype('float')
        plate_right = imi[:,  int(0.6*imi.shape[1]):].astype('float')

        if not left=="ignore":
            try:
                process_plate(plate_left,   kernel, parse_size(left), (out_folder,out_prefix+"-left"))
                print "left-done",
            except: print "left-failed",
        else: print "left-ignored",
        if not right=="ignore":
            try:
                process_plate(plate_right,  kernel, parse_size(right), (out_folder,out_prefix+"-right"))
                print "right-done",
            except: print "right-failed",
        else: print "right-ignored",
    else:
        plate = imi.astype('float')
        try:
            process_plate(plate,  kernel,  parse_size(single), (out_folder,out_prefix+"-single"))
            print "single-done",
        except:
            print "single-failed",
    print "done."
