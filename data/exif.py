#!/usr/bin/env python

import numpy
from matplotlib import pyplot
import subprocess

directory = 'kodak_db'


def get_exif(filename):
    from fractions import Fraction
    out, err = subprocess.Popen(['identify', '-format', '%[exif:FNumber],%[exif:ExposureTime],%[exif:ISOSpeedRatings]', directory + "/" + filename], stdout=subprocess.PIPE).communicate()
    return [1.0 * Fraction(x) for x in out.strip().split(',')]


def read_image_exif_data(filelist):
    imagelist = []
    for image in imagenamelist:
        item = get_exif(directory + "/" + image) + "," + image
        imagelist.append(item.split(','))
    return imagelist

def read_exif_from_cache(numfiles):
    exif_data = numpy.fromfile("exif_cache.dat", dtype=numpy.float)
    return exif_data.reshape(numfiles, 3)

def create_exif_data_cache():
    exif_data = []
    for i in range(numfiles):
        data = get_exif(filelist[i])
        exif_data.append(data)
    exif_data = numpy.array(exif_data, dtype=numpy.float)
    exif_data.tofile("exif_cache.dat")
    return exif_data

def get_filelist_from_files(directory):
    out, err = subprocess.Popen(['ls', directory], stdout=subprocess.PIPE).communicate()
    filelist = out.split()
    numpy.savetxt("filelist.txt", filelist, fmt="%s")
    return filelist

def get_filelist_from_cache():
    filelist = numpy.loadtxt("filelist.txt", dtype=numpy.string_)
    return filelist


def distmat_fig(matrix_pce, matrix_ncc, matrix_ans):
    f, (ax1, ax2, ax3) = pyplot.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
    ax1.set_adjustable('box-forced')
    ax2.set_adjustable('box-forced')
    ax3.set_adjustable('box-forced')

    ax1.imshow(matrix_pce, cmap=pyplot.cm.jet, vmax=100)
    ax1.set_title("PCE scores")
    ax2.imshow(matrix_ncc, cmap=pyplot.cm.jet, vmin=-0.002, vmax=0.005)
    ax2.set_title("NCC scores")
    ax3.imshow(matrix_ans, cmap=pyplot.cm.jet)
    ax3.set_title("ground truth")
    f.tight_layout()
    f.savefig("compare_ncc_pce.png", dpi=300)


def pce_analysis_fig(matrix_pce, matrix_fnum, matrix_exp, matrix_iso):
    f, (ax1, ax2, ax3, ax4) = pyplot.subplots(nrows=1, ncols=4, sharex=True, sharey=True)
    ax1.set_adjustable('box-forced')
    ax2.set_adjustable('box-forced')
    ax3.set_adjustable('box-forced')
    ax4.set_adjustable('box-forced')

    ax1.imshow(matrix_pce, cmap=pyplot.cm.jet, vmax=100)
    ax1.set_title("PCE scores")
    ax2.imshow(matrix_fnum, cmap=pyplot.cm.jet)
    ax2.set_title("f number")
    ax3.imshow(matrix_exp, cmap=pyplot.cm.jet)
    ax3.set_title("exposure times")
    ax4.imshow(matrix_iso, cmap=pyplot.cm.jet)
    ax4.set_title("ISO values")
    f.set_size_inches(16, 4, forward=True)
    f.tight_layout()
    f.savefig("compare_pce.png", dpi=300)







if __name__ == "__main__":

    #filelist = get_filelist_from_files(directory)
    filelist = get_filelist_from_cache()
    numfiles = len(filelist)

    #exif_data = create_exif_data_cache()
    exif_data = read_exif_from_cache(numfiles)

    #matrix_file = 'cluster-analysis/data/set_2/matrix_304_pce.txt'
    #matrix_pce = numpy.loadtxt(matrix_file, delimiter=',', usecols=range(304))
    matrix_file = 'set_2/matrix_304_pce.dat'
    matrix_pce = numpy.fromfile(matrix_file)
    matrix_pce = matrix_pce.reshape(numfiles, numfiles)

    #matrix_file = 'cluster-analysis/data/set_3/matrix_304_ncc.txt'
    #matrix_ncc = numpy.loadtxt(matrix_file, delimiter=',', usecols=range(304))
    matrix_file = 'set_3/matrix_304_ncc.dat'
    matrix_ncc = numpy.fromfile(matrix_file)
    matrix_ncc = matrix_ncc.reshape(numfiles, numfiles)



    #generate additional matrices based on exif data and ground truth
    matrix_fnum = numpy.zeros( matrix_pce.shape, dtype=numpy.float)
    matrix_exp = numpy.zeros( matrix_pce.shape, dtype=numpy.float)
    matrix_iso = numpy.zeros( matrix_pce.shape, dtype=numpy.float)
    matrix_ans = numpy.zeros( matrix_pce.shape, dtype=numpy.float)

    for i in range(matrix_pce.shape[0]):
        for j in range(matrix_pce.shape[1]):
            matrix_fnum[i][j] = numpy.sqrt(exif_data[i][0] * exif_data[j][0])
            matrix_exp[i][j] = numpy.sqrt(exif_data[i][1] * exif_data[j][1])
            matrix_iso[i][j] = numpy.sqrt(exif_data[i][2] * exif_data[j][2])
            cam1 = filelist[i].split("_")[-2]
            cam2 = filelist[j].split("_")[-2]
            if cam1 == cam2:
                matrix_ans[i][j] = 100.0
            else:
                matrix_ans[i][j] = 1.0
            if i == j:
                matrix_ans[i][j] = 0.0
        matrix_fnum[i][i] = 0.0
        matrix_exp[i][i] = 0.0
        matrix_iso[i][i] = 0.0


    #plot everything
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = pyplot.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    ax1.set_adjustable('box-forced')
    ax2.set_adjustable('box-forced')
    ax3.set_adjustable('box-forced')
    ax4.set_adjustable('box-forced')
    ax5.set_adjustable('box-forced')
    ax6.set_adjustable('box-forced')

    ax1.imshow(matrix_pce, cmap=pyplot.cm.jet, vmax=100)
    ax1.set_title("PCE scores")
    ax2.imshow(matrix_ncc, cmap=pyplot.cm.jet, vmin=-0.002, vmax=0.005)
    ax2.set_title("NCC scores")
    ax3.imshow(matrix_ans, cmap=pyplot.cm.jet)
    ax3.set_title("ground truth")
    ax4.imshow(matrix_fnum, cmap=pyplot.cm.jet)
    ax4.set_title("f number")
    ax5.imshow(matrix_exp, cmap=pyplot.cm.jet)
    ax5.set_title("exposure times")
    ax6.imshow(matrix_iso, cmap=pyplot.cm.jet)
    ax6.set_title("iso values")

    f.tight_layout()
    f.savefig("pce_ncc_correlation.png", dpi=300)

    pyplot.show()
    raw_input()

