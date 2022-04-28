#!/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import argparse

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def main():
    print('loading images...')
    parser = argparse.ArgumentParser()
    parser.add_argument('left', default='rectimgA.png', help='left rectified image')
    parser.add_argument('right', default='rectimgB.png', help='right rectified image')
    args = parser.parse_args()

    imgL = cv.pyrDown(cv.imread(args.left))  # downscale images for faster processing
    imgR = cv.pyrDown(cv.imread(args.right))

    # disparity range is tuned for 'aloe' image pair
    window_size = 15 #3
    min_disp = 3 #16
    num_disp = 112-min_disp
    stereo = cv.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    # Select ROI
    r = cv.selectROI(imgL)
    # Crop image
    cropped_disp = disp[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    print('generating 3d point cloud...',)
    h, w = imgL.shape[:2]
    f = 0.8*w                          # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    points = cv.reprojectImageTo3D(cropped_disp, Q)
    colors = cv.cvtColor(imgL[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])], cv.COLOR_BGR2RGB)
    mask = cropped_disp > cropped_disp.min()
    out_points = points[mask]
    out_colors = colors[mask]

    out_fn = 'out.ply'
    write_ply(out_fn, out_points, out_colors)
    print('%s saved' % out_fn)

    cv.imshow('left', imgL)
    cv.imshow('disparity', (disp-min_disp)/num_disp)
    cv.waitKey()

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
