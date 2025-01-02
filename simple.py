import cv2 as cv
import numpy as np
import os

images_dir = "sample_images"
output_dir = "result_images"
output_name = "result.png"

def load_cv_imgs():
    # get all file names from sample images directory
    fileNames = os.listdir(images_dir)
    files = []
    for name in fileNames:
        if name.endswith(".jpg"):
            files.append(os.path.join(images_dir, name))
    files.sort()
    
    # load images
    imgs = []
    for img_name in  files:
        img = cv.imread(img_name)
        if img is None:
            print("could not read image " + img_name)
        else:
            imgs.append(img)
    return imgs


def simple_stitch():
    imgs = load_cv_imgs()
    
    # stitch images
    stitcher = cv.Stitcher.create(cv.STITCHER_SCANS)
    status, pano = stitcher.stitch(imgs)

    if status != cv.Stitcher_OK:
        print("Can't stitch images, error code = %d" % status)
    else:
        outputName = os.path.join(output_dir, output_name)
        cv.imwrite(outputName, pano)
        print("stitching completed successfully. %s saved!" % outputName)
 
    print('Done')