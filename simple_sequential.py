"""
continuously add onto large image knowing we are just matching sequentially
DOES NOT WORK BECAUSE OF BLUR
"""

import cv2 as cv
import os
import time

images_dir = "sample_images"
output_dir = "result_images"
output_name = "simple_sequential.png"
fileType = "jpg"
num_imgs = 10

outputName = os.path.join(output_dir, output_name)


def load_cv_imgs():
    # get all file names from sample images directory
    fileNames = os.listdir(images_dir)
    # files <= only valid files + full path
    for name in fileNames:
        if not name.endswith('.' + fileType) and not name.split(".")[0].isdigit():
            fileNames.remove(name)
    
    # sort so we can go through sequentially
    fileNames.sort(key=lambda x: int(x.split(".")[0]))
    fileNames = [ os.path.join(images_dir, x) for x in fileNames]
    
    # load images
    imgs = []
    for img_name in  fileNames:
        img = cv.imread(img_name)
        if img is None:
            print("could not read image " + img_name)
        else:
            imgs.append(img)
    return fileNames[:num_imgs]


def main():
    fileNames = load_cv_imgs()
    stitcher = cv.Stitcher.create(cv.STITCHER_SCANS)
    # conf threshold should be 0.3 but is too low for it to stitch everything
    stitcher.setPanoConfidenceThresh(0.1)

    imgs = []
    for img_name in  fileNames:
        img = cv.imread(img_name)
        if img is None:
            print("could not read image " + img_name)
        else:
            imgs.append(img)


    # start with base file
    # assume each next image will overlap with the previous
    final_img = imgs[0]
    for i in range(1, len(imgs)):
        status, pano = stitcher.stitch([final_img, imgs[i]])


        if status != cv.Stitcher_OK:
            print("Can't stitch images, error code = %d" % status)
            print(fileNames[i])
            cv.imwrite(outputName, final_img)
            exit()
        else:
            final_img = pano
    

    cv.imwrite(outputName, final_img)
    print("stitching completed successfully. %s saved!" % outputName)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"total time = {end - start}")