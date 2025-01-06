import cv2 as cv
import os
import time

images_dir = "sample_images"
output_dir = "result_images"
output_name = "result.png"

def load_cv_imgs():
    # get all file names from sample images directory
    fileNames = os.listdir(images_dir)

    # files <= only valid files + full path
    for name in fileNames:
        if not name.endswith(".jpg") and not name.split(".")[0].isdigit():
            fileNames.remove(name)
    fileNames = [ os.path.join(images_dir, x) for x in fileNames]
    
    # load images
    imgs = []
    for img_name in  fileNames:
        img = cv.imread(img_name)
        if img is None:
            print("could not read image " + img_name)
        else:
            imgs.append(img)
    return imgs

def simple():
    imgs = load_cv_imgs()

    # stitch images
    stitcher = cv.Stitcher.create(cv.STITCHER_SCANS)
    status, pano = stitcher.stitch(imgs)

    if status != cv.Stitcher_OK:
        print("Can't stitch images, error code = %d" % status)
    else:
        outputName = os.path.join(output_dir, output_name)
        cv.imwrite(os.path.join(output_dir, outputName), pano)
        print("stitching completed successfully. %s saved!" % outputName)

    print('Done')

if __name__ == "__main__":
    start = time.perf_counter()

    simple()

    end = time.perf_counter()
    print(f"total time = {end - start}")