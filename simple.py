import cv2 as cv
import os
import time

images_dir = "sample_images"
output_dir = "result_images"
output_name = "simple.png"

fileType = "jpg"
num_imgs = 10

def get_image_names():
    # get all file names from sample images directory
    fileNames = os.listdir(images_dir)
    
    # files <= only valid files + full path
    for name in fileNames:
        if not name.endswith('.' + fileType) and not name.split(".")[0].isdigit():
            fileNames.remove(name)
    fileNames.sort(key = lambda x: int(x.split(".")[0])) # sort based on numerical order

    fileNames = [ os.path.join(images_dir, x) for x in fileNames]

    return fileNames[:num_imgs]

def main():
    files = get_image_names()

    # load images
    imgs = []
    for img_name in files:
        img = cv.imread(cv.samples.findFile(img_name))
        if img is None:
            print("can't read image " + img_name)
        imgs.append(img)


    stitcher = cv.Stitcher.create(cv.STITCHER_SCANS)
    stitcher.setPanoConfidenceThresh(0.3)
    status, pano = stitcher.stitch(imgs)

    if status != cv.Stitcher_OK:
        print("Can't stitch images, error code = %d" % status)

    
    
    output_file = os.path.join(output_dir, output_name)
    print(output_file)

    cv.imwrite(output_file, pano)
    print("stitching completed successfully. %s saved!" % output_file)

    print('Done')

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()

    print(f"program took {end - start} seconds")
