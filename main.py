import cv2 as cv
import os

if __name__ == "__main__":
    images_directory = "sample_images"
    
    # get image files and sort based on name (ascending)
    files = []
    for f in os.listdir(images_directory):
        if f.endswith(".jpg"):
            files.append(os.path.join(images_directory, f))
    files.sort()
    # print(files)


    # stitcher to stitch images together
    stitcher = cv.Stitcher.create(cv.STITCHER_SCANS)
    stitcher.setPanoConfidenceThresh(0.3) # stitching less images requires us to lower the confidence
    
    
    completed_image = cv.imread(files[0])

    for file in files[1:]:
        current_image = (cv.imread(file))
        print(file)

        status, new_image = stitcher.stitch([completed_image, current_image])
        if status == cv.Stitcher_OK:
            print("Stitching completed successfully.")
            completed_image = new_image # so one bad image does ruin everything
        
        else: # if not successful show an error message
            if status == cv.Stitcher_ERR_NEED_MORE_IMGS:
                print("Need more images to stitch.")
            elif status == cv.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
                print("Homography estimation failed.")
            elif status == cv.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
                print("Camera parameters adjustment failed.")

    cv.imwrite(os.path.join("result_images", f"output.png"), completed_image)
