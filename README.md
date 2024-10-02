# UNDER DEVELOPMENT

# image-stitching

## Feature Matching
A few approaches are:
- SIFT
  - Accurate and reliable from our (very small) tests
  - took a lot of resources and time to find features and match them.
- SURF
  - Faster, maybe better, than sift
  - patented :(
- ORB
  - produces features much more densely closer to the center of the image
  - faster than SIFT and also made by OpenCV so patent free.

We are trying out ORB as it performed relatively well at finding features and matching them.

## useful Resources
[OpenCV Feature Detection and Description](https://docs.opencv.org/4.10.0/db/d27/tutorial_py_table_of_contents_feature2d.html)