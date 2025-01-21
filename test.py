import cv2 as cv
import os

image_path = os.path.join(
    ".", "photo", "png-clipart-pokemon-logo-pokemon-logo-thumbnail.png"
)

img = cv.imread(image_path)

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow("image", img)
cv.imshow("image", img_gray)
cv.waitKey(0)
