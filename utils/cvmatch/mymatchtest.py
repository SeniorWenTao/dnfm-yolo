# SIFT 和 bf knn matcher
import time

import numpy as np
import cv2

img1 = cv2.imread("../技能.jpg", 0)
img2 = cv2.imread("../frame_26.jpg", 0)


width = int(img2.shape[1] * 0.5)
height = int(img2.shape[0] * 0.5)
img3 = cv2.resize(img2, (width, height))

# sift = cv2.ORB_create()
sift = cv2.SIFT_create()
# sift = cv2.xfeatures2d.SURF_create()

kp1, des1 = sift.detectAndCompute(img1, None)
s = time.time()
kp2, des2 = sift.detectAndCompute(img3, None)
print("sift time:", (time.time() - s)*1000)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.5*n.distance:
        good.append([m])

img_res = cv2.drawMatchesKnn(img1, kp1, img3, kp2, good,
                            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite("sift-bfknnmatches.jpg", img_res)
# cv2.imshow("sift-bfknnmatches", img_res)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
