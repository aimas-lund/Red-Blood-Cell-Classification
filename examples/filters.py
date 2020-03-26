import cv2
import numpy as np
import os
from DataHandler import DataHandler


path = "C:/Users/Aimas/Desktop/DTU/01-BSc/6_semester/01_Bachelor_Project/data/freja"
folders = os.listdir(path)
files = os.listdir(os.path.join(path, folders[0]))
filename = files[255]
dh = DataHandler(data_dir=path,
                 categories=folders)
img = dh.fetch_image(filename, dh.CATEGORIES[0], grayscale=True)

laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="int")
sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")
sobelY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int")

sobelX_output = cv2.filter2D(img, -1, sobelX)
sobelY_output = cv2.filter2D(img, -1, sobelY)
laplacian_output = cv2.filter2D(img, -1, laplacian)
cv2.imshow("original", img)
cv2.imshow("sobel X filter applied", sobelX_output)
cv2.imshow("sobel Y filter applied", sobelY_output)
cv2.imshow("laplacian filter applied", laplacian_output)
cv2.waitKey(0)
cv2.destroyAllWindows()