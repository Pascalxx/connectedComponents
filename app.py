import numpy as np
import cv2

# Read the image for dilation
img = cv2.imread("gData/test2.jpg", cv2.IMREAD_GRAYSCALE)
# Acquire size of the image
p, q = img.shape
# Show the image
cv2.namedWindow('ii', 0)
cv2.imshow('ii', img)

# Define new image to store the pixels of dilated image
imgDilate = np.zeros((p, q), dtype=np.uint8)
# Define the structuring element
SED = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
constant1 = 1
group_label = 1

# Dilation operation without using inbuilt CV2 function
for i in range(constant1, p - constant1):
    for j in range(constant1, q - constant1):
        if img[i, j] > 0:
            temp = img[i - constant1:i + constant1 + 1, j - constant1:j + constant1 + 1]
            if np.max(imgDilate[i - constant1:i + constant1 + 1, j - constant1:j + constant1 + 1]) == 0:
                # print('a', i, j)
                # print(imgDilate[i - constant1:i + constant1 + 1, j - constant1:j + constant1 + 1])

                product = temp * SED
                product[product > 0] = group_label
                imgDilate[i - constant1:i + constant1 + 1, j - constant1:j + constant1 + 1] = product
                # imgDilate[i, j] = np.max(product)

                # print(temp)
                # print(product)
                # print(imgDilate)

                group_label += 1

            else:
                min_temp = imgDilate[i - constant1:i + constant1 + 1, j - constant1:j + constant1 + 1]
                min_temp[min_temp == 0] = group_label
                min_label = np.min(min_temp)
                product = temp * SED
                product[product > 0] = min_label if min_label <= group_label else group_label
                imgDilate[i - constant1:i + constant1 + 1, j - constant1:j + constant1 + 1] = product
                # if i >= 12 and j >= 31:
                #     print('b', i, j)
                #     print(imgDilate[i - constant1:i + constant1 + 1, j - constant1:j + constant1 + 1])
                #     print(temp)
                #     print(product)
                #     print(imgDilate)

for i in range(constant1, p - constant1):
    for j in range(constant1, q - constant1):
        imgDilate[i][j] = imgDilate[i][j] * 20

print(group_label)
cv2.namedWindow('test', 0)
cv2.imshow('test', imgDilate)
cv2.waitKey(0)
cv2.destroyAllWindows()
