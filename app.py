import numpy as np
import cv2


def label_fix(i, j):
    if imgDilate[i, j] > 0:
        temp = imgDilate[i - constant1:i + constant1 + 1, j - constant1:j + constant1 + 1]
        min_label = min(filter(lambda x: x > 0, temp.reshape(-1, )))
        temp[temp > min_label] = min_label
        imgDilate[i - constant1:i + constant1 + 1, j - constant1:j + constant1 + 1] = temp


# # # # # # # # # # #
if __name__ == '__main__':
    # Read the image for dilation
    img = cv2.imread("gData/nchu.jpg", cv2.IMREAD_GRAYSCALE)
    # Acquire size of the image
    p, q = img.shape
    # Show the image
    # cv2.namedWindow('ii', 0)
    cv2.imshow('ii', img)

    # Define new image to store the pixels of dilated image
    imgDilate = np.zeros((p, q), dtype=np.uint8)
    # Define the structuring element
    SED = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    constant1 = 1
    group_label = 1

    # Start
    for i in range(constant1, p - constant1):
        for j in range(constant1, q - constant1):
            if img[i, j] > 0:
                temp = img[i - constant1:i + constant1 + 1, j - constant1:j + constant1 + 1]
                if np.max(imgDilate[i - constant1:i + constant1 + 1, j - constant1:j + constant1 + 1]) == 0:
                    product = temp * SED
                    product[product > 0] = group_label
                    imgDilate[i - constant1:i + constant1 + 1, j - constant1:j + constant1 + 1] = product
                    group_label += 1

                else:
                    min_temp = imgDilate[i - constant1:i + constant1 + 1, j - constant1:j + constant1 + 1]
                    min_label = min(filter(lambda x: x > 0, min_temp.reshape(-1, )))
                    product = temp * SED
                    product[product > 0] = min_label
                    imgDilate[i - constant1:i + constant1 + 1, j - constant1:j + constant1 + 1] = product

    # label fix
    for i in range(p - constant1, constant1, -1):
        for j in range(q - constant1, constant1, -1):
            label_fix(i, j)

    # label fix sec
    for i in range(constant1, p - constant1):
        for j in range(constant1, q - constant1):
            label_fix(i, j)

    total_cal = {}
    for i in range(constant1, p - constant1):
        for j in range(constant1, q - constant1):
            if imgDilate[i][j] > 0:
                attr_str = str(imgDilate[i][j])
                if attr_str in total_cal:
                    total_cal[attr_str] += 1
                else:
                    total_cal[attr_str] = 1
            imgDilate[i][j] = imgDilate[i][j] * 20

    print(total_cal)
    # cv2.namedWindow('test', 0)
    cv2.imshow('test', imgDilate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
