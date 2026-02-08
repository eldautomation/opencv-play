import cv2
print("This is printing!")


img_grayscale = cv2.imread('test.jpg',0)
print("grayscale")
print(img_grayscale)
print("Greyscale now")

#cv2.imshow('graycsale image',img_grayscale)
#cv2.waitKey(0)

#cv2.destroyAllWindows()
#cv2.imwrite('grayscale.jpg',img_grayscale)