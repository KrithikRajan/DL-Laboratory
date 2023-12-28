from PIL import Image,ImageFilter
image = Image.open("anantha.jpg")
image = image.convert("L")
image = image.filter(ImageFilter.FIND_EDGES)
image.save("output.png")
image.show("output.png")
import cv2 
img = cv2.imread("photo-1561731216-c3a4d99437d5.jpg")
t_lower = 50
t_upper = 150 
edge = cv2.Canny(img, t_lower, t_upper) 
cv2.imshow('original', img) 
cv2.imshow('edge', edge) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 
