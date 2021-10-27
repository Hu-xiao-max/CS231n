import imageManip
from imageManip import *
import matplotlib.pyplot as plt
image1_path= '../images/image1.jpg'
image2_path = '../images/image2.jpg'
def display(img):
    # Show image
    plt.figure(figsize = (5,5))
    plt.imshow(img, cmap='Greys')
    '''cmaps['Sequential'] = [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']'''
    plt.axis('off')
    plt.show()
image1 = load(image1_path)
image2 = load(image2_path)

#print(image1)
#print(np.shape(image1))


grey_image =np.array(0.2125*image1[:, :, 0]+0.7154*image1[:, :, 1]+0.0721*image1[:, :, 2])
print(grey_image)
print(np.shape(grey_image))
t=np.array(grey_image)
display(t)


import cv2
cv2.imshow("img_gray", t)
cv2.waitKey()