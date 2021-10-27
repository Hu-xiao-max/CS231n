import imageManip
from imageManip import *
import matplotlib.pyplot as plt
image1_path= '../images/image1.jpg'
image2_path = '../images/image2.jpg'
def display(img):
    # Show image
    plt.figure(figsize = (5,5))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
image1 = load(image1_path)
image2 = load(image2_path)


print(image1)
print(np.shape(image1))



print(imageManip.dim_image(image1))
display(image1)
display(image2)


new_image = dim_image(image1)
display(new_image)




without_red = rgb_exclusion(image1, 'R')
without_blue = rgb_exclusion(image1, 'B')
without_green = rgb_exclusion(image1, 'G')

print("Below is the image without the red channel.")
display(without_red)

print("Below is the image without the green channel.")
display(without_green)

print("Below is the image without the blue channel.")
display(without_blue)



grey_image =np.array(0.2125*image1[:, :, 0]+0.7154*image1[:, :, 1]+0.0721*image1[:, :, 2])
display(grey_image)
#print(np.shape(grey_image))看是否是单通道
image_l = lab_decomposition(image1, 'L')
image_a = lab_decomposition(image1, 'A')
image_b = lab_decomposition(image1, 'B')


print("Below is the image with only the L channel.")
display(image_l)

print("Below is the image with only the A channel.")
display(image_a)

print("Below is the image with only the B channel.")
display(image_b)



image_h = hsv_decomposition(image1, 'H')
image_s = hsv_decomposition(image1, 'S')
image_v = hsv_decomposition(image1, 'V')

print("Below is the image with only the H channel.")
display(image_h)

print("Below is the image with only the S channel.")
display(image_s)

print("Below is the image with only the V channel.")
display(image_v)


image_mixed = mix_images(image1, image2, channel1='R', channel2='G')
display(image_mixed)

#Sanity Check: the sum of the image matrix should be 76421.98
np.sum(image_mixed)

