import cv2
import numpy as np
import matplotlib.pyplot as plt

#load image
def read_file(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #Plot image
    # plt.imshow(img)
    # plt.show()
    return img

filename = "orcaHappy.jpg"
img = read_file(filename)

org_img = np.copy(img)

#increase edges for cartoon feel
#edge mask

def edge_mask(img, line_size, blur_value):
    #input - Input image
    #output - edges
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)

    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)

    
    return edges

line_size = 7
blur_value = 7
edges = edge_mask(img, line_size, blur_value)
#Plot
# plt.imshow(edges, cmap = "binary")
# plt.show()

#reduce color palette
def color_quan(img, k):
    #transform image
    data = np.float32(img).reshape((-1,3))

    #determine criteria
    crit = (cv2.TERM_CRITERIA_EPS+ cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    #implemeant k-means
    ret, label, center = cv2.kmeans(data, k, None, crit, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)

    result = center[label.flatten()]
    result = result.reshape(img.shape)

    return result

img = color_quan(img, k = 9)

# plt.imshow(img)
# plt.show()

#reduce noise
blurr = cv2.bilateralFilter(img, d = 3, sigmaColor=200, sigmaSpace=200)

# plt.imshow(blurr)
# plt.show()

#combine edge mask with color quan

def animation():
    c = cv2.bitwise_and(blurr, blurr, mask=edges)
    
    # plt.imshow(c)
    # plt.show()

    plt.subplot(1, 2, 1)
    plt.title("Animated")
    plt.imshow(c)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Original")
    plt.imshow(org_img)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
animation()


     