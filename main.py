import skimage as ski
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.util import compare_images
from skimage.exposure import rescale_intensity
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import erosion, dilation
from skimage.morphology import square
from skimage import measure

def pad_num(n):
    s = str(n)
    while len(s) < 3:
        s = "0" + s
    return s

for i in range(136, 137):

    image0 = ski.io.imread(f"source/video-frame00{pad_num(i - 1)}.png")
    image1 = ski.io.imread(f"source/video-frame00{pad_num(i)}.png")
    image2 = ski.io.imread(f"source/video-frame00{pad_num(i + 1)}.png")


    grayscale0 = rgb2gray(image0)
    grayscale1 = rgb2gray(image1)
    grayscale2 = rgb2gray(image2)
    diff1 = compare_images(grayscale0, grayscale1, method='diff')
    diff2 = compare_images(grayscale1, grayscale2, method='diff')

    rediff1 = rescale_intensity(diff1, in_range='image', out_range=(0,1))
    rediff2 = rescale_intensity(diff2, in_range='image', out_range=(0,1))

    anded = np.zeros(diff2.shape)
    threshold = 0.4
    for x in range(diff2.shape[0]):
        for y in range(diff2.shape[1]):
            
            val = rediff1[x][y] * rediff2[x][y]
            if val > threshold:
                anded[x][y] = val
            
    otsu1 = threshold_otsu(image=anded)
    print(otsu1)
    otsu1 = anded > otsu1

    dilated = dilation(otsu1, square(10))
    eroded = erosion(dilated, square(7))

    # Find contours at a constant value of 0.8
    contours = measure.find_contours(eroded, 0.8)

    # Display the image and plot all contours found


    plt.show()

    fig = plt.figure(figsize=(8, 9))
    gs = GridSpec(3, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])

    ax0.imshow(image1, cmap='gray')
    ax0.set_title('Original')
    ax1.imshow(anded, cmap='gray')
    ax1.set_title('Equalized')
    ax2.imshow(otsu1, cmap='gray')
    ax2.set_title('Checkerboard comparison')
    ax3.imshow(dilated, cmap='gray')
    ax3.set_title('Checkerboard comparison')
    ax4.imshow(eroded, cmap='gray')
    ax4.set_title('Equalized')
    ax5.imshow(image1, cmap='gray')
    ax5.set_title('Equalized')

    for contour in contours:
        ax5.plot(contour[:, 1], contour[:, 0], color='green', linewidth=2)

    ax5.axis('image')
    ax5.set_xticks([])
    ax5.set_yticks([])

    # for a in (ax0, ax1, ax2):
    #     a.set_axis_off()

    # fig.tight_layout()
    plt.show()
    input()
    plt.close()

# plt.imsave(f'result/img_{pad_num(i - 1)}.png', otsu1)
