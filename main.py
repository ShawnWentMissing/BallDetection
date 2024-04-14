import skimage as ski
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.util import compare_images
from skimage.exposure import rescale_intensity
import numpy as np
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import erosion, dilation
from skimage.morphology import square
from skimage import measure
from time import perf_counter

def pad_num(n):
    s = str(n)
    while len(s) < 3:
        s = "0" + s
    return s

i = 136
images = [
    ski.io.imread(f"source/video-frame00{pad_num(i - 1)}.png"),
    ski.io.imread(f"source/video-frame00{pad_num(i)}.png"),
    ski.io.imread(f"source/video-frame00{pad_num(i + 1)}.png")
]

def get_ball_position(court, valid_region, valid_serve_region, images):
    a = perf_counter()
    image0, image1, image2 = images

    grayscale0 = gaussian(rgb2gray(image0), sigma=3)
    grayscale1 = gaussian(rgb2gray(image1), sigma=3)
    grayscale2 = gaussian(rgb2gray(image2), sigma=3)
    
    g = perf_counter()

    diff1 = np.absolute(grayscale0 - grayscale1)
    diff2 = np.absolute(grayscale1 - grayscale2)

    print(np.max(diff1), np.max(diff2))
    print(np.min(diff1), np.min(diff2))
# 6.803830177740122e-05
# 5.162638454849153e-05

    d = perf_counter()
    # rediff1 = np.clip(rescale_intensity(diff1, in_range='image', out_range=(0,1.5)), 0, 1)
    # rediff2 = np.clip(rescale_intensity(diff2, in_range='image', out_range=(0,1.5)), 0, 1)

    anded = np.multiply(diff1, diff2)
    anded[anded < 0.0001] = 0
    print(np.min(anded), np.max(anded))    
    ande = perf_counter()
            
    otsu1 = threshold_otsu(image=anded)
    otsu1 = anded > otsu1
    
    ott = perf_counter()

    dilated = gaussian(otsu1, sigma=5) > 0
    
    dit = perf_counter()
    
    eroded = erosion(dilated, square(10))
    
    erot = perf_counter()

    # Find contours at a constant value of 0.8
    contours = measure.find_contours(eroded, 0.8)
    
    cont = perf_counter()
    print(g - a)
    print(d - g)
    print(ande - d)
    print(ott -  ande)
    print(dit - ott)
    print(erot - dit)
    print(cont - erot)
    # Display the image and plot all contours found


    plt.show()

    fig = plt.figure(figsize=(8,9))
    gs = GridSpec(4, 3)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2])
    ax20 = fig.add_subplot(gs[2, 0])
    ax21 = fig.add_subplot(gs[2, 1])
    ax22 = fig.add_subplot(gs[2, 2])
    ax30 = fig.add_subplot(gs[3, 0])
    ax31 = fig.add_subplot(gs[3, 1])
    ax32 = fig.add_subplot(gs[3, 2])

    ax00.imshow(grayscale0, cmap='gray')
    ax00.set_title('Original')
    
    ax01.imshow(grayscale1, cmap='gray')
    ax01.set_title('Original')
    
    ax02.imshow(grayscale2, cmap='gray')
    ax02.set_title('Original')
    
    ax10.imshow(diff1, cmap='gray')
    ax10.set_title('Diff1')
    ax11.imshow(diff2, cmap='gray')
    ax11.set_title('Diff2')
    # ax20.imshow(rediff1, cmap='gray')
    # ax20.set_title('Normalised1')
    # ax21.imshow(rediff2, cmap='gray')
    ax21.set_title('Normalised2')
    ax30.imshow(anded, cmap='gray')
    ax30.set_title('Anded')
    ax31.imshow(otsu1, cmap='gray')
    ax31.set_title('Otsu')
    ax12.imshow(dilated, cmap='gray')
    ax12.set_title('Dilated')
    ax22.imshow(eroded, cmap='gray')
    ax22.set_title('Eroded')
    ax32.imshow(image1, cmap='gray')
    ax32.set_title('Contours')
    

    for contour in contours:
        ax32.plot(contour[:, 1], contour[:, 0], color='green', linewidth=2)

    ax32.axis('image')
    ax32.set_xticks([])
    ax32.set_yticks([])

    # for a in (ax0, ax1, ax2):
    #     a.set_axis_off()

    # fig.tight_layout()
    plt.show()

get_ball_position(1,1,1,images)
    # plt.imsave(f'result/img_{pad_num(i - 1)}.png', otsu1)
