import PIL.ImageFilter
import skimage as ski
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
# from skimage.color import rgb2gray
from skimage.util import compare_images
from skimage.exposure import rescale_intensity
import numpy as np
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import erosion, dilation
from skimage.morphology import square
from skimage import measure
from time import perf_counter
from PIL.ImageOps import grayscale
from PIL import Image
from PIL.ImageFilter import GaussianBlur
from PIL.ImageChops import difference, multiply

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
    start = perf_counter()
    image0, image1, image2 = images

    grayscale0 = grayscale(Image.fromarray(image0))
    grayscale1 = grayscale(Image.fromarray(image1))
    grayscale2 = grayscale(Image.fromarray(image2))
    
    after_grayscale = perf_counter()
    
    gaussian0 = grayscale0.filter(GaussianBlur(radius=3))
    gaussian1 = grayscale1.filter(GaussianBlur(radius=3))
    gaussian2 = grayscale2.filter(GaussianBlur(radius=3))
    
    after_gaussian = perf_counter()

    diff1 = np.array(difference(gaussian0, gaussian1))/255
    diff2 = np.array(difference(gaussian1, gaussian2))/255

    print(np.max(diff1), np.max(diff2))

    after_difference = perf_counter()
    # rediff1 = np.clip(rescale_intensity(diff1, in_range='image', out_range=(0,1.5)), 0, 1)
    # rediff2 = np.clip(rescale_intensity(diff2, in_range='image', out_range=(0,1.5)), 0, 1)

    anded = np.multiply(diff1, diff2)
    
    # back view: 0.005
    # wall view: 0.0001
    
    anded[anded < 0.005] = 0

    after_and = perf_counter()
            
    otsu1 = threshold_otsu(image=anded)
    otsu1 = anded > otsu1
    
    after_otsu = perf_counter()

    dilated = gaussian(otsu1, sigma=5) > 0
    
    after_dilation = perf_counter()
    
    eroded = erosion(dilated, square(10))
    
    after_erosion = perf_counter()

    # Find contours at a constant value of 0.8
    contours = measure.find_contours(eroded, 0.8)
    
    after_contours = perf_counter()
    
    print("Grayscale:", after_grayscale - start)
    print("Gaussian:", after_gaussian - after_grayscale)
    print("Difference:", after_difference - after_gaussian)
    print("And:", after_and - after_difference)
    print("Otsu:", after_otsu - after_and)
    print("Dilation:", after_dilation - after_otsu)
    print("Erosion:", after_erosion - after_dilation)
    print("Contours:", after_contours - after_erosion)
    print("Total:", after_contours - start - (after_gaussian - after_grayscale) - (after_erosion - after_dilation))
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

    ax00.imshow(gaussian0, cmap='gray')
    ax00.set_title('Original')
    
    ax01.imshow(gaussian1, cmap='gray')
    ax01.set_title('Original')
    
    ax02.imshow(gaussian2, cmap='gray')
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
