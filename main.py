
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
# from skimage.color import rgb2gray

import numpy as np
import math
from time import perf_counter
import cv2 as cv

import checkRegion

def pad_num(n):
    s = str(n)
    while len(s) < 4:
        s = "0" + s
    return s

BACK_CAMERA = 0
FRONT_WALL = 1
FLOOR = 2

def get_ball_position(images, camera, debug):
    start = perf_counter()
    image0, image1, image2 = images

    grayscale0 = cv.cvtColor(image0, cv.COLOR_BGR2GRAY)
    grayscale1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    grayscale2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    
    after_grayscale = perf_counter()
    
    gaussian0 = cv.GaussianBlur(grayscale0, (3, 3), 0)
    gaussian1 = cv.GaussianBlur(grayscale1, (3, 3), 0)
    gaussian2 = cv.GaussianBlur(grayscale2, (3, 3), 0)
    
    after_gaussian = perf_counter()

    diff1 = cv.absdiff(gaussian0, gaussian1)
    diff2 = cv.absdiff(gaussian1, gaussian2)

    after_difference = perf_counter()
    # rediff1 = np.clip(rescale_intensity(diff1, in_range='image', out_range=(0,1.5)), 0, 1)
    # rediff2 = np.clip(rescale_intensity(diff2, in_range='image', out_range=(0,1.5)), 0, 1)

    anded = cv.bitwise_and(diff1, diff2)
    
    # back view: 60
    # wall view: 12
    # print(anded.max(), "anded max")
    threshold = 0
    if camera == BACK_CAMERA:
        threshold = 60
    elif camera == FRONT_WALL:
        threshold = 25
    
    
    anded[anded < threshold] = 0

    after_and = perf_counter()
            
    _, otsu = cv.threshold(anded, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    after_otsu = perf_counter()

    dilated = cv.dilate(otsu, cv.getStructuringElement(cv.MORPH_RECT, (45, 45)))
    
    after_dilation = perf_counter()
    
    eroded = cv.erode(dilated, cv.getStructuringElement(cv.MORPH_RECT, (10, 10)))
    
    after_erosion = perf_counter()

    # Find contours at a constant value of 0.8
    _, thresh = cv.threshold(eroded, 0.8*255, 255, 0)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    after_contours = perf_counter()
    
    data = []
    for cnt in contours:
        m = cv.moments(cnt)
        cx = int(m['m10']/m['m00'])
        cy = int(m['m01']/m['m00'])
        data.append((cnt, cx, cy))
    
    ball_candidates = []
    
    contoured = image1.copy()
    
    min_area = 0
    max_area = 0
    if camera == BACK_CAMERA:
        min_area = 1200
        max_area = 1800
    elif camera == FRONT_WALL:
        min_area = 1000
        max_area = 8000
    elif camera == FLOOR:
        min_area = 2000
        max_area = 3000
        
    
    for cnt,cx,cy in data:
        area = cv.contourArea(cnt)
        # print(area)
        if min_area < area < max_area:
            ball_candidates.append((cnt,cx,cy))
    
    if camera == FRONT_WALL and len(contours) > 1:
        ball_candidates = []
    
    if len(ball_candidates) == 2 and camera == BACK_CAMERA:
        if ball_candidates[0][2] < ball_candidates[1][2]:
            ball_candidates.pop()
        else:
            ball_candidates.pop(0)
    elif len(ball_candidates) > 2:
        best_guess = None
        max_dist = 0
        
        for cnt,cx1,cy1 in ball_candidates:
            total_dist = 0
            
            for _,cx2,cy2 in data:
                dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                total_dist += dist
            if total_dist > max_dist:
                max_dist = total_dist
                best_guess = (cnt,cx1,cy1)
        ball_candidates = [best_guess]        
        
        
    centre = None
    if len(ball_candidates) == 1:
        m = cv.moments(cnt)
        
        _,cx,cy = ball_candidates[0]
        
        centre = cx, cy
    
        if debug:
            cv.drawContours(contoured, [ball_candidates[0][0]], -1, (0,255,0), 10)
    
    if debug:
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
        ax20.set_title('Normalised1')
        # ax21.imshow(rediff2, cmap='gray')
        ax21.set_title('Normalised2')
        ax30.imshow(anded, cmap='gray')
        ax30.set_title('Anded')
        ax31.imshow(otsu, cmap='gray')
        ax31.set_title('Otsu')
        ax12.imshow(dilated, cmap='gray')
        ax12.set_title('Dilated')
        ax22.imshow(eroded, cmap='gray')
        ax22.set_title('Eroded')
        ax32.imshow(contoured)
        ax32.set_title('Contours')
        
        fig.tight_layout()
        plt.show()
    
    # cv.imwrite(f"result/img{pad_num(i)}.png", contoured)
    
    return centre

bounced_wall = False
previous_wall_x = float('inf')
previous_floor_pos = None
bounced_floor = False

def check_bounce(images, debug=False):
    global previous_wall_x
    global previous_floor_y
    global bounced_wall
    global previous_floor_pos
    global bounced_floor
    
    before = images[:3]
    current = images[3:6]
    after = images[6:]

    back_images = [before[BACK_CAMERA], current[BACK_CAMERA], after[BACK_CAMERA]]
    floor_images = [before[FLOOR], current[FLOOR], after[FLOOR]]
    front_images = [before[FRONT_WALL], current[FRONT_WALL], after[FRONT_WALL]]
    
    wall_centre = get_ball_position(front_images, FRONT_WALL, debug=False)
    if wall_centre is not None:
        if not bounced_wall:
            if wall_centre[0] < previous_wall_x:
                previous_wall_x = wall_centre[1]
            else:
                ball_pos = get_ball_position(back_images, BACK_CAMERA, debug=debug)
                bounced_wall = True
                
                return (ball_pos, BACK_CAMERA)
    else:
        previous_wall_x = float('inf')
        bounced_wall = False
    
    floor_centre = get_ball_position(floor_images, FLOOR, debug=False)
    if floor_centre is not None:
        
        # dist = math.sqrt((floor_centre[0] - previous_floor_pos[0]) ** 2 + (floor_centre[1] - previous_floor_pos[1]) ** 2)
        # if dist < 1000:
        if floor_centre[1] < previous_floor_pos[1]:
            bounced_floor = False
            previous_floor_pos = floor_centre
            # previous_floor_pos = floor_centre
        else:
            previous_floor_pos = floor_centre
            if not bounced_floor:
                bounced_floor = True
            # ball_pos = get_ball_position([before[BACK_CAMERA], current[BACK_CAMERA], after[BACK_CAMERA]], BACK_CAMERA)
                return ((1,1), FLOOR)
    else:
        previous_floor_pos = (float('inf'), float('inf'))
        bounced_floor = False

    # centre = get_ball_position([before[FRONT_WALL], current[FRONT_WALL], after[FRONT_WALL]], FRONT_WALL)
    # if centre is not None:
    #     if centre[0] < previous_x:
    #         previous_x = centre[0]
    #     else:
    #         # Bounce happened
    #         ball_pos = get_ball_position([before[BACK_CAMERA], current[BACK_CAMERA], after[BACK_CAMERA]], BACK_CAMERA)
    #         previous_x = float('inf')
            
    #         return ball_pos

for i in range(2, 5172):
    images = [
        cv.imread(f"../Videos/Back/video-frame0{pad_num(i - 1)}.png"),
        cv.imread(f"../Videos/Front/video-frame0{pad_num(i - 1)}.png"),
        cv.imread(f"../Videos/Floor/video-frame0{pad_num(i - 1)}.png"),
        cv.imread(f"../Videos/Back/video-frame0{pad_num(i)}.png"),
        cv.imread(f"../Videos/Front/video-frame0{pad_num(i)}.png"),
        cv.imread(f"../Videos/Floor/video-frame0{pad_num(i)}.png"),
        cv.imread(f"../Videos/Back/video-frame0{pad_num(i + 1)}.png"),
        cv.imread(f"../Videos/Front/video-frame0{pad_num(i + 1)}.png"),
        cv.imread(f"../Videos/Floor/video-frame0{pad_num(i + 1)}.png"),
    ]
    
    pos = check_bounce(images, False)
    if pos is not None:
        print(pos[0], 2 if pos[1] == 0 else 0, (i/60.0))
        print(checkRegion.checkRegion(pos[0], 2 if pos[1] == 0 else 0, time=i/60.0))
        
        