import cv2
import numpy as np
import math
from line_merging import *

def setup(src="court.png"):
    lines = process_lines(src)
    for line in lines:
        y1, x1 = line[0]
        y2, x2 = line[1]
        slope = (y2 - y1) / (x2 - x1) if (x2-x1)!=0 else (y2 - y1) / 0.000001
        intercept = y1 - slope * x1
        line.append(slope)
        line.append(intercept)
    
    linesDict = name_lines(lines)
    return linesDict

def name_lines(lines):
    l = {"leftwall":-1, "rightwall":-1, "upper-back":-1, "middle-back":-1, "lower-back":-1, "middle-floor":-1}
    #determine which line is which by comparing their relative y-intercepts, slopes, etc.
    #relatively simple process, but for now just hardcode:
  
    l["leftwall"] = lines[0]
    l["rightwall"] = lines[5]

    l["middle-floor"] = lines[1]

    l["upper-back"] = lines[2]
    l["middle-back"] = lines[3]
    l["lower-back"] = lines[4]

    for i in range(6, len(lines)):
        s = "other" + str(i)
        l[s] = lines[i]
 
    return l

def get_region(point, linesDict, wall=0):
    messages = {0:"OutsideBoundary", 1:"FrontWithinBoundaryServe", 2:"FrontWithinBoundary"}
    #-> int wall: indicates which wall the ball bounced off
    #-> int,int point: co-ords of ball on collision
    #<- string: output region
    row, col = point

    #case 1: ball bounces of back wall
    if wall==0:
        #ball could have bounced in one of three regions
        #check if ball is above top line
        top = linesDict["upper-back"]
        # get co-ords for line given x position
        row2 = top[2]*col + top[3]
        if(row<=row2): # above top line 
            return messages[0]
        # else below top line:
        # check above middle line
        middle = linesDict["middle-back"]
        row3 = middle[2]*col + top[3]
        if(row<row3): # Above middle line
            return messages[1]
        # else below middle line
        bottom = linesDict["lower-back"]
        row4 = bottom[2]*col + bottom[3]
        if(row<row4):
            return messages[2]
        # below bottom line
        return messages[1]

    #case 2: ball bounces off floor
    if wall==1:
        # check if it is above the mid line on the floor
        middle = linesDict["middle-floor"]
        row2 = middle[2]*col + middle[3]
        return messages[2] if row<=row2 else messages[1]

    #case 3: ball bounces off any other surface (just check it is below the line to be legal)
    else:
        # check which wall it bounced off
        left = linesDict["leftwall"]
        right = linesDict["rightwall"]
        # check the ranges of x values contained within left and right walls
        # wont overlap due to back wall
        # for now take shortcut, see which edge the point is closer to:
        if(col < right[1][1]//2):
            # left wall
            row2 = left[2]*col + left[3]
            return messages[0] if row<=row2 else messages[2]
        else:
            row2 = left[2]*col + left[3]
            return messages[0] if row<=row2 else messages[2]

#Testing:
def test():
    src = "court.png"
    linesDict = setup(src)
    img = cv2.imread(src)

    for key in linesDict.keys():
        line = linesDict[key]
        cv2.line(img, (line[0][0], line[0][1]), (line[1][0],line[1][1]), (0,0,255), 6)
    cv2.imwrite('prediction/test.jpg',img)

def test2():
    src = "court.png"
    lineDict = setup(src)

    img = cv2.imread(src)

    for key in lineDict.keys():
        colour = (0,0,255) if key[0]=="o" else (0,255,0)
        line = lineDict[key]
        cv2.line(img, (line[0][0], line[0][1]), (line[1][0],line[1][1]), colour, 10)

    cv2.imwrite('prediction/test.jpg',img)

def test3():
    src = "court.png"
    img = cv2.imread(src)
    print(img.shape)

    lineDict = setup(src)

    # 0=back, 1=floor, 2=wall
    coords = [
        (0,0, 2),
        (100,100, 2),
        (200,1000, 2),
        (1000,1000, 1),
        (1000,1200, 1)
        #(0,10),
        #(150,250),
        #(700, 1600)
    ]

    for c in coords:
        r = get_region(c[0:2], lineDict, c[2])
        print(c,end=" ")
        print(r)
        cv2.drawMarker(img, (c[0], c[1]),(0,0,255), markerType=cv2.MARKER_STAR, 
            markerSize=40, thickness=2, line_type=cv2.LINE_AA)
    
    cv2.imwrite('prediction/test2.jpg',img)
    
def checkRegion(coord, wall, src="court.png", time=0):
    lineDict = setup(src)
    output = get_region(coord, lineDict, wall=wall)
    return (time, output)

# c = (100,100)
# img = cv2.imread("test_img.png")
# out = checkRegion(c,2,frame=img)
# print(out)

