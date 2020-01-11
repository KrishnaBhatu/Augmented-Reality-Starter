'''
 * BSD 3-Clause License
 * @copyright (c) 2019, Krishna Bhatu, Hrishikesh Tawade, Kapil Rawal
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * @file    Project1_Lena.py
 * @author  Krishna Bhatu, Hrishikesh Tawade, Kapil Rawal
 * @version 1.0
 * @brief  Detecting a custom AR Tag Id and placing image of lena on the Tag  
 *
 '''

import numpy as np
import cv2
import math

flag = -1
counter = 0

'''
Function Description of testVideo:
    @param path: It contains the path to video as input
    @param outputPath: It has the path where the output video will be stored.

    @return : None

    @brief : This function plays the input video
'''
def testVideo(path, outputPath):
    cap1 = cv2.VideoCapture(path)
    frame_width = int(cap1.get(3))
    frame_height = int(cap1.get(4))
    out = cv2.VideoWriter(outputPath,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    while(cap1.isOpened()):
        ret, img_rgb = cap1.read()
        if(ret == True):
            img, status = putLena(img_rgb)  
            if(status == 1):
                #cv2.namedWindow('output',cv2.WINDOW_NORMAL)
                #cv2.resizeWindow('output', 500,500)
                #cv2.imshow('output',img)
                out.write(img)
                if cv2.waitKey(20) and 0xFF == ord('q'):
                    break
            else:
                pass
        else:
            cap1.release()
            out.release()

'''
Function Description of testImage:
    @param path: It contains the path to image as input

    @return : None

    @brief :This function reads the image input and passes if the image is obtained. 
'''        
def testImage(path):
    img_rgb = cv2.imread(path,1)
    img, status = putLena(img_rgb)
    if(status == 1):
        cv2.namedWindow('output',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('output', 500,500)
        cv2.imshow('output',img)
        cv2.waitKey(0)
    else:
        pass

'''
Function Description of putLena:
    @param frame: Contains image of Lena that is to be placed on the tag 

    @return frame, inside : frames and the hierarchy value

    @brief : This function finds the countours and its parent in the tag. 
'''    
def putLena(frame):
    # Find contours
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    minPeri = 90
    maxPeri = 900
    inside = 0
    j = 0
    k = 0
    # Find parent contours and then detect tag
    for contourInfo in hierarchy[0]:
        if(contourInfo[3] == -1 and findChildren(k,hierarchy,0) >= 2):
            max_index = k
            for i in hierarchy[0]:
                if(i[3] == max_index):
                    perimeter = cv2.arcLength(contours[j],True)
                    if(perimeter > minPeri and perimeter < maxPeri):
                        inside = 1
                        cnt = contours[j]
                        frame, status = putLenaOnContour(cnt, frame)
                        j = 0
                        break
                j = j+1
        j = 0
        k = k + 1
    if(inside == 1):
        return(frame,inside)
    else:
        return(frame, 0)

'''
Function Description of putCubeOnContour:
    @param cnt: Detected countour points. 
    @param frame: Frame in which countour cnt is detected 

    @return frame, draw: Returns frame and placed lena the tag

    @brief : This function detects contours in the frame
           and calls imageOnTag place lena the tag
'''
def putLenaOnContour(cnt, frame):
    global flag 
    global counter
    # Read Marker Image
    lena =  cv2.imread('Lena.png')
    # Store ref mareker's dimensions
    ht,wt, channel = lena.shape
    # Store corners in image dimensions
    x,y,w,h = cv2.boundingRect(cnt)
    # Find important contour points only
    approx = cv2.approxPolyDP(cnt,0.05*cv2.arcLength(cnt,True),True)
    pts1 = np.zeros([4,2],dtype = 'float32')
    # check if the contour is a rectangle
    if(len(approx) == 4):
        n = 0
        # put the image points in an array
        for j in approx:
            if(n<4):
                pts1[n][0] = j[0][0]
                pts1[n][1] = j[0][1]
            n += 1
        # points of upright tag
        pts2 = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
        # world coordinates
        pts3 = np.float32([[0,0],[wt-1,0],[wt-1,ht-1],[0,ht-1]])
        # find the H matrix
        H = findHMatrix(pts2,pts1) # Transforming second to first
        # Make the tag upright
        uprightTag = cv2.warpPerspective(frame,H,(w-1,h-1))
        # Convert it to grayscale
        grayUprightTag  = cv2.cvtColor(uprightTag , cv2.COLOR_BGR2GRAY)
        # Convert to binary
        ret,binaryUprightTag  = cv2.threshold(grayUprightTag ,240,255,cv2.THRESH_BINARY)
        # Smoothen the edges
        binaryUprightTag = cv2.blur(binaryUprightTag,(5,5))
        binaryUprightTag = cv2.bilateralFilter(binaryUprightTag,5,100,100)
        # Detect the corners
        pts3, index = alignTag(uprightTag, binaryUprightTag, pts2, pts3)
        pts4 = np.roll(pts2, index, axis = 0)
        HForTag =  findHMatrix(pts4,pts2)
        rotatedTag = cv2.warpPerspective(uprightTag,HForTag,(w-1,h-1))
        # Calculate tag ID
        tagID = giveTag(rotatedTag)
        flag = tagID
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,"tagID = " + str(flag),(pts1[0][0],pts1[0][1]), font, 1.2,(0,0,255),3,cv2.LINE_AA)
        # find the H matrix
        H = findHMatrix(pts1,pts3) # Transforming second to first
        # Fit lena on to the color image
        tagSizedLena = cv2.warpPerspective(lena,H,(frame.shape[1],frame.shape[0]))
        return(imageOnTag(frame, tagSizedLena), 1)
    else:
        return(frame, 0)

'''
Function Description of findHMatrix:
    @param pts1: source points. 
    @param pts2: Destination points. 

    @return b: 3x3 Homography matrix 

    @brief :This function finds the homography between two
           points pts1 and pts2 using svd method
''' 
def findHMatrix(pts1,pts2):
        A = []
        # Image coordinates
        xc1,yc1 = pts1[0]
        xc2,yc2 = pts1[1]
        xc3,yc3 = pts1[2]
        xc4,yc4 = pts1[3]

        # World Coordinates
        xw1,yw1 = pts2[0]
        xw2,yw2 = pts2[1]
        xw3,yw3 = pts2[2]
        xw4,yw4 = pts2[3]

        # Transforming World to Image coordinates
        A = [[xw1,yw1,1,0,0,0,-xc1*xw1,-xc1*yw1,-xc1],
             [0,0,0,xw1,yw1,1,-yc1*xw1,-yc1*yw1,-yc1],
             [xw2,yw2,1,0,0,0,-xc2*xw2,-xc2*yw2,-xc2],
             [0,0,0,xw2,yw2,1,-yc2*xw2,-yc2*yw2,-yc2],
             [xw3,yw3,1,0,0,0,-xc3*xw3,-xc3*yw3,-xc3], #pdf
             [0,0,0,xw3,yw3,1,-yc3*xw3,-yc3*yw3,-yc3],
             [xw4,yw4,1,0,0,0,-xc4*xw4,-xc4*yw4,-xc4],
             [0,0,0,xw4,yw4,1,-yc4*xw4,-yc4*yw4,-yc4]]
        u, s, V = np.linalg.svd(A, full_matrices = True)

        #Converting to Hommogeneous coordinates
        a = []
        if V[8][8] == 1:
            for i in range(0,9):
                a.append(V[8][i])
        else:
            for i in range(0,9):
                a.append(V[8][i]/V[8][8])

        # H matrix in 3X3 shape
        b = np.reshape(a, (3, 3))
        return b

'''
Function Description of findChildren:
    @param k: Index of child  
    @param hierarchy: contour hierarchy
    @param childs: previous number of children

    @return childs: total children count

    @brief :This function finds children of the parent contour
'''
def findChildren(k, hierarchy,childs):
    n = 0
    for row in hierarchy[0]:
        if(row[3] == k):
            childs = childs + 1
            childs = findChildren(n, hierarchy,childs)
        n = n+1
    return childs

'''
Function Description of alignTag:
    @param uprightTag: Image of the upright tag. 
    @param binaryImage: Tags binary image 
    @param pts : points of upright tag 
    @param pts1: points of upright lena

    @return pts1, index-3: Upright image points and the index associated with it.

    @brief : This function aligns the lena image with the
             rotation of the tag.
'''         
def alignTag(uprightTag, binaryImage, pts, pts1):
    # Find contours in upright tag image
    tagContours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find biggest contour
    areas = [cv2.contourArea(c) for c in tagContours]
    max_index = np.argmax(areas)
    cnt = tagContours[max_index]
    # Smoothen the contour to find the corner points
    approx = cv2.approxPolyDP(cnt,0.1*cv2.arcLength(cnt,True),True)
    # Find the corner associated with rotation
    minDistance = 0
    firstTime = 1
    index = 0
    for corners in approx:
        x,y = corners.ravel()
        #cv2.circle(image,(x,y),2,[0,255,0],-1)
        i = 0
        for points in pts:
            i = i+1
            borderX,borderY = points
            distance = math.sqrt((borderX - x)**2 + (borderY - y)**2)
            if  distance < minDistance or firstTime:
                firstTime = 0
                minDistance = distance
                index = i
    pts1 = np.roll(pts1, index-3, axis = 0)
    return (pts1, index - 3)

'''
Function Description of imageOnTag:
    @param frame: farme containing tag. 
    @param lena: image of lena to be placed on the tag

    @return dst: lena placed on tag in frame

    @brief : This function places lena image on the tag
'''       
def imageOnTag(frame,lena):
    rows,cols,channels = lena.shape
    roi = frame[0:rows, 0:cols ]
    
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(lena,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(lena,lena,mask = mask)
    dst = cv2.add(img1_bg,img2_fg)
    
    return dst

'''
Function Description of giveTag:
    @param dstTag: upright Tag image input
    
    @return ans: Tag id 

    @brief : This function detects the tag id in the
           region of interest and calulates the id.
'''
def giveTag(dstTag):
    # divide image into eight parts
    row1 = int(dstTag.shape[0]/8)
    col1  = int(dstTag.shape[1]/8)
    reqRegion = np.zeros((4,2),dtype = 'int32')
    reqRegion[0][0] = 3*row1
    reqRegion[0][1] = 3*col1
    reqRegion[3][0] = 4*row1
    reqRegion[3][1] = 3*col1
    reqRegion[2][0] = 4*row1
    reqRegion[2][1] = 4*col1
    reqRegion[1][0] = 3*row1
    reqRegion[1][1] = 4*col1
    lst = []
    # Check the values of the encoding region
    for i in reqRegion:
            ROI = dstTag[i[0]:i[0]+row1,i[1]:i[1]+col1]
            meanL = ROI.mean(axis=0).mean(axis=0)
            mean = meanL.sum()/3
            if(mean > 240):
                    lst.append(1)
            else:
                    lst.append(0)
    ans = lst[0]*1 + lst[1]*2 + lst[2]*4 + lst[3]*8          
    return ans

# Opening Tag0.mp4 video and saving as LenaTag0.avi
testVideo('Tag0.mp4', 'LenaTag0.avi')

# Opening Tag1.mp4 video and saving as LenaTag2.avi
testVideo('Tag1.mp4', 'LenaTag1.avi')

# Opening Tag2.mp4 video and saving as LenaTag2.avi
testVideo('Tag2.mp4', 'LenaTag2.avi')

# Opening multipleTags.mp4 video and saving as LenamultipleTags.avi
testVideo('multipleTags.mp4', 'LenamultipleTags.avi')

print("All videos Processed. Please check for output videos in folder of code")
cv2.destroyAllWindows() 
