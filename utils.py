import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

class LineAnnotation:
    def init(self,img_bgr,style):
        self.lines = []
        self.image = cv2.resize(img_bgr,(640,480))
        self.start_pos = None
        self.end_pos = None
        self.is_drawing = None
        self.style = style

    def draw_lines(self, window_name='roi'):
        if self.style == 'diy':
            img_bgr = self.image.copy()
            cv2.imshow(window_name, img_bgr)
            cv2.setMouseCallback(window_name, self.mouse_handler)
            while cv2.waitKey(50) & 0xFF != ord('q'):
                if self.start_pos is not None and self.end_pos is not None:
                    x, y = self.start_pos
                    cv2.putText(img_bgr, f'{x},{y}', (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
                    # cv2.line(self.image, self.start_pos, self.end_pos, (0, 0, 255))
                    cv2.arrowedLine(img_bgr, self.start_pos, self.end_pos, (0, 0, 255))
                    x, y = self.end_pos
                    cv2.putText(img_bgr, f'{x},{y}', (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
                    cv2.imshow(window_name, img_bgr)
                img_bgr = self.image.copy()
                cv2.waitKey(30)
            cv2.setMouseCallback(window_name, self.empty_handler)
            cv2.destroyWindow(window_name)
            return self.lines    
        else:
            self.lines = [[(259, 406), (451, 321)], [(260, 407), (194, 298)], \
                          [(259, 407), (265, 257)], [(184, 160), (393, 124)], \
                          [(184, 160), (264, 258)], [(185, 160), (196, 301)]]
            return self.lines
        
    def empty_handler(self, event, x, y, flags, params):
        pass

    def mouse_handler(self, event, x, y, flags, params):
        self.mouse_callback(event, x, y, flags, params)

    def mouse_callback(self, event, x, y, flags, params):
        if event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing:
                self.end_pos = (x,y)
        if event == cv2.EVENT_LBUTTONDOWN:
            # left single click
            self.is_drawing = True
            self.start_pos = (x, y)
        if event == cv2.EVENT_LBUTTONUP:
            # left release
            self.is_drawing = False
            if self.start_pos is not None and self.end_pos is not None:
                self.lines.append([self.start_pos, self.end_pos])
            self.start_pos = None
            self.end_pos = None
    
    def draw_image(self):
        img = self.image.copy()
        colors = [(34,34,178),(35,142,107),(139,139,0)] # red,green, blue
        for i in range(len(self.lines)):
            line = self.lines[i]
            start = line[0] 
            end = line[1]
            cv2.line(img,start,end,tuple(colors[i%3]),thickness=3)
        return img

class helper:
    def ortho(self,pts):
        # compute orthocenter given 3 points
        x1 = pts[0][0]; y1 = pts[0][1]
        x2 = pts[1][0]; y2 = pts[1][1]
        x3 = pts[2][0]; y3 = pts[2][1]
        A1 = y2-y3; B1 = x3-x2
        A2 = y1-y3; B2 = x3-x1
        C1 = A1*y1-B1*x1; C2 = A2*y2-B2*x2
        x = (A1*C2-A2*C1)/(A2*B1-A1*B2)
        y = (B1*C2-B2*C1)/(A2*B1-A1*B2)
        return np.array([x,y,1])

    def distanceP2L(self,pt,line_pt1,line_pt2):
        # Return distance from point to line, using area
        vec1 = line_pt1 - pt
        vec2 = line_pt2 - pt
        distance = np.abs(np.cross(vec1,vec2)) / np.linalg.norm(line_pt1-line_pt2)
        return distance

    def footPoint(self,pt,pto,dis):
        # Return foot point given one vertex, orthocenter and distance \
        # from orthocenter to another edge.
        direction = (pto-pt)/np.linalg.norm(pto-pt)
        return pto+dis*direction

class LineDetector:
    def init(self,img):
        self.img = cv2.resize(img,(640,480))

    def detect(self,mask_on=True,self_adjust=False):
        # a) Add masks when the image boundry colors are hard to tell
        if mask_on:
            if self_adjust:
                masked_img = self.color_filter()
            else:
                lower = np.array([16,53,45])
                upper = np.array([97,118,255])
                img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
                masked_img = cv2.inRange(img_hsv,lower,upper)
                cv2.imshow('masked image',masked_img)
                cv2.waitKey()
                cv2.destroyAllWindows()
        else:
            masked_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            cv2.imshow('original image',masked_img)
            cv2.waitKey()
            cv2.destroyAllWindows()

        # b) Blur the image to de-noise
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(masked_img,(kernel_size, kernel_size),0)

        # c) Detect edges using Canny
        low_thresh = 60
        high_thresh = 150
        img_edges = cv2.Canny(blur_gray, low_thresh, high_thresh)
        cv2.imshow('canny detection',img_edges)
        cv2.waitKey()
        cv2.destroyAllWindows()

        # d) Obtain lines using HoughlineP
        lines = cv2.HoughLinesP(img_edges,1,np.pi/180,20,minLineLength=20,maxLineGap=50)
        lines = lines.reshape((lines.shape[0],-1))
        img_lines = copy.deepcopy(self.img)
        for line in lines:
            cv2.line(img_lines,(line[0],line[1]),(line[2],line[3]),(80,106,238),1)

        cv2.imshow("hough line detection",img_lines)
        cv2.waitKey(0)   
        cv2.destroyAllWindows()

        return lines     

    def color_filter(self):
        # This is to get HSV color from a given image, used for color filtering
        frame = copy.deepcopy(self.img)
        def nothing(x):
            pass
        cv2.namedWindow("Trackbars",)
        cv2.createTrackbar("lh","Trackbars",0,179,nothing)
        cv2.createTrackbar("ls","Trackbars",0,255,nothing)
        cv2.createTrackbar("lv","Trackbars",0,255,nothing)
        cv2.createTrackbar("uh","Trackbars",179,179,nothing)
        cv2.createTrackbar("us","Trackbars",255,255,nothing)
        cv2.createTrackbar("uv","Trackbars",255,255,nothing)
        while True:
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

            lh = cv2.getTrackbarPos("lh","Trackbars")
            ls = cv2.getTrackbarPos("ls","Trackbars")
            lv = cv2.getTrackbarPos("lv","Trackbars")
            uh = cv2.getTrackbarPos("uh","Trackbars")
            us = cv2.getTrackbarPos("us","Trackbars")
            uv = cv2.getTrackbarPos("uv","Trackbars")

            lower = np.array([lh,ls,lv])
            upper = np.array([uh,us,uv])
            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_or(frame,frame,mask=mask)

            cv2.imshow("result",result)
            cv2.imshow("mask",mask)
            key = cv2.waitKey(1)
            #press esc to exit
            if key == 27:
                break
        cv2.destroyAllWindows()
        result = cv2.cvtColor(frame,cv2.COLOR_HSV2BGR)
        return mask

if __name__ == '__main__':
    ############# class testing
    img = cv2.imread('box.jpg')
    

    ###LineDetector
    lsd = LineDetector()
    lsd.init(img)
    lsd.detect(mask_on=False,self_adjust=False) # original image
    # lsd.detect(self_adjust=False) # masked image with default mask
    # lsd.detect(self_adjust=True) # masked image with self-adjusted mask


