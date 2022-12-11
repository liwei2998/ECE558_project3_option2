import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import logging

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
        image = self.img
        return image

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
        lines = cv2.HoughLinesP(img_edges,1,np.pi/180,20,minLineLength=50,maxLineGap=10)
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

class VanishingPoint:
    def compute_edgelets(self,lines, sigma = 3):
        #compute locations, directions, strengths given detected lines
        #locations: ndarray of shape (n_edgelets, 2)
        #    Locations of each of the edgelets.
        #directions: ndarray of shape (n_edgelets, 2)
        #    Direction of the edge (tangent) at each of the edgelet.
        #strengths: ndarray of shape (n_edgelets,)
        #    Length of the line segments detected for the edgelet.
        locations = []
        directions = []
        strengths = []
        for p0x,p0y,p1x,p1y in lines:
            p0, p1 = np.array([p0x, p0y]), np.array([p1x, p1y])
            locations.append((p0 + p1) / 2)
            directions.append(p1 - p0)
            strengths.append(np.linalg.norm(p1 - p0))
        # convert to numpy arrays and normalize
        locations = np.array(locations)
        directions = np.array(directions)
        strengths = np.array(strengths)
        directions = np.array(directions) / \
            np.linalg.norm(directions, axis=1)[:, np.newaxis]
        return (locations, directions, strengths)

    def edgelet_lines(self, edgelets):
        #compute lines in homogenous system for edglets.
        locations, directions, _ = edgelets
        normals = np.zeros_like(directions)
        normals[:, 0] = directions[:, 1]
        normals[:, 1] = -directions[:, 0]
        p = -np.sum(locations * normals, axis=1)
        lines = np.concatenate((normals, p[:, np.newaxis]), axis=1)
        return lines

    def compute_votes(self, edgelets, model, threshold_inlier=15):
        #compute votes for each of the edgelet against a given vanishing point
        #votes for edgelets which lie inside threshold are same as their strengths, otherwise zero
        vp = model[:2] / model[2]
        locations, directions, strengths = edgelets
        est_directions = locations - vp
        dot_prod = np.sum(est_directions * directions, axis=1)
        abs_prod = np.linalg.norm(directions, axis=1) * \
            np.linalg.norm(est_directions, axis=1)
        abs_prod[abs_prod == 0] = 1e-5
        cosine_theta = dot_prod / abs_prod
        theta = np.arccos(np.abs(cosine_theta))
        theta_thresh = threshold_inlier * np.pi / 180
        return (theta < theta_thresh) * strengths

    def ransac_vanishing_point(self, edgelets, num_ransac_iter=2000, threshold_inlier=10):
        #estimate vanishing point using Ransac
        locations, directions, strengths = edgelets
        lines = self.edgelet_lines(edgelets)
        num_pts = strengths.size
        arg_sort = np.argsort(-strengths)
        first_index_space = arg_sort[:num_pts // 3]
        second_index_space = arg_sort[:num_pts // 2]
        best_model = None
        best_votes = np.zeros(num_pts)
        for ransac_iter in range(num_ransac_iter):
            ind1 = np.random.choice(first_index_space)
            ind2 = np.random.choice(second_index_space)
            l1 = lines[ind1]
            l2 = lines[ind2]
            current_model = np.cross(l1, l2)
            if np.sum(current_model**2) < 1 or current_model[2] == 0:
                # reject degenerate candidates
                continue
            current_votes = self.compute_votes(
                edgelets, current_model, threshold_inlier)

            if current_votes.sum() > best_votes.sum():
                best_model = current_model
                best_votes = current_votes
                logging.info("Current best model has {} votes at iteration {}".format(
                    current_votes.sum(), ransac_iter))
        return best_model

    def reestimate_model(self, model, edgelets, threshold_reestimate=2):
        #Reestimate vanishing point using inliers and least squares.
        locations, directions, strengths = edgelets
        inliers = self.compute_votes(edgelets, model, threshold_reestimate) > 0
        locations = locations[inliers]
        directions = directions[inliers]
        strengths = strengths[inliers]
        lines = self.edgelet_lines((locations, directions, strengths))
        a = lines[:, :2]
        b = -lines[:, 2]
        est_model = np.linalg.lstsq(a, b, rcond='warn')[0]
        return np.concatenate((est_model, [1.]))
    
    def remove_inliers(self, model, edgelets, threshold_inlier=2):
    #Remove all inlier edglets of a given model.
        inliers = self.compute_votes(edgelets, model) > 0
        locations, directions, strengths = edgelets
        locations = locations[~inliers]
        directions = directions[~inliers]
        strengths = strengths[~inliers]
        edgelets = (locations, directions, strengths)
        return edgelets

    def vis_edgelets(self, image, edgelets, show=True):
        #Helper function to visualize edgelets
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        locations, directions, strengths = edgelets
        for i in range(locations.shape[0]):
            xax = [locations[i, 0] - directions[i, 0] * strengths[i] / 2,
                   locations[i, 0] + directions[i, 0] * strengths[i] / 2]
            yax = [locations[i, 1] - directions[i, 1] * strengths[i] / 2,
                   locations[i, 1] + directions[i, 1] * strengths[i] / 2]

            plt.plot(xax, yax, 'r-')

        if show:
            plt.show()

    def vis_model(self, image, edgelets, final_model, show=True):
        #Helper function to visualize computed model.
        locations, directions, strengths = edgelets
        inliers = self.compute_votes(edgelets, final_model) > 0

        edgelets = (locations[inliers], directions[inliers], strengths[inliers])
        locations, directions, strengths = edgelets
        self.vis_edgelets(image, edgelets, False)
        vp = final_model / final_model[2]
        plt.plot(vp[0], vp[1], 'bo')
        for i in range(locations.shape[0]):
            xax = [locations[i, 0], vp[0]]
            yax = [locations[i, 1], vp[1]]
            plt.plot(xax, yax, 'b-.')

        if show:
            plt.show()
    
    
if __name__ == '__main__':
    ############# class testing
    img = cv2.imread('box.jpg')
    
    ###LineDetector
    lsd = LineDetector()
    image = lsd.init(img)
    #lines = lsd.detect(mask_on=False,self_adjust=False) # original image
    lines = lsd.detect(self_adjust=False) # masked image with default mask
    #lines = lsd.detect(self_adjust=True) # masked image with self-adjusted mask
    vp = VanishingPoint()
    edgelets1 = vp.compute_edgelets(lines)
    vp1 = vp.ransac_vanishing_point(edgelets1)
    vp1 = vp.reestimate_model(vp1, edgelets1)
    vp.vis_model(image, edgelets1, vp1)
    
    edgelets2 = vp.remove_inliers(vp1, edgelets1)
    vp2 = vp.ransac_vanishing_point(edgelets2, num_ransac_iter=2000)
    vp2 = vp.reestimate_model(vp2, edgelets2)
    vp.vis_model(image, edgelets2, vp2) # Visualize the vanishing point model

    edgelets3 = vp.remove_inliers(vp2, edgelets2)
    vp3 = vp.ransac_vanishing_point(edgelets3, num_ransac_iter=2000)
    vp3= vp.reestimate_model(vp3, edgelets3)
    vp.vis_model(image, edgelets3, vp3) # Visualize the vanishing point model
