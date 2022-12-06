import cv2

class LineAnnotation:
    def __init__(self,img_bgr,style):
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


