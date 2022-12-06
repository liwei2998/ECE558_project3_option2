import utils
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ########Preparation: instantiate helpful classes
    hp = utils.helper()
    la = utils.LineAnnotation()

    # Step 1: Image acquisition
    img = cv2.imread('box.jpg')

    # Step 2.1: Image annotation, choose lines either by diy or by default
    la.init(img,style='default')
    lines = la.draw_lines()
    img=la.draw_image()
    cv2.imshow('annotated image',img)
    cv2.waitKey() # show the annotated image, press q to exit 
    cv2.destroyAllWindows()
    lines = np.array(lines)
    lines = lines[[0,3,1,4,2,5],:] # permutation, 0-2: red, 2-4:green, 4-6:blue

    # Step 2.2: Compute vanishing points
    vps = []
    for i in range(0,len(lines),2):
        e11 = np.append(lines[i][0],1)
        e12 = np.append(lines[i][1],1)
        l1 = np.cross(e11,e12)
        e21 = np.append(lines[i+1][0],1)
        e22 = np.append(lines[i+1][1],1)
        l2 = np.cross(e21,e22)    
        vp = np.cross(l1,l2)
        vp = vp/vp[2]
        vps.append(vp)
    vps = np.array(vps)

    # Step 2.3: Compute projection matrix
    # a) compute focal length f (Pythagoras) and intrisic matrix
    orthocenter = hp.ortho(vps)
    dis = hp.distanceP2L(orthocenter[:2],vps[1][:2],vps[2][:2])
    fp = hp.footPoint(vps[0][:2],orthocenter[:2],dis)
    f = np.sqrt(np.linalg.norm(fp-vps[1][:2])*np.linalg.norm(fp-vps[2][:2])-dis*dis)
    u0 = 320 # ideally u0 and v0 are located at the center
    v0 = 240
    K = np.array([[f,0,u0],[0,f,v0],[0,0,1]]) # intrisic matrix

    # b) compute rotation matrix
    r2 = np.dot(np.linalg.inv(K),vps[1])
    r2 = r2/np.linalg.norm(r2)
    r1 = np.dot(np.linalg.inv(K),vps[0])
    r1 = r1/np.linalg.norm(r1)
    #r3 could be computed either by cross r1 & r2, or using the third vanishing point, \
    # be careful about the direction of r3 if using vanishing point
    r3 = np.cross(r1,r2) 
    R = np.array([r1,r2,r3]) # rotation matrix

    # c) compute translation matrix
    ow = np.array([0,0,0,1]) #origin in world coordinate
    oc = (lines[0][0]+lines[2][0]+lines[4][0])/3 #origin in image coordinate
    oc = np.append(oc,1.0)
    t_eq1 = np.dot(np.linalg.inv(K),oc.reshape(3,1)) #t_eq1=[t1/t3,t2/t3,1]
    # The most tricky part, scale pw to make projected picture looks nicer.
    pw = np.array([240,0,0,1]) #point in world coordinate (14.2 is the length of the box)
    pc = lines[0][1] #corresponding point in image coordinate
    pc = np.append(pc,1.)
    # t_eq2=[(14.2*r11+t1)/(14.2*r13+t3),(14.2*r12+t2)/(14.2*r13+t3),1]
    t_eq2 = np.dot(np.linalg.inv(K),pc.reshape(3,1)) 
    # Simple (trival) math to solve t1,t2,t3 using the two equations about t
    t3 = (-240*r1[0]+240*r1[2]*t_eq2[0][0])/(t_eq1[0][0]-t_eq2[0][0])
    t1 = t3*t_eq1[0][0]
    t2 = t3*t_eq1[1][0]
    T = np.array([t1,t2,t3])

    # d) obtain projection matrix (finally!) and homography matrix
    P_mat = np.append(R,T.reshape(1,3),axis=0)
    P_mat = np.transpose(P_mat)
    P_mat = np.dot(K,P_mat)
    hxy = P_mat[:,[0,1,3]]
    hyz = P_mat[:,[1,2,3]]
    hzx = P_mat[:,[0,2,3]]

    # Step 3: Compute texture maps
    src_img = img.copy()
    rows, cols = src_img.shape[:2]

    # Using opencv warpPerspective function, the homography obtained before is to \
    # transform from world to image. To transform from inage to world, we need to \
    # use the inverse homography matrix.
    img_xy = cv2.warpPerspective(src_img, np.linalg.inv(hxy), (cols, rows))
    cv2.imshow("texture image xy plane", img_xy)
    cv2.waitKey()
    img_yz = cv2.warpPerspective(src_img, np.linalg.inv(hyz), (cols, rows))
    cv2.imshow("texture image yz plane", img_yz)
    cv2.waitKey()
    img_zx = cv2.warpPerspective(src_img, np.linalg.inv(hzx), (cols, rows))
    cv2.imshow("texture image zx plane", img_zx)
    cv2.waitKey()
    cv2.destroyAllWindows()






    

