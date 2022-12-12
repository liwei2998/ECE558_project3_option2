import utils
import cv2
import numpy as np
import os
os.system('cls')

if __name__ == '__main__':
    ########Preparation: instantiate helpful classes
    hp = utils.helper()

    # Step 1: Image acquisition
    img = cv2.imread('box.png')
    
    # Step 2.1: Line segmentation with canny and houghline. Add mask for mcolor filtering. 
    lsd = utils.LineDetector()
    image = lsd.init(img)
    # detect_lines = lsd.detect(mask_on=False,self_adjust=False) # original image
    detect_lines = lsd.detect(self_adjust=False) # masked image with default mask
    # detect_lines = lsd.detect(self_adjust=True) # masked image with self-adjusted mask
    
    #Step 2.2: Compute vanishing point using LSD and RANSAC
    vps = []
    vp = utils.VanishingPoint()
    edgelets = vp.compute_edgelets(detect_lines)
    model = vp.ransac_vanishing_point(edgelets,20)
    model = vp.reestimate_model(model, edgelets,20)
    vp.vis_model(image, edgelets, model, show = True)
    vps.append(model)
    for i in range(2):
        edgelets = vp.remove_inliers(model, edgelets,20)
        model = vp.ransac_vanishing_point(edgelets,20)
        model = vp.reestimate_model(model, edgelets, 20)
        vp.vis_model(image, edgelets, model)
        vps.append(model)
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
    # r3 could be computed either by cross r1 & r2, or using the third vanishing point, \
    # be careful about the direction of r3 if using vanishing point
    r3 = np.cross(r1,r2) 
    R = np.array([r1,r2,r3]) # rotation matrix

    # c) compute translation matrix
    ow = np.array([0,0,0,1]) #origin in world coordinate
    oc = np.array([259.3, 406.7, 1])
    t_eq1 = np.dot(np.linalg.inv(K),oc.reshape(3,1)) #t_eq1=[t1/t3,t2/t3,1]
    # The most tricky part, scale pw to make projected picture looks nicer.
    pw = np.array([240,0,0,1]) #point in world coordinate (14.2 is the length of the box)
    pc = np.array([451.,321.,1.]) #corresponding point in image coordinate
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
    cv2.imwrite('xy_texture_crop.png', img_xy)
    cv2.waitKey()
    img_yz = cv2.warpPerspective(src_img, np.linalg.inv(hyz), (cols, rows))
    cv2.imshow("texture image yz plane", img_yz)
    cv2.imwrite('yz_texture_crop.png', img_yz)
    cv2.waitKey()
    img_zx = cv2.warpPerspective(src_img, np.linalg.inv(hzx), (cols, rows))
    cv2.imshow("texture image zx plane", img_zx)
    cv2.imwrite('zx_texture_crop.png', img_zx)
    cv2.waitKey()
    cv2.destroyAllWindows()