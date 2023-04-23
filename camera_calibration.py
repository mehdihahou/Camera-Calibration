import numpy as np
import cv2 as cv
import glob
import yaml

# termination criteria (the maximum number of iterations and/or the desired accuracy): 
# In this case the maximum number of iterations is set to 30 and epsilon = 0.001
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
path="data_image"
images=glob.glob(path+"\\*.png")

# cnt for counting images
cnt=1
for i in (images):
    
    img=cv.imread(i)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#    Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,5), None) #you need to enter number of (Black square -1, white square -1) of your own chessBoard
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,5), corners2, ret)
        # cv.imshow(f"image {cnt}",img)
        cnt=cnt+1
        cv.waitKey(0)
   

cv.destroyAllWindows()
_, mtx, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print(f"Matrix:\n{mtx}")
print("********** Camera matrix parametre (Intrensics prm) *************")
print(f"fx:{mtx[0][0]},\nfy:{mtx[1][1]},\nCx:{mtx[0][2]},\nCy{mtx[1][2]}")
print("********** distortion coefficients *************")
print(dist)
print("**********  *************")

cal_data={"camera_matrix": np.asarray(mtx).tolist(),"distortion_coef":np.asarray(dist).tolist()}

with open("calibration_matrix", "w")as f:
    yaml.dump(cal_data,f)

np.savez("calibration_parametres", mtx=mtx, dist=dist)