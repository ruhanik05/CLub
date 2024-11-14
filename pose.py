#Pose estimation for aruco markers
import cv2
import numpy as np

#Loading camera calibration data
camera_matrix = np.array([[587.87480421, 0, 334.6959379],[0, 585.9052345, 265.59217462],[0, 0, 1]])
dist_coeffs = np.array([-0.28296289, 0.79188295, 0.0052144, 0.00694443, -0.6273732])

#Predefined dictionary for ArUco markers
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters_create()

#Marker size in meters
marker_size = 0.05

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Detect markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        #Draw marker borders
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        #Estimate pose for each marker
        for i in range(len(ids)):
            #Estimate pose
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_size, camera_matrix, dist_coeffs)
            
            #Calculate depth(distance to marker in Z-axis)
            depth = tvec[0][0][2]
            
            #Draw the axes on the marker
            cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)  # Axis length is 3 cm here

            #Display translation vector, rotation vector, and depth
            tvec_str = f"tvec: {tvec[0][0]}"
            rvec_str = f"rvec: {rvec[0][0]}"
            depth_str = f"Depth: {depth:.2f} m"
            cv2.putText(frame, tvec_str, (10, 30 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, rvec_str, (10, 50 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, depth_str, (10, 70 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    #Display the frame
    cv2.imshow("ArUco Pose Estimation with Depth", frame)

    if cv2.waitKey(1) & 0xFF == ord('d'):
        break

cap.release()
cv2.destroyAllWindows()