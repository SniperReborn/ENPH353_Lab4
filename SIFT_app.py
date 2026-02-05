#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys
import numpy as np

class My_App(QtWidgets.QMainWindow):

    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)
		
        self._cam_id = "/dev/video0"
        self._cam_fps = 2
        self._is_cam_enabled = False
        self._is_template_loaded = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240) 

		## @brief Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(1000 / self._cam_fps)

        self.sift = cv2.SIFT_create()

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.image_des = None
        self.image_kp = None
        self.ref_image = None
    
    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]

        ## @brief Process the reference image to extract SIFT features.
        self.ref_image = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)

        self.image_kp, self.image_des = self.sift.detectAndCompute(self.ref_image, None)

        pixmap = QtGui.QPixmap(self.template_path)
        self.template_label.setPixmap(pixmap)

        print("Loaded template image file: " + self.template_path)
    
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, 
					 bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):
        ret, frame = self._camera_device.read()

        if self.image_des is not None:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            ## @brief Process the camera frame to extract SIFT features.
            frame_kp, frame_des = self.sift.detectAndCompute(gray_frame, None)

            ## @brief So long as frame_des is found, find matching SIFT descriptions between the two images using KNN
            if frame_des is not None:
                matches = self.flann.knnMatch(self.image_des, frame_des, k=2)
                
                good_matches = []
                
                ## @brief Find matches that have a euclidean distance of less than 0.8
                for m, n in matches:
                    if m.distance < 0.8 * n.distance:
                        good_matches.append(m)

                ## @brief RANSAC algorithim! (I HAVE NO IDEA HOW THIS WORKS)
                if len(good_matches) > 8:
                    image_pts = np.float32([self.image_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    frame_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    matrix, mask = cv2.findHomography(image_pts, frame_pts, cv2.RANSAC, 5.0)
                    matchesMask = mask.ravel().tolist()

                    ## @brief Apparently called a "Perspective Transform"
                    h, w = self.ref_image.shape
                    pts = np.float32([[0,0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, matrix)

                    homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
                    frame = homography
                else:
                    matchesMask = None


        pixmap = self.convert_cv_to_pixmap(frame)
        self.live_image_label.setPixmap(pixmap)
    
    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")


if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	myApp = My_App()
	myApp.show()
	sys.exit(app.exec_())


