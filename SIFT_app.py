#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import numpy as np
import cv2
import sys

class My_App(QtWidgets.QMainWindow):

	def __init__(self):
		super(My_App, self).__init__()
		loadUi("./SIFT_app.ui", self)

		self.video_path = '/home/fizzer/SIFT_app/SIFT_Video.mp4'
		self._camera_device = cv2.VideoCapture(self.video_path)
		self._cam_fps = int(self._camera_device.get(cv2.CAP_PROP_FPS))
		self._is_cam_enabled = False
		self._is_template_loaded = False
		self.template_photo = None

		self.browse_button.clicked.connect(self.SLOT_browse_button)
		self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

		# Timer used to trigger the camera
		self._timer = QtCore.QTimer(self)
		self._timer.timeout.connect(self.SLOT_query_camera)
		self._timer.setInterval(1000 / self._cam_fps)

	def SLOT_browse_button(self):
		dlg = QtWidgets.QFileDialog()
		dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
		if dlg.exec_():
			self.template_path = dlg.selectedFiles()[0]

		pixmap = QtGui.QPixmap(self.template_path)
		self.template_label.setPixmap(pixmap)
		self.template_photo = cv2.imread(self.template_path)
		print("Loaded template image file: " + self.template_path)

		# Source: stackoverflow.com/questions/34232632/
	def convert_cv_to_pixmap(self, cv_img):
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		height, width, channel = cv_img.shape
		bytesPerLine = channel * width
		q_img = QtGui.QImage(cv_img.data, width, height, 
					 bytesPerLine, QtGui.QImage.Format_RGB888)
		return QtGui.QPixmap.fromImage(q_img)

	def SLOT_query_camera(self):
		ret, frame = self._camera_device.read()
		
		# Apply SIFT
		sift_image = self.apply_SIFT(frame, self.template_photo)

		pixmap = self.convert_cv_to_pixmap(sift_image)
		
		pixmap_resized = pixmap.scaled(
            self.live_image_label.width(),
            self.live_image_label.height(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
		self.live_image_label.setPixmap(pixmap_resized)

	def SLOT_toggle_camera(self):
		if self._is_cam_enabled:
			self._timer.stop()
			self._is_cam_enabled = False
			self.toggle_cam_button.setText("&Enable camera")
		else:
			self._timer.start()
			self._is_cam_enabled = True
			self.toggle_cam_button.setText("&Disable camera")

	def apply_SIFT(self, frame, template):
		
		# Initiate SIFT detector
		sift = cv2.SIFT_create()

		# find the template keypoints and descriptors with SIFT
		key_points_template, desc_vector_template = sift.detectAndCompute(template, None); # None = no mask

		# find video keypoints and descriptors with SIFT 
		frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		key_points_video, desc_vector_video = sift.detectAndCompute(frame_grey, None) # None = no mask

		# Set up FLANN (Fast Lib for Approx Nearest Neighbors)

		# Creates two K-V pairs, the key algorithm with value 0 tells 
		# FLANN to use the the FLANN_INDEX-KDTREE algorithm which is good for SIFT.
		# The key trees with value 5 tells FLANN how many K-dimensional trees to use.
		index_params = dict(algorithm=0, trees=5)
		# Search parameters for FLANN are empty
		search_params = dict()
		# Create FLANN matcher
		flann = cv2.FlannBasedMatcher(index_params, search_params)


		# Using FLANN and K-Nearest neighbors, compare the template image to video frame
		# k=2 means return 2 nearest matches for template in the video.
		matches = flann.knnMatch(desc_vector_template, desc_vector_video, k=2)

		# Go through matches and use Lowe's ratio test to filter out bad matches.
		good_points = []

		# k = 2 so return two matches, m is best match, n is second best match.
		# If the distance to the greatest match is significantly lower than to the second match
		for m, n in matches:
			if m.distance < 0.6 * n.distance:
				good_points.append(m) # m is a vector descriptor

		
		# Apply Homography 
		query_pts = np.float32([key_points_template[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
		train_pts = np.float32([key_points_video[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
		matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
		matches_mask = mask.ravel().tolist()

		# Perspective transform
		h, w = template.shape[:2]
		pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
		dst = cv2.perspectiveTransform(pts, matrix)
		homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)

		return homography

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	myApp = My_App()
	myApp.show()
	sys.exit(app.exec_())

