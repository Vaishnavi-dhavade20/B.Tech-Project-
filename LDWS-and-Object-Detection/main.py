import cv2, time, os
import numpy as np
import logging
import tkinter as tk
import pycuda.driver as drv
from datetime import datetime
from VideoPath import DrivingAssistanceApp

from TaskConditions import TaskConditions, Logger
from ObjectDetection.yoloDetector import YoloDetector
from ObjectDetection.utils import ObjectModelType,  CollisionType
from ObjectDetection.distanceMeasure import SingleCamDistanceMeasure

from LaneDetection.ultrafastLaneDetector.ultrafastLaneDetector import UltrafastLaneDetector
from LaneDetection.ultrafastLaneDetector.ultrafastLaneDetectorV2 import UltrafastLaneDetectorV2
from LaneDetection.ultrafastLaneDetector.perspectiveTransformation import PerspectiveTransformation
from LaneDetection.ultrafastLaneDetector.utils import LaneModelType, OffsetType, CurvatureType
LOGGER = Logger(None, logging.INFO, logging.INFO )

lane_config = {
	"model_path": "./LaneDetection/models/culane_res34.trt",
	"model_type" : LaneModelType.UFLDV2_CULANE
}
object_config = {
	"model_path": './ObjectDetection/models/Detection.trt',
	"model_type" : ObjectModelType.YOLOV8,
	"classes_path" : './ObjectDetection/models/coco_label.txt',
	"box_score" : 0.4,
	"box_nms_iou" : 0.45
}

# Priority : FCWS > LDWS > LKAS
class ControlPanel(object):
	CollisionDict = {
						CollisionType.UNKNOWN : (0, 255, 255),
						CollisionType.NORMAL : (0, 255, 0),
						CollisionType.PROMPT : (0, 102, 255),
						CollisionType.WARNING : (0, 0, 255)
	 				}

	OffsetDict = { 
					OffsetType.UNKNOWN : (0, 255, 255), 
					OffsetType.RIGHT :  (0, 0, 255), 
					OffsetType.LEFT : (0, 0, 255), 
					OffsetType.CENTER : (0, 255, 0)
				 }

	CurvatureDict = { 
						CurvatureType.UNKNOWN : (0, 255, 255),
						CurvatureType.STRAIGHT : (0, 255, 0),
						CurvatureType.EASY_LEFT : (0, 102, 255),
						CurvatureType.EASY_RIGHT : (0, 102, 255),
						CurvatureType.HARD_LEFT : (0, 0, 255),
						CurvatureType.HARD_RIGHT : (0, 0, 255)
					}

	def __init__(self):
		self.fps = 0
		self.frame_count = 0
		self.start = time.time()
		self.curve_status = None

		self.asset_scale = 360 / 720

		self.collision_warning_img = cv2.imread('./assets/FCWS-warning.png', cv2.IMREAD_UNCHANGED)
		self.collision_prompt_img = cv2.imread('./assets/FCWS-prompt.png', cv2.IMREAD_UNCHANGED)
		self.collision_normal_img = cv2.imread('./assets/FCWS-normal.png', cv2.IMREAD_UNCHANGED)
		self.left_curve_img = cv2.imread('./assets/left_turn.png', cv2.IMREAD_UNCHANGED)
		self.right_curve_img = cv2.imread('./assets/right_turn.png', cv2.IMREAD_UNCHANGED)
		self.keep_straight_img = cv2.imread('./assets/straight.png', cv2.IMREAD_UNCHANGED)
		self.determined_img = cv2.imread('./assets/warn.png', cv2.IMREAD_UNCHANGED)
		self.left_lanes_img = cv2.imread('./assets/LTA-left_lanes.png', cv2.IMREAD_UNCHANGED)
		self.right_lanes_img = cv2.imread('./assets/LTA-right_lanes.png', cv2.IMREAD_UNCHANGED)

		self.resize_assets()
	
	def resize_assets(self):
		self.collision_warning_img = cv2.resize(self.collision_warning_img, (int(150 * (720/1080)), int(150 * (720/1080))))
		self.collision_prompt_img = cv2.resize(self.collision_prompt_img, (int(150 * (720/1080)), int(150 * (720/1080))))
		self.collision_normal_img = cv2.resize(self.collision_normal_img, (int(150 * (720/1080)), int(150 * (720/1080))))
		self.left_curve_img = cv2.resize(self.left_curve_img, (int(200 * self.asset_scale), int(200 * self.asset_scale)))
		self.right_curve_img = cv2.resize(self.right_curve_img, (int(200 * self.asset_scale), int(200 * self.asset_scale)))
		self.keep_straight_img = cv2.resize(self.keep_straight_img, (int(200 * self.asset_scale), int(200 * self.asset_scale)))
		self.determined_img = cv2.resize(self.determined_img, (int(200 * self.asset_scale), int(200 * self.asset_scale)))
		self.left_lanes_img = cv2.resize(self.left_lanes_img, (int(300 * self.asset_scale), int(200 * self.asset_scale)))
		self.right_lanes_img = cv2.resize(self.right_lanes_img, (int(300 * self.asset_scale), int(200 * self.asset_scale)))

	def _updateFPS(self) :
		"""
		Update FPS.

		Args:
			None

		Returns:
			None
		"""
		self.frame_count += 1
		if self.frame_count >= 30:
			self.end = time.time()
			self.fps = self.frame_count / (self.end - self.start)
			self.frame_count = 0
			self.start = time.time()
	
	def DisplaySignsPanel(self, main_show, offset_type, curvature_type) :
		"""
		Display Signs Panel on image.

		Args:
			main_show: image.
			offset_type: offset status by OffsetType. (UNKNOWN/CENTER/RIGHT/LEFT)
			curvature_type: curature status by CurvatureType. (UNKNOWN/STRAIGHT/HARD_LEFT/EASY_LEFT/HARD_RIGHT/EASY_RIGHT)

		Returns:
			main_show: Draw sings info on frame.
		"""

		W = 220
		H = 200
		widget = np.copy(main_show[:H, :W])
		widget //= 2
		widget[0:2,:] = [0, 0, 255]  # top
		widget[-2:-1,:] = [0, 0, 255] # bottom
		widget[:,0:2] = [0, 0, 255]  #left
		widget[:,-2:-1] = [0, 0, 255] # right
		main_show[:H, :W] = widget

		if curvature_type == CurvatureType.UNKNOWN and offset_type in { OffsetType.UNKNOWN, OffsetType.CENTER } :
			y, x = self.determined_img[:,:,3].nonzero()
			main_show[y+5, x-50+W//2] = self.determined_img[y, x, :3]
			self.curve_status = None

		elif (curvature_type == CurvatureType.HARD_LEFT or self.curve_status== "Left") and \
			(curvature_type not in { CurvatureType.EASY_RIGHT, CurvatureType.HARD_RIGHT }) :
			y, x = self.left_curve_img[:,:,3].nonzero()
			main_show[y+5, x-50+W//2] = self.left_curve_img[y, x, :3]
			self.curve_status = "Left"

		elif (curvature_type == CurvatureType.HARD_RIGHT or self.curve_status== "Right") and \
			(curvature_type not in { CurvatureType.EASY_LEFT, CurvatureType.HARD_LEFT }) :
			y, x = self.right_curve_img[:,:,3].nonzero()
			main_show[y+5, x-50+W//2] = self.right_curve_img[y, x, :3]
			self.curve_status = "Right"
		
		
		if ( offset_type == OffsetType.RIGHT ) :
			y, x = self.left_lanes_img[:,:,2].nonzero()
			main_show[y+5, x-65+W//2] = self.left_lanes_img[y, x, :3]
		elif ( offset_type == OffsetType.LEFT ) :
			y, x = self.right_lanes_img[:,:,2].nonzero()
			main_show[y+5, x-65+W//2] = self.right_lanes_img[y, x, :3]
		elif curvature_type == CurvatureType.STRAIGHT or self.curve_status == "Straight" :
			y, x = self.keep_straight_img[:,:,3].nonzero()
			main_show[y+5, x-50+W//2] = self.keep_straight_img[y, x, :3]
			self.curve_status = "Straight"

		self._updateFPS()
		cv2.putText(main_show, "LDWS : " + offset_type.value, (10, 130), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=self.OffsetDict[offset_type], thickness=1)
		cv2.putText(main_show, "LKAS : " + curvature_type.value, org=(10, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=self.CurvatureDict[curvature_type], thickness=1)
		cv2.putText(main_show, "FPS  : %.2f" % self.fps, (10, widget.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
		return main_show

	def DisplayCollisionPanel(self, main_show, collision_type, obect_infer_time, lane_infer_time, show_ratio=0.15) :
		"""
		Display Collision Panel on image.

		Args:
			main_show: image.
			collision_type: collision status by CollisionType. (WARNING/PROMPT/NORMAL)
			obect_infer_time: object detection time -> float.
			lane_infer_time:  lane detection time -> float.

		Returns:
			main_show: Draw collision info on frame.
		"""

		W = int(main_show.shape[1]* show_ratio)
		H = int(main_show.shape[0]* show_ratio)

		widget = np.copy(main_show[H+20:2*H, -W-20:])
		widget //= 2
		widget[0:2,:] = [0, 0, 255]  # top
		widget[-2:-1,:] = [0, 0, 255] # bottom
		widget[:,-2:-1] = [0, 0, 255] #left
		widget[:,0:2] = [0, 0, 255]  # right
		main_show[H+20:2*H, -W-20:] = widget

		if (collision_type == CollisionType.WARNING) :
			y, x = self.collision_warning_img[:,:,3].nonzero()
			main_show[H+y+40, (x-W-4)] = self.collision_warning_img[y, x, :3]
		elif (collision_type == CollisionType.PROMPT) :
			y, x =self.collision_prompt_img[:,:,3].nonzero()
			main_show[H+y+40, (x-W-4)] = self.collision_prompt_img[y, x, :3]
		elif (collision_type == CollisionType.NORMAL) :
			y, x = self.collision_normal_img[:,:,3].nonzero()
			main_show[H+y+40, (x-W-4)] = self.collision_normal_img[y, x, :3]

		cv2.putText(main_show, "FCWS : " + collision_type.value, ( main_show.shape[1]- int(W) + 130 , 220), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=self.CollisionDict[collision_type], thickness=1)
		cv2.putText(main_show, "object-infer : %.2f s" % obect_infer_time, ( main_show.shape[1]- int(W) + 130, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
		cv2.putText(main_show, "lane-infer : %.2f s" % lane_infer_time, ( main_show.shape[1]- int(W) + 130, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
		return main_show


if __name__ == "__main__":
	#Loading Screen....
	root = tk.Tk()
	app = DrivingAssistanceApp(root)
	root.resizable(False, False)
	root.mainloop()
	# Initialize read and save video 
	cap = cv2.VideoCapture(app.video_path)
	width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
	current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
	vout = cv2.VideoWriter('./Output/' + f'{current_time}.mp4', fourcc , 30.0, (1920, 1080))
	cv2.namedWindow("ADAS Simulation", cv2.WINDOW_NORMAL)	
	
	#==========================================================
	#					Initialize Class
	#==========================================================
	LOGGER.info("[Pycuda] Cuda Version: {}".format(drv.get_version()))
	LOGGER.info("[Driver] Cuda Version: {}".format(drv.get_driver_version()))

	# lane detection model
	LOGGER.info("UfldDetector Model Type : {}".format(lane_config["model_type"].name))
	if ( "UFLDV2" in lane_config["model_type"].name) :
		UltrafastLaneDetectorV2.set_defaults(lane_config)
		laneDetector = UltrafastLaneDetectorV2(logger=LOGGER)
	else :
		UltrafastLaneDetector.set_defaults(lane_config)
		laneDetector = UltrafastLaneDetector(logger=LOGGER)
	transformView = PerspectiveTransformation( (width, height) )

	# object detection model
	LOGGER.info("YoloDetector Model Type : {}".format(object_config["model_type"].name))
	YoloDetector.set_defaults(object_config)
	ObjectDetection = YoloDetector(logger=LOGGER)
	distanceDetector = SingleCamDistanceMeasure()

	# display panel
	displayPanel = ControlPanel()
	analyzeMsg = TaskConditions()
	while cap.isOpened():

		ret, frame = cap.read() # Read frame from the video
		if ret:
			frame = cv2.resize(frame, (1920, 1080))
			frame_show = frame.copy()

			#========================== Detect Model =========================
			obect_time = time.time()
			ObjectDetection.DetectFrame(frame)
			obect_infer_time = round(time.time() - obect_time, 2)
			lane_time = time.time()
			laneDetector.DetectFrame(frame)
			lane_infer_time = round(time.time() - lane_time, 4)

			#========================= Analyze Status ========================
			distanceDetector.calcDistance(ObjectDetection.object_info)
			vehicle_distance = distanceDetector.calcCollisionPoint(laneDetector.draw_area_points)

			analyzeMsg.UpdateCollisionStatus(vehicle_distance, laneDetector.draw_area)


			if (not laneDetector.draw_area or analyzeMsg.CheckStatus()) :
				transformView.updateTransformParams(laneDetector.lanes_points[1], laneDetector.lanes_points[2], analyzeMsg.transform_status)
			birdview_show = transformView.transformToBirdView(frame_show)

			birdview_lanes_points = [transformView.transformToBirdViewPoints(lanes_point) for lanes_point in laneDetector.lanes_points]
			(vehicle_direction, vehicle_curvature) , vehicle_offset = transformView.calcCurveAndOffset(birdview_show, birdview_lanes_points[1], birdview_lanes_points[2])

			analyzeMsg.UpdateOffsetStatus(vehicle_offset)
			analyzeMsg.UpdateRouteStatus(vehicle_direction, vehicle_curvature)

			#========================== Draw Results =========================
			transformView.DrawDetectedOnBirdView(birdview_show, birdview_lanes_points, analyzeMsg.offset_msg)
			if (LOGGER.clevel == logging.DEBUG) : transformView.DrawTransformFrontalViewArea(frame_show)
			laneDetector.DrawDetectedOnFrame(frame_show, analyzeMsg.offset_msg)
			frame_show = laneDetector.DrawAreaOnFrame(frame_show, displayPanel.CollisionDict[analyzeMsg.collision_msg])
			ObjectDetection.DrawDetectedOnFrame(frame_show)
			distanceDetector.DrawDetectedOnFrame(frame_show)

			frame_show = displayPanel.DisplaySignsPanel(frame_show, analyzeMsg.offset_msg, analyzeMsg.curvature_msg)	
			frame_show = displayPanel.DisplayCollisionPanel(frame_show, analyzeMsg.collision_msg, obect_infer_time, lane_infer_time )
			cv2.imshow("ADAS Simulation", frame_show)

		else:
			break
		vout.write(frame_show)	
		if cv2.waitKey(1) == ord('q'): # Press key clear q to stop
			break

	vout.release()
	cap.release()
	cv2.destroyAllWindows()