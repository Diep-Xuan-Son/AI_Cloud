import numpy as np
import cv2
import tritonclient.grpc as grpcclient
from utils import non_max_suppression, convert_torch2numpy_batch, scale_boxes, scale_coords
import torch
from utils_ocr import *

def check_plate_sqare(img_plate):
	"""
	if plate sqare : Split the plate in half then merge it into 1 line
	if not plate square: return 0
	"""
	height, width, _ = img_plate.shape
	scale = int(width / height)
	img_list = []
	# if scale < 2:
	x3, y3, x4, y4 = 0, int(height / 2), width, height
	x1, y1, x2, y2 = 0, 0, width, height - int(height / 2)
	up_plate = img_plate[y1:y2, x1:x2]
	down_plate = img_plate[y3:y4, x3:x4]
	down_plate = cv2.resize(down_plate, (up_plate.shape[1], up_plate.shape[0]))
	horizontal_plate = cv2.hconcat([up_plate, down_plate])
	img_list = [up_plate, down_plate]
	return horizontal_plate, img_list
	# else:
	# 	return None, None

def align_image(img, bbox, kpt): # type_lp = 1 -> biển dài, type_lp = 0 -> biển ngắn
	bbox = bbox.astype(int)
	kpt = kpt.astype(int)
	img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
	pnt0 = np.maximum(kpt[0],bbox[:2])
	pnt1 = np.array([np.minimum(kpt[1][0],bbox[2]), np.maximum(kpt[1][1],bbox[1])])
	pnt2 = np.minimum(kpt[2],bbox[2:4])
	pnt3 = np.array([np.maximum(kpt[3][0],bbox[0]), np.minimum(kpt[3][1],bbox[3])])
	points_norm = np.concatenate(([pnt0], [pnt1], [pnt2], [pnt3]))
	min_cor = np.min(points_norm, axis=0)
	points_trans = points_norm-min_cor

	source_points = points_trans.astype(np.float32)
	width = np.linalg.norm(source_points[0] - source_points[1]).astype(int)
	height = np.linalg.norm(source_points[0] - source_points[3]).astype(int)

	dest_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
	M = cv2.getPerspectiveTransform(source_points, dest_points)
	dst = cv2.warpPerspective(img, M, (width, height))

	if np.linalg.det(M) == 0:
		M_inv = M
	else:
		M_inv = np.linalg.inv(M)

	if width > 2.5*height:
		dst = dst
	else:
		dst,_ = check_plate_sqare(dst)
	return dst, M_inv, M

def postprocess(preds, imgszs, orig_imgs, oriszs):
	conf = 0.7
	iou = 0.7
	agnostic_nms = False
	max_det = 300
	classes = 0
	kpt_shape = [4, 2]

	"""Return detection results for a given input image or list of images."""
	preds = non_max_suppression(
		torch.tensor(preds),
		conf,
		iou,
		agnostic=agnostic_nms,
		max_det=max_det,
		classes=classes,
		nc=1,
	)

	# if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
	#     orig_imgs = convert_torch2numpy_batch(orig_imgs)

	results_crop = []
	results_bbox = []
	results_num_crop = []
	results_kpt = []
	for i, pred in enumerate(preds):
		orig_img = orig_imgs[i]
		imgsz = imgszs[i]
		orisz = oriszs[i]
		orig_img = cv2.resize(orig_img, orisz[::-1], interpolation=cv2.INTER_AREA)
		pred[:, :4] = scale_boxes(imgsz, pred[:, :4], orisz).round()
		pred_kpts = pred[:, 6:].view(len(pred), *kpt_shape) if len(pred) else pred[:, 6:]
		pred_kpts = scale_coords(imgsz, pred_kpts, orisz)
		boxes = pred[:, :6].cpu().numpy()
		print(boxes)
		print(pred_kpts)
		pred_kpts = pred_kpts.cpu().numpy()
		# cv2.rectangle(orig_img, boxes[0][:2].astype(int), boxes[0][2:4].astype(int), (255,255,255), 2)
		# cv2.circle(orig_img, pred_kpts[0][0], 10, (255,0,0), -1)
		# cv2.circle(orig_img, pred_kpts[0][1], 10, (0,255,0), -1)
		# cv2.circle(orig_img, pred_kpts[0][2], 10, (0,0,255), -1)
		# cv2.circle(orig_img, pred_kpts[0][3], 10, (255,0,255), -1)
		# cv2.imshow("sdfasdf", orig_img)
		# cv2.waitKey(0)
		results_bbox.extend(boxes)
		results_kpt.extend(pred_kpts)
		results_num_crop.append([len(boxes)])
		img_object = []
		for i, bbox in enumerate(boxes):
			kpt = pred_kpts[i]
			crop, M_inv, M = align_image(orig_img, bbox, kpt)
			crop = cv2.resize(crop, (150,50), interpolation=cv2.INTER_AREA)
			img_object.append(crop)
			cv2.imshow("sdfasdf", crop)
			cv2.waitKey(0)
		results_crop.extend(img_object)
	return np.array(results_crop), np.array(results_bbox), np.array(results_kpt), np.array(results_num_crop)

imgsz = (640,640)
client = grpcclient.InferenceServerClient(url="192.168.6.130:8001")
image_data = cv2.imread("data_test/lp1.jpg")
# print(image_data.shape[:2])
image_data0 = np.expand_dims(image_data, axis=0)
orisz = np.expand_dims(image_data.shape[:2], axis=0).astype(np.int32)
image_data = cv2.resize(image_data, imgsz, interpolation=cv2.INTER_AREA)
image_data = np.expand_dims(image_data, axis=0)
imgsz = np.expand_dims(imgsz, axis=0).astype(np.int32)

input_tensors = [grpcclient.InferInput("input_image", image_data.shape, "UINT8"), grpcclient.InferInput("input_resize", imgsz.shape, "INT32"), grpcclient.InferInput("input_orisize", orisz.shape, "INT32")]
input_tensors[0].set_data_from_numpy(image_data)
input_tensors[1].set_data_from_numpy(imgsz)
input_tensors[2].set_data_from_numpy(orisz)
results = client.infer(model_name="yolov8Pose_licencePlate_ensemble", inputs=input_tensors)
# output_data = results.as_numpy("output")
# print(output_data.shape)
# crops, bboxs, kpts, num_crop = postprocess(output_data, imgsz, image_data, orisz)
# print(crops.shape)
# print(bboxs.shape)
# print(kpts.shape)
# print(num_crop)

bboxs = results.as_numpy("output_bbox")
print(bboxs)
kpts = results.as_numpy("output_kpt")
print(kpts)
num_crop = results.as_numpy("output_num_crop")
print(num_crop)
crops = []
for orig_img in image_data0:
	for i, bbox in enumerate(bboxs):
		kpt = kpts[i]
		crop, M_inv, M = align_image(orig_img, bbox, kpt)
		# crop = cv2.resize(crop, (150,50), interpolation=cv2.INTER_AREA)
		crops.append(crop)
		cv2.imshow("sdfasdf", crop)
		cv2.waitKey(0)
# print(np.array(crops).shape)

#---------------------------------------------------------------------

# input_tensors = [grpcclient.InferInput("yolov8_postprocessing_input", output_data.shape, "FP32"),\
# 				grpcclient.InferInput("yolov8_postprocessing_input_orisize", orisz.shape, "INT32"),\
# 				grpcclient.InferInput("yolov8_postprocessing_input_resize", imgsz.shape, "INT32")]
# input_tensors[0].set_data_from_numpy(output_data)
# input_tensors[1].set_data_from_numpy(orisz)
# input_tensors[2].set_data_from_numpy(imgsz)
# results = client.infer(model_name="yolov8_postprocessing", inputs=input_tensors)
# bboxs = results.as_numpy("yolov8_postprocessing_output_bbox")
# print(bboxs)
# kpts = results.as_numpy("yolov8_postprocessing_output_kpt")
# print(kpts)
# num_crop = results.as_numpy("yolov8_postprocessing_output_num_crop")
# crops = []
# for orig_img in image_data0:
# 	for i, bbox in enumerate(bboxs):
# 		kpt = kpts[i]
# 		crop, M_inv, M = align_image(orig_img, bbox, kpt)
# 		crop = cv2.resize(crop, (150,50), interpolation=cv2.INTER_AREA)
# 		crops.append(crop)
# 		cv2.imshow("sdfasdf", crop)
# 		cv2.waitKey(0)
# print(np.array(crops).shape)
#///////////////////////////////////////////////////////////////////


#---------------------OCR------------------------------------------
# model_path= 'weights/ppOCR.onnx'
# providers = [("CUDAExecutionProvider", {"device_id": 0}), 'CPUExecutionProvider']
# predictor = ort.InferenceSession(model_path, providers=providers)
# w = predictor.get_inputs()[0].shape[3:][0]
# print(w)
# h = predictor.get_inputs()[0].shape[2:3][0]
# print(h)
# exit()
norm_img_batch = []
max_wh_ratio = 0
for crop in crops:
	h, w = crop.shape[0:2]
	wh_ratio = w * 1.0 / h
	max_wh_ratio = max(max_wh_ratio, wh_ratio)
for crop in crops:
	norm_img = resize_norm_img(crop, max_wh_ratio)
	norm_img = norm_img[np.newaxis, :]
	norm_img_batch.append(norm_img)
norm_img_batch = np.concatenate(norm_img_batch)
print(norm_img_batch.shape)

#/////////////////////////////////////////////////////////////////