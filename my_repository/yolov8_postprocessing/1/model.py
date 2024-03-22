# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import io
import json

import numpy as np
import torch
from utils import non_max_suppression, convert_torch2numpy_batch, scale_boxes, scale_coords, align_image
# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
	"""Your Python model must use the same class name. Every Python model
	that is created must have "TritonPythonModel" as the class name.
	"""

	def initialize(self, args):
		"""`initialize` is called only once when the model is being loaded.
		Implementing `initialize` function is optional. This function allows
		the model to initialize any state associated with this model.
		Parameters
		----------
		args : dict
		  Both keys and values are strings. The dictionary keys and values are:
		  * model_config: A JSON string containing the model configuration
		  * model_instance_kind: A string containing model instance kind
		  * model_instance_device_id: A string containing model instance device ID
		  * model_repository: Model repository path
		  * model_version: Model version
		  * model_name: Model name
		"""

		# You must parse model_config. JSON string is not parsed here
		model_config = json.loads(args["model_config"])

		# Get OUTPUT0 configuration
		# output_crop_config = pb_utils.get_output_config_by_name(
		# 	model_config, "yolov8_postprocessing_output_crop"
		# )
		output_bbox_config = pb_utils.get_output_config_by_name(
			model_config, "yolov8_postprocessing_output_bbox"
		)
		output_kpt_config = pb_utils.get_output_config_by_name(
			model_config, "yolov8_postprocessing_output_kpt"
		)
		output_numcrop_config = pb_utils.get_output_config_by_name(
			model_config, "yolov8_postprocessing_output_num_crop"
		)

		# Convert Triton types to numpy types
		# self.output_crop_dtype = pb_utils.triton_string_to_numpy(
		# 	output_crop_config["data_type"]
		# )
		self.output_bbox_dtype = pb_utils.triton_string_to_numpy(
			output_bbox_config["data_type"]
		)
		self.output_kpt_dtype = pb_utils.triton_string_to_numpy(
			output_kpt_config["data_type"]
		)
		self.output_numcrop_dtype = pb_utils.triton_string_to_numpy(
			output_numcrop_config["data_type"]
		)
		self.imagesz = (640, 640)
		self.conf = 0.7
		self.iou = 0.7
		self.agnostic_nms = False
		self.max_det = 300
		self.classes = 0
		self.kpt_shape = [4, 2]

	def postprocess(self, preds, imgszs, oriszs):
		preds = non_max_suppression(
			torch.tensor(preds),
			self.conf,
			self.iou,
			agnostic=self.agnostic_nms,
			max_det=self.max_det,
			classes=self.classes,
			nc=1,
		)

		# results_crop = []
		results_bbox = []
		results_num_crop = []
		results_kpt = []
		for i, pred in enumerate(preds):
			# orig_img = orig_imgs[i]
			imgsz = imgszs[i]
			orisz = oriszs[i]
			# orig_img = cv2.resize(orig_img, orisz[::-1], interpolation=cv2.INTER_AREA)
			pred[:, :4] = scale_boxes(imgsz, pred[:, :4], orisz).round()
			pred_kpts = pred[:, 6:].view(len(pred), *self.kpt_shape) if len(pred) else pred[:, 6:]
			pred_kpts = scale_coords(imgsz, pred_kpts, orisz)
			boxes = pred[:, :6].cpu().numpy()
			pred_kpts = pred_kpts.cpu().numpy()

			results_bbox.extend(boxes)
			results_kpt.extend(pred_kpts)
			results_num_crop.append([len(boxes)])
			# img_object = []
			# for i, bbox in enumerate(boxes):
			# 	kpt = pred_kpts[i]
			# 	crop, M_inv, M = align_image(orig_img, bbox, kpt)
			# 	crop = cv2.resize(crop, (150,50), interpolation=cv2.INTER_AREA)
			# 	img_object.append(crop)
			# results_crop.extend(img_object)
		return np.array(results_bbox), np.array(results_kpt), np.array(results_num_crop)

	def execute(self, requests):
		"""`execute` MUST be implemented in every Python model. `execute`
		function receives a list of pb_utils.InferenceRequest as the only
		argument. This function is called when an inference request is made
		for this model. Depending on the batching configuration (e.g. Dynamic
		Batching) used, `requests` may contain multiple requests. Every
		Python model, must create one pb_utils.InferenceResponse for every
		pb_utils.InferenceRequest in `requests`. If there is an error, you can
		set the error argument when creating a pb_utils.InferenceResponse
		Parameters
		----------
		requests : list
		  A list of pb_utils.InferenceRequest
		Returns
		-------
		list
		  A list of pb_utils.InferenceResponse. The length of this list must
		  be the same as `requests`
		"""

		# output_crop_dtype = self.output_crop_dtype
		output_bbox_dtype = self.output_bbox_dtype
		output_kpt_dtype = self.output_kpt_dtype
		output_numcrop_dtype = self.output_numcrop_dtype

		responses = []

		# Every Python backend must iterate over everyone of the requests
		# and create a pb_utils.InferenceResponse for each of them.
		for request in requests:
			# Get INPUT0
			# in_image = pb_utils.get_input_tensor_by_name(
			# 	request, "yolov8_postprocessing_input_image"
			# )
			in_0 = pb_utils.get_input_tensor_by_name(
				request, "yolov8_postprocessing_input"
			)
			in_orisize = pb_utils.get_input_tensor_by_name(
				request, "yolov8_postprocessing_input_orisize"
			)
			in_resize = pb_utils.get_input_tensor_by_name(
				request, "yolov8_postprocessing_input_resize"
			)

			
			in_0 = in_0.as_numpy()
			in_orisize = in_orisize.as_numpy()
			in_resize = in_resize.as_numpy()
			# in_image = in_image.as_numpy()

			bboxs, kpts, num_crop = self.postprocess(in_0, in_resize, in_orisize)

			# out_tensor_crops = pb_utils.Tensor(
			# 	"yolov8_postprocessing_output_crop", crops.astype(output_dtype)
			# )
			out_tensor_bboxs = pb_utils.Tensor(
				"yolov8_postprocessing_output_bbox", bboxs.astype(output_bbox_dtype)
			)
			out_tensor_kpts = pb_utils.Tensor(
				"yolov8_postprocessing_output_kpt", kpts.astype(output_kpt_dtype)
			)
			out_tensor_num_crop = pb_utils.Tensor(
				"yolov8_postprocessing_output_num_crop", num_crop.astype(output_numcrop_dtype)
			)

			# Create InferenceResponse. You can set an error here in case
			# there was a problem with handling this inference request.
			# Below is an example of how you can set errors in inference
			# response:
			#
			# pb_utils.InferenceResponse(
			#    output_tensors=..., TritonError("An error occurred"))
			inference_response = pb_utils.InferenceResponse(
				output_tensors=[out_tensor_bboxs, out_tensor_kpts, out_tensor_num_crop]
			)
			responses.append(inference_response)
		# You should return a list of pb_utils.InferenceResponse. Length
		# of this list must match the length of `requests` list.
		return responses

	def finalize(self):
		"""`finalize` is called only once when the model is being unloaded.
		Implementing `finalize` function is OPTIONAL. This function allows
		the model to perform any necessary clean ups before exit.
		"""
		print("Cleaning up...")
