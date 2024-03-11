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
import torchvision.transforms as transforms
import cv2
# from sklearn import preprocessing
from face_preprocess import preprocess as face_preprocess
# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
from PIL import Image


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
		output_config = pb_utils.get_output_config_by_name(
			model_config, "recognize_face_preprocessing_output"
		)

		# Convert Triton types to numpy types
		self.output_dtype = pb_utils.triton_string_to_numpy(
			output_config["data_type"]
		)
		self.imgsz = [112,112]
		self.use_detection = False

	def preprocess(self, img, bboxes, landms):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# biggestBox = None
		maxArea = 0
		for j, bbox in enumerate(bboxes):
			x1, y1, x2, y2 = bbox
			area = (x2-x1) * (y2-y1)
			if area > maxArea:
			# if area > maxArea:
				maxArea = area
				biggestBox = bbox
				landmarks = landms[j]
		# if biggestBox is not None:
		bbox = np.array(biggestBox)
		landmarks = np.array([landmarks[0], landmarks[2], landmarks[4], landmarks[6], landmarks[8],
					landmarks[1], landmarks[3], landmarks[5], landmarks[7], landmarks[9]])
		landmarks = landmarks.reshape((2,5)).T

		nimg = face_preprocess(img, bbox, landmarks, image_size=self.imgsz)
		# cv2.imwrite("aaaaa.jpg", nimg)
		nimg = cv2.resize(nimg, (112,112), interpolation=cv2.INTER_AREA)
		nimg_transformed = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
		nimg_transformed = np.transpose(nimg, (2,0,1))

		# input_blob = np.expand_dims(nimg_transformed, axis=0)
		return nimg_transformed

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

		output_dtype = self.output_dtype

		responses = []

		# Every Python backend must iterate over everyone of the requests
		# and create a pb_utils.InferenceResponse for each of them.
		for request in requests:
			# Get INPUT0
			in_img = pb_utils.get_input_tensor_by_name(
				request, "recognize_face_preprocessing_input_image"
			)
			in_det = pb_utils.get_input_tensor_by_name(
				request, "recognize_face_preprocessing_input_dets"
			)
			in_landm = pb_utils.get_input_tensor_by_name(
				request, "recognize_face_preprocessing_input_landms"
			)
			input_transs = []
			for i, image in enumerate(in_img.as_numpy()):
				bboxes = in_det.as_numpy()
				bboxes = bboxes[bboxes[:,4]==i][:,:4]
				landms = in_landm.as_numpy()
				landms = landms[landms[:,10]==i][:,:10]
				input_trans = self.preprocess(image, bboxes, landms)
				input_transs.append(np.array(input_trans))

			# image = in_img.as_numpy()[0]
			# bboxes = in_det.as_numpy()
			# landms = in_landm.as_numpy()

			# input_trans = self.preprocess(image, bboxes, landms)
			input_transs = np.array(input_transs)

			out_tensor = pb_utils.Tensor(
				"recognize_face_preprocessing_output", input_transs.astype(output_dtype)
			)

			# Create InferenceResponse. You can set an error here in case
			# there was a problem with handling this inference request.
			# Below is an example of how you can set errors in inference
			# response:
			#
			# pb_utils.InferenceResponse(
			#    output_tensors=..., TritonError("An error occurred"))
			inference_response = pb_utils.InferenceResponse(
				output_tensors=[out_tensor]
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
