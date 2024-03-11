# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import cv2
import torch
import tritonclient.grpc as grpcclient

def preProcess(img):
		im_height, im_width, _ = img.shape
		scale = [im_width, im_height, im_width, im_height]
		img = cv2.resize(img, (640,640), interpolation=cv2.INTER_AREA)
		img = np.float32(img)
		img -= (104, 117, 123)
		img = img.transpose(2, 0, 1)
		img = torch.from_numpy(img).unsqueeze(0)
		#img = img.to(self.device)
		#scale = scale.to(self.device)
		return [img, scale, im_height, im_width]

client = grpcclient.InferenceServerClient(url="192.168.6.161:8001")

image_data = cv2.imread("2.jpg")
image_data1 = cv2.imread("img1.jpeg")
image_data1 = cv2.resize(image_data1, (image_data.shape[1], image_data.shape[0]))
# input = preProcess(image_data)
# image_data = np.array(input[0])
# image_data = np.concatenate((image_data,image_data), axis=0)
# print(image_data.shape)
# input_tensors = [grpcclient.InferInput("inputs", image_data.shape, "FP32")]
# input_tensors[0].set_data_from_numpy(image_data)
# results = client.infer(model_name="detection_retinaface", inputs=input_tensors)
# cls = results.as_numpy("output_1")
# det = results.as_numpy("output_0")
# lm = results.as_numpy("output_2")
# print(cls.shape)

# image_data = np.fromfile("img1.jpeg", dtype="uint8")
image_data = np.expand_dims(image_data, axis=0)
image_data1 = np.expand_dims(image_data1, axis=0)
image_data = np.concatenate((image_data,image_data1), axis=0)

input_tensors = [grpcclient.InferInput("input_image", image_data.shape, "UINT8")]
input_tensors[0].set_data_from_numpy(image_data)
results = client.infer(model_name="detection_retinaface_ensemble", inputs=input_tensors)
output_data = results.as_numpy("confs")
print(output_data[:,:-1])
print(output_data[:,-1].astype(int))
