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
# import torch
import tritonclient.grpc as grpcclient
import os 
os.environ['PYTHONIOENCODING'] = "UTF-8"

client = grpcclient.InferenceServerClient(url="192.168.6.159:8001")

# image_datas = []
image_data = cv2.imread("test/data_skinlesion/Mun_coc.jpg")
image_data = np.expand_dims(image_data, axis=0)
print(image_data.shape)
# image_data1 = np.flip(image_data, axis=2)
# image_datas.extend(image_data)
# image_datas.extend(image_data1)
# image_datas = np.array(image_datas)
# print(image_datas.shape)
# image_data = np.array([]).astype(np.uint8)

input_tensors = [grpcclient.InferInput("imgs", image_data.shape, "UINT8")]
input_tensors[0].set_data_from_numpy(image_data)
results = client.infer(model_name="skinlesion_recognition", inputs=input_tensors)
output_data = results.as_numpy("class_top5")
print(output_data.shape)
output_data = output_data.squeeze(0)
output_data = [x.decode("utf-8") for x in output_data]
print(output_data)
print(results.as_numpy("conf_top5"))