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

# a = np.array([[b'x\xc3\xa1c \xc4\x91\xe1\xbb\x8bnh \xc4\x91\xe1\xbb\x91i t\xc6\xb0\xe1\xbb\xa3ng trong b\xe1\xbb\xa9c \xe1\xba\xa3nh']])
# print(a)
# print(type(a[0][0]))
# print(a[0][0].decode("utf-8"))
# print(np.char.decode(a, encoding= "utf-8"))
# b = np.char.decode(a, encoding= "utf-8")
# print(type(b.tolist()[0]))
# exit()
#-----------------vi2en-------------------------
# text = np.array(["xác định đối tượng trong bức ảnh"])
# text = np.expand_dims(text, axis=0)
# text = np.char.encode(text, encoding = 'utf-8')
# print(type(text[0][0]))
# print(text)
# print(np.char.decode(text, encoding= "utf-8"))
# # exit()

# input_tensors = [grpcclient.InferInput("texts", text.shape, "BYTES")]
# input_tensors[0].set_data_from_numpy(text)
# results = client.infer(model_name="vinai_translate_vi2en", inputs=input_tensors)
# output_data = results.as_numpy("en_texts")
# print(output_data.astype(str))
#///////////////////////////////////////////////

#--------------------en2vi-----------------------
# dict_disease = {0: 'Acne and Rosacea Photos', 1: 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 2: 'Atopic Dermatitis Photos', 3: 'Bullous Disease Photos', 4: 'Cellulitis Impetigo and other Bacterial Infections', 5: 'Eczema Photos', 6: 'Exanthems and Drug Eruptions', 7: 'Hair Loss Photos Alopecia and other Hair Diseases', 8: 'Herpes HPV and other STDs Photos', 9: 'Light Diseases and Disorders of Pigmentation', 10: 'Lupus and other Connective Tissue diseases', 11: 'Melanoma Skin Cancer Nevi and Moles', 12: 'Nail Fungus and other Nail Disease', 13: 'Poison Ivy Photos and other Contact Dermatitis', 14: 'Psoriasis pictures Lichen Planus and related diseases', 15: 'Scabies Lyme Disease and other Infestations and Bites', 16: 'Seborrheic Keratoses and other Benign Tumors', 17: 'Systemic Disease', 18: 'Tinea Ringworm Candidiasis and other Fungal Infections', 19: 'Urticaria Hives', 20: 'Vascular Tumors', 21: 'Vasculitis Photos', 22: 'Warts Molluscum and other Viral Infections'}
# for disease in dict_disease.values():
	# print(disease)
	# text = np.array([f"{disease}"])
text = np.array(["This image is Viêm da cơ địa disease"])
text = np.expand_dims(text, axis=0)
text = np.char.encode(text, encoding = 'utf-8')

input_tensors = [grpcclient.InferInput("texts", text.shape, "BYTES")]
input_tensors[0].set_data_from_numpy(text)
results = client.infer(model_name="vinai_translate_en2vi", inputs=input_tensors)
output_data = results.as_numpy("vi_texts")
print(output_data[0][0].decode("utf-8"))
#////////////////////////////////////////////////