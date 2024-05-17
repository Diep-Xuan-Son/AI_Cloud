from skinlesion import YOLO as skYOLO
import os
import numpy as np
import json
import cv2 

from ultralytics import YOLO 
from pathlib import Path
import triton_python_backend_utils as pb_utils

# CLASS_SKINLESION = {0: "Mụn trứng cá đỏ",
#                     1: "Ung thư biểu mô tế bào đáy dày sừng quang hóa và các thương tổn ác tính khác",
#                     2: "Viêm da cơ địa",
#                     3: "Bệnh da bọng nước",
#                     4: "Bệnh chốc và các loại nhiễm khuẩn khác",
#                     5: "Bệnh chàm",
#                     6: "Phát ban phản ứng do thuốc",
#                     7: "Rụng tóc từng mảng và các bệnh về tóc khác",
#                     8: "Nhiễm virus HPV và các bệnh lây truyền qua đường tình dục khác",
#                     9: "Bệnh rối loạn sắc tố",
#                     10: "Bệnh Lupus ban đỏ và các bệnh mô liên kết khác",
#                     11: "Ung thư da hắc tố",
#                     12: "Nấm móng và các bệnh về móng khác",
#                     13: "Bệnh viêm da do cây thường xuân và các bệnh viêm da do tiếp xúc khác",
#                     14: "Bệnh vảy nến Lichen planus",
#                     15: "Bệnh ghẻ Lyme và các bệnh nhiễm trùng do vết cắn khác",
#                     16: "Bệnh dày sừng tiết bã và các khối u lành tính khác",
#                     17: "Bệnh hệ thống",
#                     18: "Bệnh nấm Candida và các bênh nhiễm nấm khác",
#                     19: "Bệnh nổi mề đay",
#                     20: "Bệnh U máu gan",
#                     21: "Bệnh viêm mạch máu",
#                     22: "Mụn cóc"}

CLASS_SKINLESION = {0: "Acne Rosacea",
                    1: "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
                    2: "Atopic Dermatitis",
                    3: "Bullous Disease",
                    4: "Cellulitis Impetigo and other Bacterial Infections",
                    5: "Eczema",
                    6: "Drug reaction rash",
                    7: "Hair Loss and other Hair Diseases",
                    8: "HPV infection and other sexually transmitted diseases",
                    9: "Disorders of Pigmentation",
                    10: "Lupus erythematosus and other connective tissue diseases",
                    11: "Melanoma Skin Cancer",
                    12: "Nail Fungus and other Nail Disease",
                    13: "Poison Ivy and other Contact Dermatitis",
                    14: "Psoriasis Lichen Planus",
                    15: "Scabies Lyme Disease and other Bites Infestations ",
                    16: "Seborrheic keratosis and other Benign Tumors",
                    17: "Systemic Disease",
                    18: "Candidiasis and other Fungal Infections",
                    19: "Urticaria Hives",
                    20: "Vascular Tumors",
                    21: "vasculitis",
                    22: "warts Molluscum"}

class TritonPythonModel:
    def initialize(self, args):
        self.model = skYOLO("/models/skinlesion_recognition/1/best_skin_lesion_ep67.pt")

        model_config = json.loads(args["model_config"])
        class_config = pb_utils.get_output_config_by_name(
            model_config, "class_top5"
        )

        self.class_dtype = pb_utils.triton_string_to_numpy(
            class_config["data_type"]
        )
        
        conf_config = pb_utils.get_output_config_by_name(
            model_config, "conf_top5"
        )

        self.conf_dtype = pb_utils.triton_string_to_numpy(
            conf_config["data_type"]
        )

    def execute(self, requests):
        class_dtype = self.class_dtype
        conf_dtype = self.conf_dtype
        responses = []
        for request in requests:
            in_img = pb_utils.get_input_tensor_by_name(
                request, "imgs"
            )

            out_cls = []
            out_conf = []
            for image in in_img.as_numpy():
                result = self.model(image, verbose=True)[0]
                top5cls = np.array(result.probs.top5)
                top5conf = result.probs.top5conf.detach().cpu().numpy()
                print(top5cls)

                top5cls = [CLASS_SKINLESION[k] for k in top5cls]
                out_cls.append(top5cls)
                out_conf.append(top5conf)

            out_cls = np.array(out_cls)
            out_conf = np.array(out_conf)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "class_top5", out_cls.astype(class_dtype)
                    ),
                    pb_utils.Tensor(
                        "conf_top5", out_conf.astype(conf_dtype)
                    )
                ]
            )
            responses.append(inference_response)
        return responses