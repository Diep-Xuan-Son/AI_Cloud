import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys
import json
# import os
# os.environ['PYTHONIOENCODING'] = "UTF-8"
# print(os.environ.get("PYTHONIOENCODING"))

# a = np.array([[b'x\xc3\xa1c \xc4\x91\xe1\xbb\x8bnh \xc4\x91\xe1\xbb\x91i t\xc6\xb0\xe1\xbb\xa3ng trong b\xe1\xbb\xa9c \xe1\xba\xa3nh']])
# print(np.char.decode(a, encoding= "utf-8"))

class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer_vi2en = AutoTokenizer.from_pretrained("/models/vinai_translate_vi2en/1/model_vi2en", src_lang="vi_VN")
        self.model_vi2en =  AutoModelForSeq2SeqLM.from_pretrained("/models/vinai_translate_vi2en/1/model_vi2en")

        model_config = json.loads(args["model_config"])
        output_config = pb_utils.get_output_config_by_name(
            model_config, "en_texts"
        )
        print(output_config)

        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_config["data_type"]
        )
        print(self.output_dtype)

    def execute(self, requests):
        output_dtype = self.output_dtype
        responses = []
        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "texts").as_numpy()
            # print(inp)
            # print(type(inp))
            # print(np.char.decode(inp.astype(bytes), encoding='utf-8'))
            texts = np.char.decode(inp.astype(bytes), encoding='utf-8')
            out_text = []
            for vi_texts in texts:
                vi_texts = vi_texts.tolist()
                # vi_texts.append("kia là ai")
                # print(vi_texts)

                input_ids = self.tokenizer_vi2en(vi_texts, padding=True, return_tensors="pt")
                # print(input_ids)
                output_ids = self.model_vi2en.generate(
                    **input_ids,
                    decoder_start_token_id=self.tokenizer_vi2en.lang_code_to_id["en_XX"],
                    num_return_sequences=1,
                    num_beams=5,
                    early_stopping=True
                )
                en_texts = self.tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
                # print(en_texts)
                out_text.append(en_texts)

            out_text = np.array(out_text)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "en_texts", out_text.astype(output_dtype)
                    )
                ]
            )
            responses.append(inference_response)
        return responses