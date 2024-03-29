import numpy as np
import math
import cv2

def mode_rec(rec_res, threshold):
	# print('rec_res[0] : ', rec_res) 
	check_acc = False
	txt_result = ''
	sum_acc = 0
	count_txt = 0
	for txt in rec_res:
		# print('------------------------------',str(txt[0])) #("廖'纳绚", 0.9587153196334839)
		acc = round(txt[1],4)
		if acc < threshold:
			check_acc = True
		txt_result += ''+str(txt[0])
		sum_acc+=acc
		count_txt+=1
	arv_acc = sum_acc/count_txt
	return txt_result, check_acc, arv_acc

class BaseRecLabelDecode(object):
	""" Convert between text-label and text-index """

	def __init__(self, character_dict_path=None, use_space_char=False, character_str="0123456789abcdefghijklmnopqrstuvwxyz"):
		self.beg_str = "sos"
		self.end_str = "eos"

		self.character_str = []
		if character_dict_path is None:
			# self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
			self.character_str = character_str
			dict_character = list(self.character_str)
		else:
			with open(character_dict_path, "rb") as fin:
				lines = fin.readlines()
				for line in lines:
					line = line.decode('utf-8').strip("\n").strip("\r\n")
					self.character_str.append(line)
			if use_space_char:
				self.character_str.append(" ")
			dict_character = list(self.character_str)

		self.character = dict_character
		# dict_character = self.add_special_char(dict_character)
		# self.dict = {}
		# for i, char in enumerate(dict_character):
		#     self.dict[char] = i
		# self.character = dict_character

	# def add_special_char(self, dict_character):
	#     return dict_character

	def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
		""" convert text-index into text-label. """
		result_list = []
		ignored_tokens = self.get_ignored_tokens()
		batch_size = len(text_index)
		for batch_idx in range(batch_size):
			char_list = []
			conf_list = []
			for idx in range(len(text_index[batch_idx])):
				if text_index[batch_idx][idx] in ignored_tokens:
					continue
				if is_remove_duplicate:
					# only for predict
					if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
							batch_idx][idx]:
						continue
				char_list.append(self.character[int(text_index[batch_idx][idx])])
				if text_prob is not None:
					conf_list.append(text_prob[batch_idx][idx])
				else:
					conf_list.append(1)
			text = ''.join(char_list)
			if len(conf_list)==0:
				conf_list = np.nan
			result_list.append((text, np.mean(conf_list)))
		return result_list

	def get_ignored_tokens(self):
		return [0]  # for ctc blank

class CTCLabelDecode(BaseRecLabelDecode):
	""" Convert between text-label and text-index """

	def __init__(self, character_dict_path=None, use_space_char=False, character_str=None,
				 **kwargs):
		super(CTCLabelDecode, self).__init__(character_dict_path,
											 use_space_char, character_str)

	def __call__(self, preds, label=None, *args, **kwargs):
		if isinstance(preds, tuple):
			preds = preds[-1]
		# if isinstance(preds, paddle.Tensor):
		#     preds = preds.numpy()
		preds_idx = preds.argmax(axis=2)
		preds_prob = preds.max(axis=2)
		text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
		if label is None:
			return text
		label = self.decode(label)
		return text, label

	def add_special_char(self, dict_character):
		dict_character = ['blank'] + dict_character
		return dict_character


def resize_norm_img(img, max_wh_ratio):
		imgC, imgH, imgW = (3,48,320)
		assert imgC == img.shape[2]
		imgW = int((imgH * max_wh_ratio))

		# w = self.predictor.get_inputs()[0].shape[3:][0]
		# print("--------------w: ", self.predictor.get_inputs()[0])
		# if not isinstance(w, (int, float)):
		#   w = int(240*imgH/80)
		# if isinstance(w, (int, float)):
		# 	if w is not None and w > 0:
		# 		imgW = w

		# h = self.predictor.get_inputs()[0].shape[2:3][0]        
		# if isinstance(h, (int, float)):
		# 	if h is not None and h > 0:
		# 		imgH = h

		h, w = img.shape[:2]
		ratio = w / float(h)
		if math.ceil(imgH * ratio) > imgW:
			resized_w = imgW
		else:
			resized_w = int(math.ceil(imgH * ratio))
		# print("-----------size: ", resized_w, imgH)
		resized_image = cv2.resize(img, (resized_w, imgH))
		resized_image = resized_image.astype('float32')
		resized_image = resized_image.transpose((2, 0, 1)) / 255
		resized_image -= 0.5
		resized_image /= 0.5
		padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
		padding_im[:, :, 0:resized_w] = resized_image
		return padding_im