import os, sys
from pathlib import Path
import json 
import numpy as np
import cv2
from io import BytesIO
import shutil
import threading
import uvicorn
import redis
import time

from fastapi import FastAPI, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from schemes import *
from triton_services import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
IMG_AVATAR = "static/avatar"
PATH_IMG_AVATAR = f"{str(ROOT)}/{IMG_AVATAR}"

tritonClient = get_triton_client()
redisClient = redis.StrictRedis(host='192.168.6.86',
								port=6400,
								password="RedisAuth",
								db=0)
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/api/registerFace")
async def registerFace(params: Person = Depends()):
	code = params.code
	print(code)
	if redisClient.hexists("FaceInfor1", code):
		return {"success": False, "error": "This user has been registered!"}

	path_avatar = f"{IMG_AVATAR}/{code}/face_1.jpg"
	path_code = os.path.join(PATH_IMG_AVATAR, code)
	if os.path.exists(path_code):
		shutil.rmtree(path_code)
	os.mkdir(path_code)

	name = params.name
	birthday = params.birthday
	imgs = []
	img_infor = []
	for i, image in enumerate(params.images):
		image_byte = await image.read()
		nparr = np.fromstring(image_byte, np.uint8)
		img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
		img_infor.append(img.shape[:2])
		cv2.imwrite(f'{path_code}/face_{i+1}.jpg', img)
		img = cv2.resize(img, (640,640), interpolation=cv2.INTER_AREA)
		imgs.append(img)
	imgs = np.array(imgs)
	img_infor = np.array(img_infor)
	#---------------------------face det-------------------------
	in_retinaface, out_retinaface = get_io_retinaface(imgs)
	results = await tritonClient.infer(model_name="detection_retinaface_ensemble", inputs=in_retinaface, outputs=out_retinaface)
	croped_image = results.as_numpy("croped_image")
	if len(croped_image)==0:
		return {"success": False, "error": "Don't find any face"}
	# print(croped_image.shape)
	# cv2.imwrite("sadas.jpg", croped_image[0])
	#////////////////////////////////////////////////////////////

	#---------------------------face reg-------------------------
	in_ghostface, out_ghostface = get_io_ghostface(croped_image)
	results = await tritonClient.infer(model_name="ghost_face_nodet_ensemble", inputs=in_ghostface, outputs=out_ghostface)
	feature = results.as_numpy("feature_norm")
	feature = feature.astype(np.float16)
	print(feature.shape)

	# dt_person_information = {code: f"{code},./{name},./{birthday}"}
	# dt_person_feature = {code: feature.tobytes()}
	redisClient.hset("FaceInfor1", code, f"{code}@@@{name}@@@{birthday}@@@{path_avatar}")
	redisClient.hset("FaceFeature1", code, feature.tobytes())

	return {"success": True}

@app.post("/api/deleteUser")
def deleteUser(codes: List[str] = Query("001099008839", description="ID of persons")):
	print(codes)
	codes_noregister = []
	for code in codes:
		if not redisClient.hexists("FaceInfor1", code) or not redisClient.hexists("FaceFeature1", code):
			codes_noregister.append(code)
	if len(codes_noregister)>0:
		return {"success": False, "error": f"User {tuple(codes_noregister)} has not been registered!"}

	redisClient.hdel("FaceInfor1", *codes)
	redisClient.hdel("FaceFeature1", *codes)

	for code in codes:
		path_code = os.path.join(PATH_IMG_AVATAR, code)
		if os.path.exists(path_code):
			shutil.rmtree(path_code)

	return {"success": True}

@app.post("/api/deleteAllUser")
def deleteAllUser():
	redisClient.delete("FaceInfor1")
	redisClient.delete("FaceFeature1")
	if os.path.exists(PATH_IMG_AVATAR):
		shutil.rmtree(PATH_IMG_AVATAR)
		os.mkdir(PATH_IMG_AVATAR)
	return {"success": True}

@app.post("/api/getInformationUser")
def getInformationUser(codes: List[str] = Query(None)):
	infor_persons = {}
	print(codes)
	if codes is None:
		key_infor_persons = redisClient.hkeys("FaceInfor1")
		if len(key_infor_persons)==0:
			return {"success": True, "information": infor_persons}
		key_infor_persons = b'-;'.join(key_infor_persons).decode('utf-8').split("-;")
		val_infor_persons = redisClient.hvals("FaceInfor1")
		val_infor_persons = np.array(b'@@@'.join(val_infor_persons).decode('utf-8').split("@@@")).reshape(-1,4)	# shape (-1,3) for 3 field: code, name, birthday
		infor_persons = dict(zip(key_infor_persons, val_infor_persons.tolist()))
	else:
		for code in codes:
			print(redisClient.hexists("FaceInfor1", code))
			if not redisClient.hexists("FaceInfor1", code):
				infor_persons[code] = "No register"
				continue
			infor_person = redisClient.hget("FaceInfor1", code)
			infor_person = infor_person.decode("utf-8").split("@@@")
			infor_persons[code] = {"ID": infor_person[0], \
									"Name": infor_person[1], \
									"Birthday": infor_person[2], \
									"Avatar": infor_person[3]
									}
	return {"success": True, "information": infor_persons}

@app.post("/api/searchUser")
async def searchUser(image: UploadFile = File(...)):
	id_faces = redisClient.hkeys("FaceFeature1")
	if len(id_faces) == 0:
		return {"success": False, "error": "Don't have any registered user"}
	image_byte = await image.read()
	nparr = np.fromstring(image_byte, np.uint8)
	img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	t_det = time.time()
	#---------------------------face det-------------------------
	in_retinaface, out_retinaface = get_io_retinaface(img)
	results = await tritonClient.infer(model_name="detection_retinaface_ensemble", inputs=in_retinaface, outputs=out_retinaface)
	croped_image = results.as_numpy("croped_image")
	if len(croped_image)==0:
		return {"success": False, "error": "Don't find any face"}
	#////////////////////////////////////////////////////////////
	print("------Duration det: ", time.time()-t_det)

	t_reg = time.time()
	#---------------------------face reg-------------------------
	in_ghostface, out_ghostface = get_io_ghostface(croped_image)
	results = await tritonClient.infer(model_name="ghost_face_nodet_ensemble", inputs=in_ghostface, outputs=out_ghostface)
	feature = results.as_numpy("feature_norm")
	feature = feature.astype(np.float16)
	#////////////////////////////////////////////////////////////
	print("------Duration reg: ", time.time()-t_reg)

	t_db = time.time()
	ft_faces = np.array(redisClient.hvals("FaceFeature1"))
	feature_truth = np.frombuffer(ft_faces, dtype=np.float16).reshape(len(ft_faces), -1, 512)
	print("------Duration db: ", time.time()-t_db)

	t_comp = time.time()
	#---------------------------compare face----------------------
	# in_compareface, out_compareface = get_io_compareface(feature, feature_truth)
	# results = await tritonClient.infer(model_name="compare_face", inputs=in_compareface, outputs=out_compareface)
	# similarity_best = results.as_numpy("similarity")[0]
	# similarity_sort_idx_best = results.as_numpy("similarity_sort_idx")[0]
	# print(feature[:,:256].shape)
	# print(feature_truth[:,:,:256].shape)
	dist = np.linalg.norm(feature[:,::1] - feature_truth[:,:,::], axis=2)
	similarity = (np.tanh((1.23132175 - dist) * 6.602259425) + 1) / 2
	similarity = np.mean(similarity, axis=1)
	rand = np.random.random(similarity.size)
	similarity_sort_idx = np.lexsort((rand,similarity))[::-1]
	similarity_sort_idx_best = similarity_sort_idx[0]
	similarity_best = similarity[similarity_sort_idx_best]
	print(similarity_best)

	infor_face = None
	if similarity_best > 0.70:
		id_faces_best = id_faces[similarity_sort_idx_best]
		infor_face = redisClient.hget("FaceInfor1", id_faces_best)
	#/////////////////////////////////////////////////////////////
	print("------Duration db: ", time.time()-t_comp)
	if infor_face is None:
		return {"success": False, "error": "Don't find any user"}
	infor_face = infor_face.decode("utf-8").split("@@@")
	return {"success": True, "Information": {"code": infor_face[0], "name": infor_face[1], "birthday": infor_face[2], "avatar": infor_face[3], "similarity": float(similarity_best)}}

if __name__=="__main__":
	host = "0.0.0.0"
	port = 8421

	uvicorn.run("controller:app", host=host, port=port, log_level="info", reload=True)