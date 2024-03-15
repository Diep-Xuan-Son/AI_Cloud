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

from fastapi import FastAPI, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse

from schemes import *
from triton_services import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
IMG_AVATAR = "static/avatar"
PATH_IMG_AVATAR = f"{str(ROOT)}/{IMG_AVATAR}"

tritonClient = get_triton_client()
redisClient = redis.StrictRedis(host='192.168.6.86',
								port=6400,
								db=0)
app = FastAPI()

@app.post("/api/registerFace")
async def registerFace(params: Person = Depends()):
	code = params.code
	print(code)
	if redisClient.hexists("Facedb", code):
		return {"success": False, "error": "This user has been registered!"}

	path_code = os.path.join(PATH_IMG_AVATAR, code)
	if os.path.exists(path_code):
		shutil.rmtree(path_code)
	os.mkdir(path_code)

	name = params.name
	birthday = params.birthday
	imgs = []
	for i, image in enumerate(params.images):
		image_byte = await image.read()
		nparr = np.fromstring(image_byte, np.uint8)
		img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
		cv2.imwrite(f'{path_code}/face_{i+1}.jpg', img)
		imgs.append(img)
	imgs = np.array(imgs)
	#---------------------------face det-------------------------
	in_retinaface, out_retinaface = get_io_retinaface(imgs)
	results = await tritonClient.infer(model_name="detection_retinaface_ensemble", inputs=in_retinaface, outputs=out_retinaface)
	croped_image = results.as_numpy("croped_image")
	if len(croped_image)==0:
		return {"success": False, "error": "Don't find any face"}
	print(croped_image.shape)
	# cv2.imwrite("sadas.jpg", croped_image[0])
	#////////////////////////////////////////////////////////////

	#---------------------------face reg-------------------------
	in_ghostface, out_ghostface = get_io_ghostface(croped_image)
	results = await tritonClient.infer(model_name="ghost_face_nodet_ensemble", inputs=in_ghostface, outputs=out_ghostface)
	feature = results.as_numpy("feature_norm")
	feature = feature.astype(np.float16)
	print(feature.shape)

	dt_person_information = {code: f"{name},./{birthday}"}
	dt_person_feature = {code: feature.tobytes()}

	return {"success": True}

if __name__=="__main__":
	host = "0.0.0.0"
	port = 8421

	uvicorn.run("controller:app", host=host, port=port, log_level="info", reload=True)