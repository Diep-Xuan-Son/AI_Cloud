import sys
import tritonclient.grpc.aio as grpcclient

def get_triton_client():
	try:
		triton_client = grpcclient.InferenceServerClient(
			url="192.168.6.86:8001"
		)
	except Exception as e:
		print("channel creation failed: " + str(e))
		sys.exit()

	return triton_client

def get_io_retinaface(imgs):
	# Infer
	inputs = []
	outputs = []
	inputs.append(grpcclient.InferInput("input_image", imgs.shape, "UINT8"))

	# Initialize the data
	inputs[0].set_data_from_numpy(imgs)

	outputs.append(grpcclient.InferRequestedOutput("croped_image"))
	outputs.append(grpcclient.InferRequestedOutput("preprocessed_image_info"))

	return inputs, outputs

def get_io_ghostface(imgs):
	# Infer
	inputs = []
	outputs = []
	inputs.append(grpcclient.InferInput("input_image", imgs.shape, "UINT8"))

	# Initialize the data
	inputs[0].set_data_from_numpy(imgs)

	outputs.append(grpcclient.InferRequestedOutput("feature_norm"))

	return inputs, outputs