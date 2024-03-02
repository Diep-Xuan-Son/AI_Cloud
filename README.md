## Deploying multiple models
**Setting up the model repository**
A [model repository](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html) is Triton's way of reading your models and any associated metadata with each model (configurations, version files, etc.). These model repositories can live in a local or network attached filesystem, or in a cloud object store like AWS S3, Azure Blob Storage or Google Cloud Storage. For more details on model repository location, refer to [the documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html#model-repository-locations) Servers can use also multiple different model repositories. For simplicity, this explanation only uses a single repository stored in the [local filesystem](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html#local-file-system), in the following format:
```
# Example repository structure
<model-repository>/
  <model-name>/
    [config.pbtxt]
    [<output-labels-file> ...]
    <version>/
      <model-definition-file>
    <version>/
      <model-definition-file>
    ...
  <model-name>/
    [config.pbtxt]
    [<output-labels-file> ...]
    <version>/
      <model-definition-file>
    <version>/
      <model-definition-file>
    ...
  ...
```
There are three important components to be discussed from the above structure:

- `model-name`: The identifying name for the model.
- `config.pbtxt`: For each model, users can define a model configuration. This configuration, at minimum, needs to define: the backend, name, shape, and datatype of model inputs and outputs. For most of the popular backends, this configuration file is autogenerated with defaults. The full specification of the configuration file can be found in the [model_config protobuf definition](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto).
- `version`: versioning makes multiple versions of the same model available for use depending on the policy selected. [More Information about versioning](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html#model-versions).

An example for repository
```
# Expected folder layout
model_repository/
├── text_detection
│   ├── 1
│   │   └── model.onnx
│   └── config.pbtxt
└── text_recognition
    ├── 1
    │   └── model.onnx
    └── config.pbtxt
```
An example for model configuration files
```
name: "text_detection"
backend: "onnxruntime"
max_batch_size : 256
input [
  {
    name: "input_images:0"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, 3 ]
  }
]
output [
  {
    name: "feature_fusion/Conv_7/Sigmoid:0"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, 1 ]
  }
]
output [
  {
    name: "feature_fusion/concat_3:0"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, 5 ]
  }
]
```
- `name`: "name" is an optional field, the value of which should match the name of the directory of the model.
- `backend`: This field indicates which backend is being used to run the model. Triton supports a wide variety of backends like TensorFlow, PyTorch, Python, ONNX and more. For a complete list of field selection refer to [these comments](https://github.com/triton-inference-server/backend#backends).
- `max_batch_size`: As the name implies, this field defines the maximum batch size that the model can support.
- `input and output`: The input and output sections specify the name, shape, datatype, and more, while providing operations like [reshaping](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#reshape) and support for [ragged batches](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/ragged_batching.md#ragged-batching).
- 
In [most cases](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#auto-generated-model-configuration), it's possible to leave out the input and output sections and let Triton extract that information from the model files directly. Here, we've included them for clarity and because we'll need to know the names of our output tensors in the client application later on.

For details of all supported fields and their values, refer to the [model config protobuf definition file](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto).

## Installation triton server
**Run Triton server with docker**
```bash
docker run --gpus=all -it --shm-size=512m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/my_repository:/models -v ./requirements.txt:/opt/tritonserver/requirements.txt nvcr.io/nvidia/tritonserver:22.08-py3
```
**Start serving model**
```
tritonserver --model-repository=/models --model-control-mode=explicit --load-model=text_recognition
```
## 
