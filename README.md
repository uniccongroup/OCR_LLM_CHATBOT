# OCR_LLM_CHATBOT-
This is project is using OCR and LLM

# Installation

## installations requirement  

1. docker (optional)
2. nvidia-docer-toolkit
3. cuda
4. NVIDIA GPU (8GB VRAM)


### docker installation
```bash
$ docker run --rm --runtime=nvidia --gpus all --entrypoint /bin/bash -it nvidia/cuda:12.1.0-devel-ubuntu22.04

$ apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git

$ pip3 install tensorrt_llm==0.8.0  -U --pre --extra-index-url https://pypi.nvidia.com

$ python3 -c "import tensorrt_llm"

$ pip install -r requirements.txt

$ git lfs install
```


### local installations (ubuntu 20.04/22.04)

```bash
$ git clone https://github.com/uniccongroup/OCR_LLM_CHATBOT.git
$ pip3 install tensorrt_llm==0.8.0  -U --pre --extra-index-url https://pypi.nvidia.com
$ pip install -r examples/bloom/requirements.txt
$ git lfs install
```

### run inferencing 
```bash
$ cd backend
$ python main.py

```
