# Global Preprocessing
After we generate the top-down image, we should save this image to data/top_down.


## Install

It seems that SAM2 and Qwen-2.5-VL requires python>=3.10, while the habitat-sim needs python<10. Therefore, we must work in different environments.

Installing SAM2
```sh
conda create -n preprocessing python=3.10
cd preprocessing/sam2
pip install -e .
pip install opencv-python
```

In order to use Qwen2.5-VL:
```sh
pip install transformers==4.51.3 accelerate
pip install qwen-vl-utils[decord]
```

In order to use flash-attention (which is needed by Qwen model), my suggestion is to download file: flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl from https://github.com/Dao-AILab/flash-attention/releases. And install the wheel with 
```sh
pip install flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

To use the gemini model, you should:
```sh
export API_KEY='sk-LcoHn73XDQIN6RqS1Fwp90Ec6ZJ3IO85CBDqDMUOdpuallDq'
```

