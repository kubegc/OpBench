import timeit

import numpy as np
import torch
from yolort.utils import read_image_to_tensor

from yolort.models import yolov5n
import os
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity


# env
# !!! conda activate gh运行pytorch
# conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch


def getYoloPytorchData():
    from yolort.utils import get_image_from_url
    img_source = "https://huggingface.co/spaces/zhiqwang/assets/resolve/main/bus.jpg"
    img = get_image_from_url(img_source)
    img = read_image_to_tensor(img)
    return img


def loadYolo():

    model_func = yolov5n(pretrained=True,size=(640,640))
    x = torch.randn(1,3,640,640, requires_grad=True)
    torch.onnx.export(model_func, x, "/root/guohao/yolov5n.onnx")
    return model_func


def runOnPytorch(target="cpu"):
    device = torch.device(target)
    img = getYoloPytorchData()
    img = img.to(device)
    images = [img]

    img_h, img_w = 640, 640
    # Load model
    model = yolov5n(pretrained=True, score_thresh=0.45, size=(img_h, img_w))
    model = model.eval()
    model = model.to(device)
    timePytorch(model, images, target)


def timePytorch(model, inputs, target="cpu", timing_repeat=10, timing_number=10):
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for _ in range(timing_number * timing_repeat):
            model(inputs)
    print(prof.total_average())
    print(prof.key_averages())
    prof.export_chrome_trace("/root/guohao/key_averages_%s.json" % (target))
    time_result = (
            np.array(timeit.Timer(lambda: model(inputs)).repeat(repeat=timing_repeat, number=timing_number))
            * 1000
            / timing_number
    )
    statics = {
        "mean": np.mean(time_result),
        "median": np.median(time_result),
        "std": np.std(time_result)
    }
    print(statics)
    with open("/root/guohao/%s"%target,"w+") as f:
        f.write(str(prof.key_averages()))


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PATH"] = os.environ["PATH"] + ":/usr/local/cuda/bin/"

    print("***** yolov5 on pytorch cpu *****")
    runOnPytorch(target="cpu")

    print("***** yolov5 on pytorch gpu *****")
    runOnPytorch(target="cuda")
    # loadYolo()