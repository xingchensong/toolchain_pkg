#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2022-10-28] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

import os
import torch


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(30, 20, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


input_data = torch.randn(1, 30, 20, 10)
demo = Model()

torch.onnx.export(
    demo, input_data, "./demo.onnx", opset_version=11,
    input_names=['in'], output_names=['out']
)

os.makedirs("./calibration_data", exist_ok=True)
to_numpy(input_data).tofile("./calibration_data/0.bin")

os.system(
    "hb_mapper makertbin \
        --model-type \"onnx\" \
        --config \"./config_onnx2bin.yaml\""
)
