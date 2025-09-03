# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved

"""Necks are the interface between a vision backbone and the rest of the detection model"""

from copy import deepcopy

import torch.nn as nn

from .model_misc import NestedTensor


class OriginalViTDetNeck(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        position_encoding: nn.Module,
        d_model: int,
        neck_norm=None,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
    ):
        """
        SimpleFPN neck a la ViTDet
        (From detectron2, very lightly adapted)

        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        :param neck_norm: the normalization to use
        """
        super().__init__()
        self.trunk = trunk
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()

        self.scale_factors = scale_factors
        use_bias = neck_norm is None
        dim = self.trunk.channel_list[-1]

        for _, scale in enumerate(scale_factors):
            current = nn.Sequential()

            if scale == 4.0:
                current.add_module(
                    "dconv_2x2_0",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                if neck_norm is not None:
                    current.add_module(
                        "norm",
                        norm_type_to_cls(neck_norm)(dim // 2),
                    )
                current.add_module(
                    "gelu",
                    nn.GELU(),
                )
                current.add_module(
                    "dconv_2x2_1",
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                )
                out_dim = dim // 4
            elif scale == 2.0:
                current.add_module(
                    "dconv_2x2",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                out_dim = dim // 2
            elif scale == 1.0:
                out_dim = dim
            elif scale == 0.5:
                current.add_module(
                    "maxpool_2x2",
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                out_dim = dim
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            current.add_module(
                "conv_1x1",
                nn.Conv2d(
                    in_channels=out_dim,
                    out_channels=d_model,
                    kernel_size=1,
                    bias=use_bias,
                ),
            )
            if neck_norm is not None:
                current.add_module("norm_0", norm_type_to_cls(neck_norm)(d_model))
            current.add_module(
                "conv_3x3",
                nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                ),
            )
            if neck_norm is not None:
                current.add_module("norm_1", norm_type_to_cls(neck_norm)(d_model))
            self.convs.append(current)

    def forward(self, tensor_list: NestedTensor):
        xs = self.trunk(tensor_list)
        out = []
        pos = []
        x = xs[-1]  # simpleFPN
        for _, conv in enumerate(self.convs):
            x_out = NestedTensor(conv(x.tensors), x.mask)
            out.append(x_out)
            pos.append(self.position_encoding(x_out).to(x_out.tensors.dtype))
        return out, pos


class Sam3DualViTDetNeck(OriginalViTDetNeck):
    def __init__(
        self,
        trunk: nn.Module,
        position_encoding: nn.Module,
        d_model: int,
        neck_norm=None,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
    ):
        """
        SimpleFPN neck a la ViTDet
        (From detectron2, very lightly adapted)

        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        :param neck_norm: the normalization to use
        """
        super().__init__(
            trunk=trunk,
            position_encoding=position_encoding,
            d_model=d_model,
            neck_norm=neck_norm,
            scale_factors=scale_factors,
        )
        # Assumes sam2 neck is just a clone of the original neck
        self.sam2_convs = deepcopy(self.convs)

    def forward(self, tensor_list: NestedTensor):
        xs = self.trunk(tensor_list)
        sam3_out = []
        sam2_out = []
        sam3_pos = []
        sam2_pos = []
        x = xs[-1]  # simpleFPN
        for _, (conv, sam2_conv) in enumerate(zip(self.convs, self.sam2_convs)):
            sam3_x_out = NestedTensor(conv(x.tensors), x.mask)
            sam2_x_out = NestedTensor(sam2_conv(x.tensors), x.mask)
            sam3_out.append(sam3_x_out)
            sam2_out.append(sam2_x_out)

            sam3_pos.append(
                self.position_encoding(sam3_x_out).to(sam3_x_out.tensors.dtype)
            )
            sam2_pos.append(
                self.position_encoding(sam2_x_out).to(sam2_x_out.tensors.dtype)
            )
        return sam3_out, sam3_pos, sam2_out, sam2_pos
