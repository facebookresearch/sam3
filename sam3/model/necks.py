# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Necks are the interface between a vision backbone and the rest of the detection model"""

from copy import deepcopy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from sam3.model.data_misc import NestedTensor


class Sam3DualViTDetNeck(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        position_encoding: nn.Module,
        d_model: int,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        add_sam2_neck: bool = False,
    ):
        """
        SimpleFPN neck a la ViTDet
        (From detectron2, very lightly adapted)
        It supports a "dual neck" setting, where we have two identical necks (for SAM3 and SAM2), with different weights

        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        """
        super().__init__()
        self.trunk = trunk
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()

        self.scale_factors = scale_factors
        use_bias = True
        dim: int = self.trunk.channel_list[-1]

        for _, scale in enumerate(scale_factors):
            current = nn.Sequential()

            if scale == 4.0:
                current.add_module(
                    "dconv_2x2_0",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
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
            self.convs.append(current)

        self.sam2_convs = None
        if add_sam2_neck:
            # Assumes sam2 neck is just a clone of the original neck
            self.sam2_convs = deepcopy(self.convs)

    def forward(
        self, tensor_list: List[torch.Tensor]
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        xs = self.trunk(tensor_list)
        sam3_out, sam3_pos = [], []
        sam2_out, sam2_pos = None, None
        if self.sam2_convs is not None:
            sam2_out, sam2_pos = [], []
        x = xs[-1]  # simpleFPN
        for i in range(len(self.convs)):
            sam3_x_out = self.convs[i](x)
            sam3_pos_out = self.position_encoding(sam3_x_out).to(sam3_x_out.dtype)
            sam3_out.append(sam3_x_out)
            sam3_pos.append(sam3_pos_out)

            if self.sam2_convs is not None:
                sam2_x_out = self.sam2_convs[i](x)
                sam2_pos_out = self.position_encoding(sam2_x_out).to(sam2_x_out.dtype)
                sam2_out.append(sam2_x_out)
                sam2_pos.append(sam2_pos_out)
        return sam3_out, sam3_pos, sam2_out, sam2_pos


class Sam3TriViTDetNeck(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        position_encoding: nn.Module,
        d_model: int,
        neck_norm=None,
        scale_factors=(4.0, 2.0, 1.0),
    ):
        """
        SimpleFPN neck with three heads (sam3, interactive, propagation).
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
            self.convs.append(current)

        # Assumes the new necks are just clones of the original neck
        self.interactive_convs = deepcopy(self.convs)
        self.propagation_convs = deepcopy(self.convs)

    def forward(
        self,
        tensor_list,
        *,
        need_sam3_out: bool = True,
        need_interactive_out: bool = True,
        need_propagation_out: bool = True,
    ):
        xs = self.trunk(tensor_list)
        sam3_out = []
        interactive_out = []
        propagation_out = []

        sam3_pos = []
        interactive_pos = []
        propagation_pos = []
        x = xs[-1]  # simpleFPN
        # OSS trunk returns plain tensors; onevision trunk returns NestedTensors.
        # Use getattr to handle both in a torch.compile-friendly way.
        x_data = getattr(x, "tensors", x)
        x_mask = getattr(x, "mask", None)
        for _, (conv, interactive_conv, propagation_conv) in enumerate(
            zip(self.convs, self.interactive_convs, self.propagation_convs)
        ):
            if need_sam3_out:
                sam3_conv_out = conv(x_data)
                sam3_x_out = NestedTensor(sam3_conv_out, x_mask)
                sam3_out.append(sam3_x_out)
                sam3_pos.append(
                    self.position_encoding(sam3_conv_out).to(sam3_conv_out.dtype)
                )

            if need_interactive_out:
                interactive_conv_out_t = interactive_conv(x_data)
                interactive_conv_out = NestedTensor(interactive_conv_out_t, x_mask)
                interactive_out.append(interactive_conv_out)
                interactive_pos.append(
                    self.position_encoding(interactive_conv_out_t).to(
                        interactive_conv_out_t.dtype
                    )
                )

            if need_propagation_out:
                propagation_conv_out = propagation_conv(x_data)
                propagation_x_out = NestedTensor(propagation_conv_out, x_mask)
                propagation_out.append(propagation_x_out)
                propagation_pos.append(
                    self.position_encoding(propagation_conv_out).to(
                        propagation_conv_out.dtype
                    )
                )

        return (
            sam3_out,
            sam3_pos,
            interactive_out,
            interactive_pos,
            propagation_out,
            propagation_pos,
        )
