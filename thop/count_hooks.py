import argparse

import torch
import torch.nn as nn

multiply_adds = 1


def count_convNd(m, x, y):
    cin = m.in_channels

    kernel_ops = m.weight.size()[2:].numel()
    ops_per_element = cin * kernel_ops
    output_elements = y.nelement()

    # cout x oW x oH
    total_ops = output_elements * ops_per_element // m.groups
    m.total_ops = torch.Tensor([int(total_ops)])


def count_dconv2d(m, x, y):
    x = x[0]  # two inputs

    batch_size = x.size(0)

    height_in, width_in = x.size()[2:]
    cin = m.in_channels
    kernel_ops = m.weight.size()[2:].numel()

    # Add offset ops
    # Offset: [B, 18, H, W]
    add_offset_ops = 2 * kernel_ops * cin * batch_size * height_in * width_in

    ops_per_element = cin * kernel_ops
    output_elements = y.nelement()
    conv_ops = output_elements * ops_per_element // m.groups

    m.total_ops = torch.Tensor([int(add_offset_ops + conv_ops)])


def count_mdconv2d(m, x, y):
    x = x[0]  # three inputs

    batch_size = x.size(0)

    height_in, width_in = x.size()[2:]
    cin = m.in_channels
    kernel_ops = m.weight.size()[2:].numel()

    # Add offset ops
    # Offset: [B, 18, H, W]
    add_offset_ops = 2 * kernel_ops * cin * batch_size * height_in * width_in

    # Modulation ops
    # Modulation: [B, 9, H, W]
    modulation_ops = kernel_ops * cin * batch_size * height_in * width_in

    ops_per_element = cin * kernel_ops
    output_elements = y.nelement()
    conv_ops = output_elements * ops_per_element // m.groups

    m.total_ops = torch.Tensor([int(add_offset_ops + conv_ops + modulation_ops)])


def count_conv2d(m, x, y):
    x = x[0]

    cin = m.in_channels
    cout = m.out_channels
    kh, kw = m.kernel_size
    batch_size = x.size()[0]

    out_h = y.size(2)
    out_w = y.size(3)

    kernel_ops = multiply_adds * kh * kw
    bias_ops = 1 if m.bias is not None else 0
    ops_per_element = kernel_ops + bias_ops

    # total ops
    # num_out_elements = y.numel()
    output_elements = batch_size * out_w * out_h * cout
    total_ops = output_elements * ops_per_element * cin // m.groups

    m.total_ops = torch.Tensor([int(total_ops)])


def count_convtranspose2d(m, x, y):
    x = x[0]

    cin = m.in_channels
    cout = m.out_channels
    kh, kw = m.kernel_size
    # batch_size = x.size()[0]

    out_h = y.size(2)
    out_w = y.size(3)

    # ops per output element
    # kernel_mul = kh * kw * cin
    # kernel_add = kh * kw * cin - 1
    kernel_ops = multiply_adds * kh * kw * cin // m.groups
    bias_ops = 1 if m.bias is not None else 0
    ops_per_element = kernel_ops + bias_ops

    # total ops
    # num_out_elements = y.numel()
    # output_elements = batch_size * out_w * out_h * cout
    # ops_per_element = m.weight.nelement()

    output_elements = y.nelement()
    total_ops = output_elements * ops_per_element

    m.total_ops = torch.Tensor([int(total_ops)])

