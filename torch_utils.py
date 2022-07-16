from torch import nn 
import functools
import torch
import collections
import numpy as np

def is_sequence(x):
    return isinstance(x, collections.abc.Sequence) and not isinstance(x, str)


def list_if_not(x):
    return list(x) if is_sequence(x) else [x]


def ensure_len(x, size):
    x = list_if_not(x)
    assert len(x) == size or len(x) == 1
    if len(x) == 1:
        x = x * size
    return x


class Identity(nn.Module):
    def forward(self, x):
        return x


def Nonlinearity(nonlin):
    return {
        None: Identity,
        "relu": nn.ReLU,
        # TODO: figure out why inplace works for backbone, but not for everything generally
        "leaky_relu": functools.partial(nn.LeakyReLU, negative_slope=0.1, inplace=False),
    }[nonlin]()


def Normalization(input_channels, norm):
    if norm is None:
        return Identity()

    assert isinstance(norm, dict) and "method" in norm
    method = norm["method"]

    if method == "batch":
        affine = norm.get("affine", False)
        eps = norm.get("eps", 1e-5)
        momentum = norm.get("momentum", 0.01)
        return nn.BatchNorm2d(input_channels, eps=eps, momentum=momentum, affine=affine)

    elif method == "batch3d":
        affine = norm.get("affine", False)
        eps = norm.get("eps", 1e-5)
        momentum = norm.get("momentum", 0.01)
        return nn.BatchNorm3d(input_channels, eps=eps, momentum=momentum, affine=affine)

    elif method == "group":
        if "num_groups" in norm:
            num_groups = norm["num_groups"]

        elif "num_per_group" in norm:
            num_groups = input_channels // norm["num_per_group"]

        else:
            raise NotImplementedError

        if input_channels % num_groups != 0:
            print(f"Skipping group norm due to divisibility {input_channels} % {num_groups} != 0...")
            return Identity()

        affine = norm.get("affine", False)
        eps = norm.get("eps", 1e-5)
        return nn.GroupNorm(num_groups, input_channels, affine=affine, eps=eps)

    else:
        raise NotImplementedError

class Conv2d(nn.Module):
    def __init__(self, input_channels, output_channels, ksize=3, stride=1, dilation=1, nonlin="leaky_relu", norm=None):
        super().__init__()

        assert ksize % 2 == 1, "ksize must be odd"
        padding = int(np.ceil((ksize + (ksize - 1) * (dilation - 1) - stride) / 2.0))

        self.norm = Normalization(input_channels, norm)
        self.nonlin = Nonlinearity(nonlin)
        self.conv = nn.Conv2d(
            input_channels, output_channels, kernel_size=ksize, stride=stride, dilation=dilation, padding=padding
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.nonlin(x)
        x = self.conv(x)
        return x

class ConvBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        channels,
        ksize=3,
        stride=1,
        dilation=1,
        nonlin="leaky_relu",
        norm=None,
        mode=None,
        activate_final=False,
    ):
        super().__init__()
        assert mode in {None, "gated", "resnet", "gated_resnet"}, mode

        channels = list_if_not(channels)
        ksize = ensure_len(ksize, len(channels))
        dilation = ensure_len(dilation, len(channels))

        if isinstance(stride, int):
            stride = [1] * (len(channels) - 1) + [stride]
        else:
            stride = ensure_len(stride, len(channels))

        channels = [input_channels] + list(channels)
        self.dim_out = channels[-1]
        self.mode = mode

        if mode in {"gated", "gated_resnet"}:
            channels[-1] *= 2

        core = []
        for i in range(len(channels) - 1):
            core.append(
                Conv2d(
                    channels[i],
                    channels[i + 1],
                    ksize=ksize[i],
                    stride=stride[i],
                    dilation=dilation[i],
                    nonlin=nonlin,
                    norm=norm,
                )
            )
        if activate_final:
            core.append(Nonlinearity(nonlin))

        self.core = nn.Sequential(*core)

        shortcut = []
        if mode in {"resnet", "gated_resnet"}:
            if input_channels != self.dim_out:
                shortcut.append(Conv2d(input_channels, self.dim_out, ksize=1, nonlin=nonlin, norm=norm))

            if max(stride) > 1:
                shortcut.append(nn.MaxPool2d(int(np.prod(stride)), ceil_mode=True))

        self.shortcut = nn.Sequential(*shortcut)

    def forward(self, x):
        xc = self.core(x)
        if self.mode is None:
            return xc
        elif self.mode == "gated":
            xc1, xc2 = torch.split(xc, self.dim_out, dim=1)
            return xc1 * torch.sigmoid(xc2)

        elif self.mode == "resnet":
            xs = self.shortcut(x)
            return xc + xs

        elif self.mode == "gated_resnet":
            xs = self.shortcut(x)
            xc1, xc2 = torch.split(xc, self.dim_out, dim=1)
            return torch.addcmul(xs, xc1, xc2.sigmoid())

        else:
            raise NotImplementedError(self.mode)


def MLP(input_size, layer_sizes, nonlin="leaky_relu", activate_final=False):
    fc = []
    layer_sizes = [input_size] + list(layer_sizes)
    for i in range(len(layer_sizes) - 1):
        fc.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i + 2 < len(layer_sizes) or activate_final:
            fc.append(Nonlinearity(nonlin))

    return nn.Sequential(*fc)

class FlattenTrailingDimensions(nn.Module):
    def __init__(self, dims=1):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        flat_shape = tuple(x.shape[: self.dims]) + (-1,)
        return x.contiguous().view(*flat_shape)

class BufferAttend1d(nn.Module):
    def __init__(self, dim_in, key_dim, val_dim):
        super().__init__()
        self.key_dim, self.val_dim = key_dim, val_dim
        self.key_fn = nn.Linear(dim_in, key_dim)
        self.query_fn = nn.Linear(dim_in, key_dim)
        self.value_fn = nn.Linear(dim_in, val_dim)
        self.fill = -1024
        
    def forward(self, x, buffer=None, mask=None):
        if buffer is None:
            buffer = x

        query = self.query_fn(x)  # shape(..., Q, d)
        keys = self.key_fn(buffer)  # shape(..., K, d)
        vals = self.value_fn(buffer)  # shape(..., K, d)
        logits = torch.einsum("...qd, ...kd -> ...qk", query, keys) / np.sqrt(self.key_dim)  # shape(..., Q, K)

        if mask is not None:
            logits = torch.where(mask, logits, self.fill)

        probs = torch.exp(logits - logits.max(dim=-1, keepdim=True)[0])
        probs = probs / probs.sum(-1, keepdim=True)
        read = torch.einsum("...qk, ...kd -> ...qd", probs, vals)  # shape(..., Q, d)

        return read


def variance_scaling_initializer(var, fan_mode="fan_in", dist="uniform", scale=1.0):
    shape = var.shape
    fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
    fan_out = float(shape[-1])

    n = {"fan_in": fan_in, "fan_out": fan_out, "fan_avg": (fan_in + fan_out) / 2.0}[fan_mode]

    if dist == "uniform":
        limit = np.sqrt(6.0 * scale / n)
        nn.init.uniform_(var, -limit, +limit)
    elif dist == "normal":
        stddev = np.sqrt(2.6 * scale / n)
        nn.init.normal_(var, 0.0, stddev)
        with torch.no_grad():
            var.clamp_(-2.0 * stddev, 2.0 * stddev)
    else:
        raise NotImplementedError(dist)


def initialize(module, initializers=None, scale=1.0):
    if initializers is None:
        initializers = {"weight": "orthogonal", "bias": "zero"}

    for name, var in module.named_parameters():
        if not var.requires_grad:
            # print(f'Not initializing variable <{name}> because it does not require grad')
            continue

        param_type = name.split(".")[-1]
        if param_type in initializers.keys():
            mode = initializers[param_type]
        else:
            mode = "zero"

        if mode == "orthogonal":
            if len(var.shape) > 1:
                # Suppose that we do the following:
                #   >>> core = nn.Linear(d1, d2)
                #   >>> x = torch.randn(batch_size, d1)
                #   >>> y = core(x)
                # The following scheme for choosing `gain` will ensure that `y` will have the same variance as `x`.
                ratio = var.size(0) / var.size(1)
                gain = scale * np.sqrt(ratio) if ratio > 1 else scale
                nn.init.orthogonal_(var, gain=gain)
            else:
                variance_scaling_initializer(var, scale=scale)
        elif mode == "zero":
            nn.init.constant_(var, 0.0)
        else:
            fan_mode, dist = mode.split(":")
            variance_scaling_initializer(var, fan_mode, dist, scale=scale)
