import numpy as np
import torch
import warnings
from torch.nn.init import calculate_gain


class SinAct(torch.nn.Module):
    def forward(self, x):
        return x.sin()


class SincAct(torch.nn.Module):
    def forward(self, x):
        return x.sinc()


class CustomLinear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        bound_weight = kwargs.pop("bound_weight", None)
        bound_bias = kwargs.pop("bound_bias", None)
        super().__init__(*args, **kwargs)
        with torch.no_grad():
            if bound_weight is not None:
                self.weight.uniform_(-bound_weight, bound_weight)
            if bound_bias is not None:
                self.weight.uniform_(-bound_bias, bound_bias)


class PositionalEncoding(torch.nn.Module):
    def __init__(
        self,
        num_freqs,
        concat_input=True,
        dtype=torch.float32,
    ):
        super().__init__()
        if type(num_freqs) is not int or num_freqs < 0:
            raise ValueError(f"Invalid number of frequencies: {num_freqs}")
        if num_freqs == 0 and not concat_input:
            raise ValueError("Invalid combination of layer parameters")
        if num_freqs > 32:
            warnings.warn(f"Danger zone with num_freqs={num_freqs} and dtype={dtype}")
        self.num_freqs = num_freqs
        self.concat_input = concat_input
        self.dtype = dtype
        if num_freqs > 0:
            self.register_buffer(
                "freq_bands", 2 ** torch.arange(num_freqs, dtype=dtype)
            )

    @property
    def dim_multiplier(self):
        return self.num_freqs * 2 + (1 if self.concat_input else 0)

    def forward(self, x):
        """
        Embeds points into
        :param x: Tensor of shape B x D, where B is a batch dimension or dimensions, and D is the embedded space
        :return: Tensor of shape B x O, where O = (2 * F + 1) * D (+ 1 if concat_input), where F = num_freq.
        """
        if not torch.is_tensor(x) or x.dim() < 2 or x.dtype != self.dtype:
            raise ValueError("Invalid input")
        B, D = x.shape[:-1], x.shape[-1]
        out = []
        if self.concat_input:
            out = [x]  # B x D
        if self.num_freqs > 0:
            x = x.unsqueeze(-1)  # B x D x 1
            x = x * self.freq_bands  # B x D x F
            x = x.reshape(B + (self.num_freqs * D,))  # B x FD
            x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # B x 2FD
            out.append(x)
        out = torch.cat(out, dim=-1)  # B x (2F+1)D
        return out


class SirenLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, is_first=False, omega=30, act_type="sinc"):
        super().__init__()
        self.omega = omega
        if is_first:
            bound_weight = 1 / dim_in
        else:
            bound_weight = np.sqrt(6 / dim_in) / self.omega
        bound_bias = 0.1 * bound_weight
        self.linear = CustomLinear(
            dim_in, dim_out, bound_weight=bound_weight, bound_bias=bound_bias
        )
        self.act = {
            "sin": SinAct(),
            "sinc": SincAct(),
        }[act_type]

    def forward(self, x):
        x = self.omega * self.linear(x)
        x = self.act(x)
        return x


class ResBlock(torch.nn.Module):
    def __init__(
        self, dim_in, dim_out, batchnorms=True, act_type="silu", shortcut=True
    ):
        super().__init__()
        self.shortcut = shortcut
        self.linear = torch.nn.Linear(dim_in, dim_out, bias=not batchnorms)
        self.norm = torch.nn.BatchNorm1d(dim_out) if batchnorms else torch.nn.Identity()
        self.act = {
            "relu": torch.nn.ReLU(inplace=True),
            "silu": torch.nn.SiLU(inplace=True),
            "sin": SinAct(),
        }[act_type]
        if shortcut:
            if dim_in != dim_out:
                raise ValueError("Invalid layer configuration")
            self.weight = torch.nn.Parameter(torch.zeros((1,)))

    def forward(self, x):
        shortcut = x
        x = self.linear(x)
        x = self.norm(x)
        if self.shortcut:
            x = self.weight * x + shortcut
        x = self.act(x)
        return x


class SirenWithResblock(torch.nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        sirenblock_is_first=False,
        sirenblock_omega=30,
        sirenblock_act_type="sinc",
        resblock_batchnorms=True,
        resblock_act_type="silu",
        resblock_shortcut=True,
        resblock_channels_fraction=0.5,
    ):
        super().__init__()
        dim_out_resblock = max(int(resblock_channels_fraction * dim_out), 1)
        dim_out_siren = dim_out - dim_out_resblock
        self.siren = SirenLayer(
            dim_in,
            dim_out_siren,
            is_first=sirenblock_is_first,
            omega=sirenblock_omega,
            act_type=sirenblock_act_type,
        )
        self.residual = ResBlock(
            dim_in,
            dim_out_resblock,
            batchnorms=resblock_batchnorms,
            act_type=resblock_act_type,
            shortcut=resblock_shortcut,
        )

    def forward(self, x):
        return torch.cat((self.siren(x), self.residual(x)), dim=-1)


class BlockLinear(torch.nn.Module):
    def __init__(
        self,
        num_blocks,
        block_dim_in,
        block_dim_out,
        bias=True,
        init_bound_weight="auto",
        init_bound_bias="auto",
        device=None,
        dtype=None,
        checks=True,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.num_blocks = num_blocks
        self.block_dim_in = block_dim_in
        self.block_dim_out = block_dim_out
        self.dim_out = num_blocks * block_dim_out
        self.dim_in = num_blocks * block_dim_in
        self.checks = checks

        self.block_linear = torch.nn.Conv2d(
            in_channels=self.dim_in,
            out_channels=self.dim_out,
            kernel_size=1,
            groups=num_blocks,
            bias=bias,
            **factory_kwargs,
        )  # weights: [dim_out x block_dim_in x 1 x 1], bias: [dim_out]

        if init_bound_weight == "auto":
            gain = calculate_gain(
                "leaky_relu", np.sqrt(5)
            )  # same as the default Linear layer
            init_bound_weight = np.sqrt(3) * gain / np.sqrt(block_dim_in)
        if init_bound_bias == "auto":
            init_bound_bias = 1 / np.sqrt(block_dim_in)

        init_weight = torch.empty(
            self.dim_out, block_dim_in, 1, 1, **factory_kwargs
        ).uniform_(-init_bound_weight, init_bound_weight)
        init_bias = torch.empty(self.dim_out, **factory_kwargs).uniform_(
            -init_bound_bias, init_bound_bias
        )

        with torch.no_grad():
            self.block_linear.weight.copy_(init_weight)
            if bias:
                self.block_linear.bias.copy_(init_bias)

    def forward(self, x):
        B, N, D = x.shape
        if self.checks:
            if N != self.num_blocks or D != self.block_dim_in:
                raise ValueError(
                    f"Input dimension mismatch, expected ({self.num_blocks}, {self.block_dim_in}), "
                    f"encountered ({N}, {D})"
                )
        x = x.view(-1, self.dim_in, 1, 1)
        x = self.block_linear(x)
        x = x.view(B, self.num_blocks, self.block_dim_out)
        return x

    def extra_repr(self) -> str:
        return (
            f"num_blocks={self.num_blocks}, block_dim_in={self.block_dim_in}, "
            f"block_dim_out={self.block_dim_out}, dim_in={self.dim_in}, dim_out={self.dim_out}, "
            f"bias={self.block_linear.bias is not None}"
        )
