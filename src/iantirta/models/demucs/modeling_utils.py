# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.
#
# Partial code of demucs, improved by iantirta.com
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import importlib
import json
import math
import os
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import julius
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from iantirta.models.vendor.transformers.configuration_utils import PreTrainedConfig
from iantirta.models.vendor.transformers.modeling_utils import (
    PreTrainedModel,
    SpecificPreTrainedModelType,
)
from iantirta.models.vendor.transformers.utils import resolve_revision

from .configuration_utils import DemucsConfig
from .spec import ispectro, spectro
from .states import capture_init, load_model
from .transformer import CrossTransformerEncoder, LayerScale
from .utils import center_trim, unfold
from .wiener import wiener

DEFAULT_NAMESPACE = "adefossez"


def hf_repo_name(name: str) -> str:
    """Map a demucs model name to its HuggingFace repository name,
    e.g. `htdemucs_ft` -> `HTDemucs-ft`, `mdx_extra_q` -> `Demucs-mdx_extra_q`."""
    if name == 'htdemucs':
        return 'HTDemucs'
    elif name.startswith('htdemucs_'):
        return 'HTDemucs-' + name[len('htdemucs_'):]
    else:
        return 'Demucs-' + name


class BLSTM(nn.Module):
    """
    BiLSTM with same hidden units as input dim.
    If `max_steps` is not None, input will be splitting in overlapping
    chunks and the LSTM applied separately on each chunk.
    """
    def __init__(self, dim, layers=1, max_steps=None, skip=False):
        super().__init__()
        assert max_steps is None or max_steps % 4 == 0
        self.max_steps = max_steps
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = nn.Linear(2 * dim, dim)
        self.skip = skip

    def forward(self, x):
        B, C, T = x.shape
        y = x
        framed = False
        if self.max_steps is not None and T > self.max_steps:
            width = self.max_steps
            stride = width // 2
            frames = unfold(x, width, stride)
            nframes = frames.shape[2]
            framed = True
            x = frames.permute(0, 2, 1, 3).reshape(-1, C, width)

        x = x.permute(2, 0, 1)

        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        if framed:
            out = []
            frames = x.reshape(B, -1, C, width)
            limit = stride // 2
            for k in range(nframes):
                if k == 0:
                    out.append(frames[:, k, :, :-limit])
                elif k == nframes - 1:
                    out.append(frames[:, k, :, limit:])
                else:
                    out.append(frames[:, k, :, limit:-limit])
            out = torch.cat(out, -1)
            out = out[..., :T]
            x = out
        if self.skip:
            x = x + y
        return x


def rescale_conv(conv, reference):
    """Rescale initial weight scale. It is unclear why it helps but it certainly does.
    """
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
            rescale_conv(sub, reference)


class DConv(nn.Module):
    """
    New residual branches in each encoder layer.
    This alternates dilated convolutions, potentially with LSTMs and attention.
    Also before entering each residual branch, dimension is projected on a smaller subspace,
    e.g. of dim `channels // compress`.
    """
    def __init__(self, channels: int, compress: float = 4, depth: int = 2, init: float = 1e-4,
                 norm=True, attn=False, heads=4, ndecay=4, lstm=False, gelu=True,
                 kernel=3, dilate=True):
        """
        Args:
            channels: input/output channels for residual branch.
            compress: amount of channel compression inside the branch.
            depth: number of layers in the residual branch. Each layer has its own
                projection, and potentially LSTM and attention.
            init: initial scale for LayerNorm.
            norm: use GroupNorm.
            attn: use LocalAttention.
            heads: number of heads for the LocalAttention.
            ndecay: number of decay controls in the LocalAttention.
            lstm: use LSTM.
            gelu: Use GELU activation.
            kernel: kernel size for the (dilated) convolutions.
            dilate: if true, use dilation, increasing with the depth.
        """

        super().__init__()
        assert kernel % 2 == 1
        self.channels = channels
        self.compress = compress
        self.depth = abs(depth)
        dilate = depth > 0

        norm_fn: Callable[[int], nn.Module]
        norm_fn = lambda d: nn.Identity()
        if norm:
            norm_fn = lambda d: nn.GroupNorm(1, d)

        hidden = int(channels / compress)

        act: type[nn.Module]
        if gelu:
            act = nn.GELU
        else:
            act = nn.ReLU

        self.layers = nn.ModuleList([])
        for d in range(self.depth):
            dilation = 2 ** d if dilate else 1
            padding = dilation * (kernel // 2)
            mods: list[nn.Module] = [
                nn.Conv1d(channels, hidden, kernel, dilation=dilation, padding=padding),
                norm_fn(hidden), act(),
                nn.Conv1d(hidden, 2 * channels, 1),
                norm_fn(2 * channels), nn.GLU(1),
                LayerScale(channels, init),
            ]
            if attn:
                mods.insert(3, LocalState(hidden, heads=heads, ndecay=ndecay))
            if lstm:
                mods.insert(3, BLSTM(hidden, layers=2, max_steps=200, skip=True))
            layer = nn.Sequential(*mods)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class LocalState(nn.Module):
    """Local state allows to have attention based only on data (no positional embedding),
    but while setting a constraint on the time window (e.g. decaying penalty term).

    Also a failed experiments with trying to provide some frequency based attention.
    """
    def __init__(self, channels: int, heads: int = 4, nfreqs: int = 0, ndecay: int = 4):
        super().__init__()
        assert channels % heads == 0, (channels, heads)
        self.heads = heads
        self.nfreqs = nfreqs
        self.ndecay = ndecay
        self.content = nn.Conv1d(channels, channels, 1)
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        if nfreqs:
            self.query_freqs = nn.Conv1d(channels, heads * nfreqs, 1)
        if ndecay:
            self.query_decay = nn.Conv1d(channels, heads * ndecay, 1)
            # Initialize decay close to zero (there is a sigmoid), for maximum initial window.
            self.query_decay.weight.data *= 0.01
            assert self.query_decay.bias is not None  # stupid type checker
            self.query_decay.bias.data[:] = -2
        self.proj = nn.Conv1d(channels + heads * nfreqs, channels, 1)

    def forward(self, x):
        B, C, T = x.shape
        heads = self.heads
        indexes = torch.arange(T, device=x.device, dtype=x.dtype)
        # left index are keys, right index are queries
        delta = indexes[:, None] - indexes[None, :]

        queries = self.query(x).view(B, heads, -1, T)
        keys = self.key(x).view(B, heads, -1, T)
        # t are keys, s are queries
        dots = torch.einsum("bhct,bhcs->bhts", keys, queries)
        dots /= keys.shape[2]**0.5
        if self.nfreqs:
            periods = torch.arange(1, self.nfreqs + 1, device=x.device, dtype=x.dtype)
            freq_kernel = torch.cos(2 * math.pi * delta / periods.view(-1, 1, 1))
            freq_q = self.query_freqs(x).view(B, heads, -1, T) / self.nfreqs ** 0.5
            dots += torch.einsum("fts,bhfs->bhts", freq_kernel, freq_q)
        if self.ndecay:
            decays = torch.arange(1, self.ndecay + 1, device=x.device, dtype=x.dtype)
            decay_q = self.query_decay(x).view(B, heads, -1, T)
            decay_q = torch.sigmoid(decay_q) / 2
            decay_kernel = - decays.view(-1, 1, 1) * delta.abs() / self.ndecay**0.5
            dots += torch.einsum("fts,bhfs->bhts", decay_kernel, decay_q)

        # Kill self reference.
        dots.masked_fill_(torch.eye(T, device=dots.device, dtype=torch.bool), -100)
        weights = torch.softmax(dots, dim=2)

        content = self.content(x).view(B, heads, -1, T)
        result = torch.einsum("bhts,bhct->bhcs", weights, content)
        if self.nfreqs:
            time_sig = torch.einsum("bhts,fts->bhfs", weights, freq_kernel)
            result = torch.cat([result, time_sig], 2)
        result = result.reshape(B, -1, T)
        return x + self.proj(result)


class ScaledEmbedding(nn.Module):
    """
    Boost learning rate for embeddings (with `scale`).
    Also, can make embeddings continuous with `smooth`.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 scale: float = 10., smooth=False):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if smooth:
            weight = torch.cumsum(self.embedding.weight.data, dim=0)
            # when summing gaussian, overscale raises as sqrt(n), so we nornalize by that.
            weight = weight / torch.arange(1, num_embeddings + 1).to(weight).sqrt()[:, None]
            self.embedding.weight.data[:] = weight
        self.embedding.weight.data /= scale
        self.scale = scale

    @property
    def weight(self):
        return self.embedding.weight * self.scale

    def forward(self, x):
        out = self.embedding(x) * self.scale
        return out


class HEncLayer(nn.Module):
    def __init__(self, chin, chout, kernel_size=8, stride=4, norm_groups=1, empty=False,
                 freq=True, dconv=True, norm=True, context=0, dconv_kw={}, pad=True,
                 rewrite=True):
        """Encoder layer. This used both by the time and the frequency branch.

        Args:
            chin: number of input channels.
            chout: number of output channels.
            norm_groups: number of groups for group norm.
            empty: used to make a layer with just the first conv. this is used
                before merging the time and freq. branches.
            freq: this is acting on frequencies.
            dconv: insert DConv residual branches.
            norm: use GroupNorm.
            context: context size for the 1x1 conv.
            dconv_kw: list of kwargs for the DConv class.
            pad: pad the input. Padding is done so that the output size is
                always the input size / stride.
            rewrite: add 1x1 conv at the end of the layer.
        """
        super().__init__()
        norm_fn = lambda d: nn.Identity()
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)
        if pad:
            pad = kernel_size // 4
        else:
            pad = 0
        klass = nn.Conv1d
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.empty = empty
        self.norm = norm
        self.pad = pad
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            pad = [pad, 0]
            klass = nn.Conv2d
        self.conv = klass(chin, chout, kernel_size, stride, pad)
        if self.empty:
            return
        self.norm1 = norm_fn(chout)
        self.rewrite = None
        if rewrite:
            self.rewrite = klass(chout, 2 * chout, 1 + 2 * context, 1, context)
            self.norm2 = norm_fn(2 * chout)

        self.dconv = None
        if dconv:
            self.dconv = DConv(chout, **dconv_kw)

    def forward(self, x, inject=None):
        """
        `inject` is used to inject the result from the time branch into the frequency branch,
        when both have the same stride.
        """
        if not self.freq and x.dim() == 4:
            B, C, Fr, T = x.shape
            x = x.view(B, -1, T)

        if not self.freq:
            le = x.shape[-1]
            if not le % self.stride == 0:
                x = F.pad(x, (0, self.stride - (le % self.stride)))
        y = self.conv(x)
        if self.empty:
            return y
        if inject is not None:
            assert inject.shape[-1] == y.shape[-1], (inject.shape, y.shape)
            if inject.dim() == 3 and y.dim() == 4:
                inject = inject[:, :, None]
            y = y + inject
        y = F.gelu(self.norm1(y))
        if self.dconv:
            if self.freq:
                B, C, Fr, T = y.shape
                y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
            y = self.dconv(y)
            if self.freq:
                y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        if self.rewrite:
            z = self.norm2(self.rewrite(y))
            z = F.glu(z, dim=1)
        else:
            z = y
        return z


class MultiWrap(nn.Module):
    """
    Takes one layer and replicate it N times. each replica will act
    on a frequency band. All is done so that if the N replica have the same weights,
    then this is exactly equivalent to applying the original module on all frequencies.

    This is a bit over-engineered to avoid edge artifacts when splitting
    the frequency bands, but it is possible the naive implementation would work as well...
    """
    def __init__(self, layer, split_ratios):
        """
        Args:
            layer: module to clone, must be either HEncLayer or HDecLayer.
            split_ratios: list of float indicating which ratio to keep for each band.
        """
        super().__init__()
        self.split_ratios = split_ratios
        self.layers = nn.ModuleList()
        self.conv = isinstance(layer, HEncLayer)
        assert not layer.norm
        assert layer.freq
        assert layer.pad
        if not self.conv:
            assert not layer.context_freq
        for k in range(len(split_ratios) + 1):
            lay = copy.deepcopy(layer)
            if self.conv:
                lay.conv.padding = (0, 0)
            else:
                lay.pad = False
            for m in lay.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            self.layers.append(lay)

    def forward(self, x, skip=None, length=None):
        B, C, Fr, T = x.shape

        ratios = list(self.split_ratios) + [1]
        start = 0
        outs = []
        for ratio, layer in zip(ratios, self.layers):
            if self.conv:
                pad = layer.kernel_size // 4
                if ratio == 1:
                    limit = Fr
                    frames = -1
                else:
                    limit = int(round(Fr * ratio))
                    le = limit - start
                    if start == 0:
                        le += pad
                    frames = round((le - layer.kernel_size) / layer.stride + 1)
                    limit = start + (frames - 1) * layer.stride + layer.kernel_size
                    if start == 0:
                        limit -= pad
                assert limit - start > 0, (limit, start)
                assert limit <= Fr, (limit, Fr)
                y = x[:, :, start:limit, :]
                if start == 0:
                    y = F.pad(y, (0, 0, pad, 0))
                if ratio == 1:
                    y = F.pad(y, (0, 0, 0, pad))
                outs.append(layer(y))
                start = limit - layer.kernel_size + layer.stride
            else:
                if ratio == 1:
                    limit = Fr
                else:
                    limit = int(round(Fr * ratio))
                last = layer.last
                layer.last = True

                y = x[:, :, start:limit]
                s = skip[:, :, start:limit]
                out, _ = layer(y, s, None)
                if outs:
                    outs[-1][:, :, -layer.stride:] += (
                        out[:, :, :layer.stride] - layer.conv_tr.bias.view(1, -1, 1, 1))
                    out = out[:, :, layer.stride:]
                if ratio == 1:
                    out = out[:, :, :-layer.stride // 2, :]
                if start == 0:
                    out = out[:, :, layer.stride // 2:, :]
                outs.append(out)
                layer.last = last
                start = limit
        out = torch.cat(outs, dim=2)
        if not self.conv and not last:
            out = F.gelu(out)
        if self.conv:
            return out
        else:
            return out, None


class HDecLayer(nn.Module):
    def __init__(self, chin, chout, last=False, kernel_size=8, stride=4, norm_groups=1, empty=False,
                 freq=True, dconv=True, norm=True, context=1, dconv_kw={}, pad=True,
                 context_freq=True, rewrite=True):
        """
        Same as HEncLayer but for decoder. See `HEncLayer` for documentation.
        """
        super().__init__()
        norm_fn = lambda d: nn.Identity()
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)
        if pad:
            pad = kernel_size // 4
        else:
            pad = 0
        self.pad = pad
        self.last = last
        self.freq = freq
        self.chin = chin
        self.empty = empty
        self.stride = stride
        self.kernel_size = kernel_size
        self.norm = norm
        self.context_freq = context_freq
        klass = nn.Conv1d
        klass_tr = nn.ConvTranspose1d
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            klass = nn.Conv2d
            klass_tr = nn.ConvTranspose2d
        self.conv_tr = klass_tr(chin, chout, kernel_size, stride)
        self.norm2 = norm_fn(chout)
        if self.empty:
            return
        self.rewrite = None
        if rewrite:
            if context_freq:
                self.rewrite = klass(chin, 2 * chin, 1 + 2 * context, 1, context)
            else:
                self.rewrite = klass(chin, 2 * chin, [1, 1 + 2 * context], 1,
                                     [0, context])
            self.norm1 = norm_fn(2 * chin)

        self.dconv = None
        if dconv:
            self.dconv = DConv(chin, **dconv_kw)

    def forward(self, x, skip, length):
        if self.freq and x.dim() == 3:
            B, C, T = x.shape
            x = x.view(B, self.chin, -1, T)

        if not self.empty:
            x = x + skip

            if self.rewrite:
                y = F.glu(self.norm1(self.rewrite(x)), dim=1)
            else:
                y = x
            if self.dconv:
                if self.freq:
                    B, C, Fr, T = y.shape
                    y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
                y = self.dconv(y)
                if self.freq:
                    y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        else:
            y = x
            assert skip is None
        z = self.norm2(self.conv_tr(y))
        if self.freq:
            if self.pad:
                z = z[..., self.pad:-self.pad, :]
        else:
            z = z[..., self.pad:self.pad + length]
            assert z.shape[-1] == length, (z.shape[-1], length)
        if not self.last:
            z = F.gelu(z)
        return z, y


def pad1d(x: torch.Tensor, paddings: tuple[int, int], mode: str = 'constant', value: float = 0.):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen."""
    x0 = x
    length = x.shape[-1]
    padding_left, padding_right = paddings
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            extra_pad_right = min(padding_right, extra_pad)
            extra_pad_left = extra_pad - extra_pad_right
            paddings = (padding_left - extra_pad_left, padding_right - extra_pad_right)
            x = F.pad(x, (extra_pad_left, extra_pad_right))
    out = F.pad(x, paddings, mode, value)
    assert out.shape[-1] == length + padding_left + padding_right
    assert (out[..., padding_left: padding_left + length] == x0).all()
    return out


class DemucsPreTrainedModel(PreTrainedModel):
    config: DemucsConfig

    @classmethod
    def from_pretrained(
        cls: type[SpecificPreTrainedModelType],
        pretrained_model_name_or_path: str | os.PathLike | None,
        *model_args,
        config: PreTrainedConfig | str | os.PathLike | None = None,
        cache_dir: str | os.PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool | None = None,
        weights_only: bool = True,
        fusion_config: dict[str, bool | dict[str, Any]] | None = None,
        disable_mmap: bool | None = None,
        **kwargs,
    ) -> SpecificPreTrainedModelType:
        state_dict = kwargs.pop("state_dict", None)
        proxies = kwargs.pop("proxies", None)
        tqdm_class = kwargs.pop("tqdm_class", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        dtype = kwargs.pop("dtype", None)
        torch_dtype = kwargs.pop("torch_dtype", None)  # kept for BC
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_buffers = kwargs.pop("offload_buffers", False)
        quantization_config = kwargs.pop("quantization_config", None)
        subfolder = kwargs.pop("subfolder", "")
        kwargs.pop("_commit_hash", None)  # BC: not used anymore, `revision` is resolved to a commit hash instead
        variant = kwargs.pop("variant", None)
        adapter_kwargs = (kwargs.pop("adapter_kwargs", {}) or {}).copy()
        adapter_name = kwargs.pop("adapter_name", "default")
        generation_config = kwargs.pop("generation_config", None)
        gguf_file = kwargs.pop("gguf_file", None)
        distributed_config = kwargs.pop("distributed_config", None)
        device_mesh = kwargs.pop("device_mesh", None)
        tp_plan = kwargs.pop("tp_plan", None)
        tp_size = kwargs.pop("tp_size", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        allow_all_kernels = kwargs.pop("allow_all_kernels", False)
        use_kernels = kwargs.pop("use_kernels", False)
        kernel_config = kwargs.pop("kernel_config", None)
        key_mapping = kwargs.pop("key_mapping", None)
        
        namespace, name = DEFAULT_NAMESPACE, pretrained_model_name_or_path
        if "/" in pretrained_model_name_or_path:
            namespace, name = pretrained_model_name_or_path.split("/", 1)
        pretrained_model_name_or_path = f"{namespace}/{hf_repo_name(name)}"

        # Resolve the revision once and for all: config, weights, generation config and adapters are then all loaded
        # from the exact same repository state, without any further call to the Hub to revalidate a mutable revision.
        requested_revision = revision
        revision = resolve_revision(
            pretrained_model_name_or_path,
            revision,
            token=token,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )

        download_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "local_files_only": local_files_only,
            "token": token,
            "revision": revision,
            "subfolder": subfolder,
        }

        # Load config if we don't provide a configuration
        if not isinstance(config, PreTrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config_class = cls.config_class
            if config_class is None:
                raise ValueError(
                    f"{cls.__name__} does not define `config_class`; pass an explicit config to `from_pretrained`."
                )
            config, model_kwargs = config_class.from_pretrained(
                config_path,
                return_unused_kwargs=True,
                gguf_file=gguf_file,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                _configuration_file=f"{name}.yaml",
                **download_kwargs,
                **kwargs,
            )
            if "gguf_file" in model_kwargs:
                model_kwargs.pop("gguf_file")
        else:
            config = copy.deepcopy(config)
            model_kwargs = kwargs

        config.name_or_path = pretrained_model_name_or_path

        model = cls(config, *model_args, **model_kwargs)
        model.eval()
        return model


class DemucsModel(DemucsPreTrainedModel):
    sources: list[str]

    # Channels
    audio_channels: int = 2
    channels: int = 64
    growth: float = 2.

    # Main structure
    depth: int = 6
    rewrite: bool = True
    
    # Convolutions
    kernel_size: int = 8
    stride: int = 4
    context: int = 1
    
    # Normalization
    norm_starts: int = 4
    norm_groups: int = 4

    # DConv residual branch
    dconv_mode=1
    dconv_depth=2
    dconv_comp=4
    dconv_attn=4
    dconv_lstm=4
    dconv_init=1e-4

    # Weight init
    rescale: float = 0.1

    # Metadata
    samplerate: int = 44100
    segment: int = 4 * 10

    def __init__(self, config, *inputs, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        super().__init__(config, *inputs, **kwargs)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()


class DemucsDemucs(DemucsModel):
    # Main structure
    lstm_layers=0

    # Activations
    gelu=True
    glu=True

    # Pre/post processing
    normalize=True
    resample=True

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        
        self.skip_scales = nn.ModuleList()

        if self.glu:
            activation = nn.GLU(dim=1)
            ch_scale = 2
        else:
            activation = nn.ReLU()
            ch_scale = 1
        if self.gelu:
            act2 = nn.GELU
        else:
            act2 = nn.ReLU

        in_channels = self.audio_channels
        padding = 0

        for index in range(self.depth):
            norm_fn = lambda d: nn.Identity()
            if index >= self.norm_starts:
                norm_fn = lambda d: nn.GroupNorm(self.norm_groups, d)

            encode = []
            encode += [
                nn.Conv1d(
                    in_channels,
                    self.channels,
                    self.kernel_size,
                    self.stride
                ),
                norm_fn(self.channels),
                act2(),
            ]
            attn = index >= self.dconv_attn
            lstm = index >= self.dconv_lstm
            if self.dconv_mode & 1:
                encode += [
                    DConv(
                        self.channels,
                        depth=self.dconv_depth,
                        init=self.dconv_init,
                        compress=self.dconv_comp,
                        attn=attn,
                        lstm=lstm
                    )
                ]
            if self.rewrite:
                encode += [
                    nn.Conv1d(
                        self.channels,
                        ch_scale * self.channels,
                        1
                    ),
                    norm_fn(ch_scale * self.channels),
                    activation
                ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            if index > 0:
                out_channels = in_channels
            else:
                out_channels = len(self.sources) * self.audio_channels
            if self.rewrite:
                decode += [
                    nn.Conv1d(
                        self.channels,
                        ch_scale * self.channels,
                        2 * self.context + 1,
                        padding=self.context
                    ),
                    norm_fn(ch_scale * self.channels),
                    activation
                ]
            if self.dconv_mode & 2:
                decode += [
                    DConv(
                        self.channels,
                        depth=self.dconv_depth,
                        init=self.dconv_init,
                        compress=self.dconv_comp,
                        attn=attn,
                        lstm=lstm
                    )
                ]
            decode += [
                nn.ConvTranspose1d(
                    self.channels, out_channels,
                    self.kernel_size, self.stride,
                    padding=padding
                )
            ]
            if index > 0:
                decode += [norm_fn(out_channels), act2()]
            self.decoder.insert(0, nn.Sequential(*decode))
            in_channels = self.channels
            channels = int(self.growth * self.channels)

        channels = in_channels
        if self.lstm_layers:
            self.lstm = BLSTM(channels, self.lstm_layers)
        else:
            self.lstm = None

        if self.rescale:
            rescale_module(self, reference=self.rescale)

      
    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolution, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        Note that input are automatically padded if necessary to ensure that the output
        has the same length as the input.
        """
        if self.resample:
            length *= 2

        for _ in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)

        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size

        if self.resample:
            length = math.ceil(length / 2)
        return int(length)


    def forward(self, mix):
        x = mix
        length = x.shape[-1]

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            mean = mono.mean(dim=-1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            x = (x - mean) / (1e-5 + std)
        else:
            mean = 0
            std = 1

        delta = self.valid_length(length) - length
        x = F.pad(x, (delta // 2, delta - delta // 2))

        if self.resample:
            x = julius.resample_frac(x, 1, 2)

        saved = []
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)

        if self.lstm:
            x = self.lstm(x)

        for decode in self.decoder:
            skip = saved.pop(-1)
            skip = center_trim(skip, x)
            x = decode(x + skip)

        if self.resample:
            x = julius.resample_frac(x, 2, 1)
        x = x * std + mean
        x = center_trim(x, length)
        x = x.view(x.size(0), len(self.sources), self.audio_channels, x.size(-1))
        return x

    def load_state_dict(self, state, strict=True):
        # fix a mismatch with previous generation Demucs models.
        for idx in range(self.depth):
            for a in ['encoder', 'decoder']:
                for b in ['bias', 'weight']:
                    new = f'{a}.{idx}.3.{b}'
                    old = f'{a}.{idx}.2.{b}'
                    if old in state and new not in state:
                        state[new] = state.pop(old)
        super().load_state_dict(state, strict=strict)


class DemucsHDemucs(DemucsModel):
    # Channels
    channels: int = 48
    channels_time=None
    growth: float = 2

    # STFT
    nfft: int = 4096
    wiener_iters: int = 0
    end_iters: int = 0
    wiener_residual: bool = False
    cac: bool = True

    # Main structure
    hybrid: bool = True
    hybrid_old: bool = False

    # Frequency branch
    multi_freqs=None
    multi_freqs_depth=2
    emb_scale=10
    emb_smooth=True

    # Convolutions
    time_stride=2
    context_enc=0

    @capture_init
    def __init__(self, config, *inputs, freq_emb=0.2, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        
        self.freq_emb = None
        self.hop_length = self.nfft // 4

        if self.hybrid_old:
            assert self.hybrid, "hybrid_old must come with hybrid=True"
        if self.hybrid:
            assert self.wiener_iters == self.end_iters

        if self.hybrid:
            self.tencoder = nn.ModuleList()
            self.tdecoder = nn.ModuleList()
        
        chin = self.audio_channels
        chin_z = chin  # number of channels for the freq branch
        if self.cac:
            chin_z *= 2
        chout = self.channels_time or self.channels
        chout_z = self.channels
        freqs = self.nfft // 2
            
        for index in range(self.depth):
            lstm = index >= self.dconv_lstm
            attn = index >= self.dconv_attn
            norm = index >= self.norm_starts
            freq = freqs > 1
            stri = self.stride
            ker = self.kernel_size

            if not freq:
                assert freqs == 1
                ker = self.time_stride * 2
                stri = self.time_stride

            pad = True
            last_freq = False
            if freq and freqs <= self.kernel_size:
                ker = freqs
                pad = False
                last_freq = True

            kw = {
                'kernel_size': ker,
                'stride': stri,
                'freq': freq,
                'pad': pad,
                'norm': norm,
                'rewrite': self.rewrite,
                'norm_groups': self.norm_groups,
                'dconv_kw': {
                    'lstm': lstm,
                    'attn': attn,
                    'depth': self.dconv_depth,
                    'compress': self.dconv_comp,
                    'init': self.dconv_init,
                    'gelu': True,
                }
            }
            kwt = dict(kw)
            kwt['freq'] = 0
            kwt['kernel_size'] = self.kernel_size
            kwt['stride'] = self.stride
            kwt['pad'] = True
            kw_dec = dict(kw)
            multi = False

            if self.multi_freqs and index < self.multi_freqs_depth:
                multi = True
                kw_dec['context_freq'] = False

            if last_freq:
                chout_z = max(chout, chout_z)
                chout = chout_z

            enc = HEncLayer(
                chin_z, chout_z,
                dconv=self.dconv_mode & 1,
                context=self.context_enc, **kw
            )

            if self.hybrid and freq:
                tenc = HEncLayer(
                    chin, chout, dconv=self.dconv_mode & 1,
                    context=self.context_enc,
                    empty=last_freq, **kwt
                )
                self.tencoder.append(tenc)
            
            if multi:
                enc = MultiWrap(enc, self.multi_freqs)

            self.encoder.append(enc)
            
            if index == 0:
                chin = self.audio_channels * len(self.sources)
                chin_z = chin
                if self.cac:
                    chin_z *= 2
            
            dec = HDecLayer(
                chout_z, chin_z,
                dconv=self.dconv_mode & 2,
                last=index == 0,
                context=self.context,
                **kw_dec
            )
            
            if multi:
                dec = MultiWrap(dec, self.multi_freqs)
            if self.hybrid and freq:
                tdec = HDecLayer(
                    chout, chin,
                    dconv=self.dconv_mode & 2,
                    empty=last_freq,
                    last=index == 0,
                    context=self.context,
                    **kwt
                )
                self.tdecoder.insert(0, tdec)
            
            self.decoder.insert(0, dec)
            
            chin = chout
            chin_z = chout_z
            chout = int(self.growth * chout)
            chout_z = int(self.growth * chout_z)
            if freq:
                if freqs <= self.kernel_size:
                    freqs = 1
                else:
                    freqs //= self.stride
            if index == 0 and freq_emb:
                self.freq_emb_scale = freq_emb
                self.freq_emb = ScaledEmbedding(
                    freqs, chin_z,
                    smooth=self.emb_smooth,
                    scale=self.emb_scale
                )

        if self.rescale:
            rescale_module(self, reference=self.rescale)


    def _spec(self, x):
        hl = self.hop_length
        nfft = self.nfft
        x0 = x  # noqa

        if self.hybrid:
            # We re-pad the signal in order to keep the property
            # that the size of the output is exactly the size of the input
            # divided by the stride (here hop_length), when divisible.
            # This is achieved by padding by 1/4th of the kernel size (here nfft).
            # which is not supported by torch.stft.
            # Having all convolution operations follow this convention allow to easily
            # align the time and frequency branches later on.
            assert hl == nfft // 4
            le = int(math.ceil(x.shape[-1] / hl))
            pad = hl // 2 * 3
            if not self.hybrid_old:
                x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode='reflect')
            else:
                x = pad1d(x, (pad, pad + le * hl - x.shape[-1]))

        z = spectro(x, nfft, hl)[..., :-1, :]
        if self.hybrid:
            assert z.shape[-1] == le + 4, (z.shape, x.shape, le)
            z = z[..., 2:2+le]
        return z

    def _ispec(self, z, length=None, scale=0):
        hl = self.hop_length // (4 ** scale)
        z = F.pad(z, (0, 0, 0, 1))
        if self.hybrid:
            z = F.pad(z, (2, 2))
            pad = hl // 2 * 3
            if not self.hybrid_old:
                le = hl * int(math.ceil(length / hl)) + 2 * pad
            else:
                le = hl * int(math.ceil(length / hl))
            x = ispectro(z, hl, length=le)
            if not self.hybrid_old:
                x = x[..., pad:pad + length]
            else:
                x = x[..., :length]
        else:
            x = ispectro(z, hl, length)
        return x

    def _magnitude(self, z):
        # return the magnitude of the spectrogram, except when cac is True,
        # in which case we just move the complex dimension to the channel one.
        if self.cac:
            B, C, Fr, T = z.shape
            m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
            m = m.reshape(B, C * 2, Fr, T)
        else:
            m = z.abs()
        return m

    def _mask(self, z, m):
        # Apply masking given the mixture spectrogram `z` and the estimated mask `m`.
        # If `cac` is True, `m` is actually a full spectrogram and `z` is ignored.
        niters = self.wiener_iters
        if self.cac:
            B, S, C, Fr, T = m.shape
            out = m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
            out = torch.view_as_complex(out.contiguous())
            return out
        if self.training:
            niters = self.end_iters
        if niters < 0:
            z = z[:, None]
            return z / (1e-8 + z.abs()) * m
        else:
            return self._wiener(m, z, niters)

    def _wiener(self, mag_out, mix_stft, niters):
        # apply wiener filtering from OpenUnmix.
        init = mix_stft.dtype
        wiener_win_len = 300
        residual = self.wiener_residual

        B, S, C, Fq, T = mag_out.shape
        mag_out = mag_out.permute(0, 4, 3, 2, 1)
        mix_stft = torch.view_as_real(mix_stft.permute(0, 3, 2, 1))

        outs = []
        for sample in range(B):
            pos = 0
            out = []
            for pos in range(0, T, wiener_win_len):
                frame = slice(pos, pos + wiener_win_len)
                z_out = wiener(
                    mag_out[sample, frame], mix_stft[sample, frame], niters,
                    residual=residual)
                out.append(z_out.transpose(-1, -2))
            outs.append(torch.cat(out, dim=0))
        out = torch.view_as_complex(torch.stack(outs, 0))
        out = out.permute(0, 4, 3, 2, 1).contiguous()
        if residual:
            out = out[:, :-1]
        assert list(out.shape) == [B, S, C, Fq, T]
        return out.to(init)


    def forward(self, mix):
        x = mix
        length = x.shape[-1]

        z = self._spec(mix)
        mag = self._magnitude(z).to(mix.device)
        x = mag

        B, C, Fq, T = x.shape

        # unlike previous Demucs, we always normalize because it is easier.
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        # x will be the freq. branch input.

        if self.hybrid:
            # Prepare the time branch input.
            xt = mix
            meant = xt.mean(dim=(1, 2), keepdim=True)
            stdt = xt.std(dim=(1, 2), keepdim=True)
            xt = (xt - meant) / (1e-5 + stdt)

        # okay, this is a giant mess I know...
        saved = []  # skip connections, freq.
        saved_t = []  # skip connections, time.
        lengths = []  # saved lengths to properly remove padding, freq branch.
        lengths_t = []  # saved lengths for time branch.
        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if self.hybrid and idx < len(self.tencoder):
                # we have not yet merged branches.
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    # save for skip connection
                    saved_t.append(xt)
                else:
                    # tenc contains just the first conv., so that now time and freq.
                    # branches have the same shape and can be merged.
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                # add frequency embedding to allow for non equivariant convolutions
                # over the frequency axis.
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb

            saved.append(x)

        x = torch.zeros_like(x)
        if self.hybrid:
            xt = torch.zeros_like(x)
        # initialize everything to zero (signal will go through u-net skips).

        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            # `pre` contains the output just before final transposed convolution,
            # which is used when the freq. and time branch separate.

            if self.hybrid:
                offset = self.depth - len(self.tdecoder)
            if self.hybrid and idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    assert pre.shape[2] == 1, pre.shape
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        # Let's make sure we used all stored skip connections.
        assert len(saved) == 0
        assert len(lengths_t) == 0
        assert len(saved_t) == 0

        S = len(self.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        # to cpu as mps doesnt support complex numbers
        # demucs issue #435 ##432
        # NOTE: in this case z already is on cpu
        # TODO: remove this when mps supports complex numbers
        x_is_mps_xpu = x.device.type in ["mps", "xpu"]
        x_device = x.device
        if x_is_mps_xpu:
            x = x.cpu()

        zout = self._mask(z, x)
        x = self._ispec(zout, length)

        # back to mps device
        if x_is_mps_xpu:
            x = x.to(x_device)

        if self.hybrid:
            xt = xt.view(B, S, -1, length)
            xt = xt * stdt[:, None] + meant[:, None]
            x = xt + x
        return x


class DemucsHTDemucs(DemucsModel):
    # Channels
    channels: int = 48
    channels_time = None

    # STFT
    nfft: int = 4096
    wiener_iters: int = 0
    end_iters: int = 0
    wiener_residual: bool = False
    cac: bool = True

    # Main structure
    depth: int = 4

    # Frequency branch
    multi_freqs=None
    multi_freqs_depth=3
    emb_scale=10
    emb_smooth=True
    
    # Convolutions
    time_stride=2
    context_enc=0

    # DConv residual branch
    dconv_comp=8
    dconv_init=1e-3

    # Before the Transformer
    bottom_channels=0

    # Transformer
    t_layers=5
    t_emb="sin"
    t_hidden_scale=4.0
    t_heads=8
    t_dropout=0.0
    t_max_positions=10000
    t_norm_in=True
    t_norm_in_group=False
    t_group_norm=False
    t_norm_first=True
    t_norm_out=True
    t_max_period=10000.0
    t_weight_decay=0.0
    t_lr=None
    t_layer_scale=True
    t_gelu=True
    t_weight_pos_embed=1.0
    t_sin_random_shift=0
    t_cape_mean_normalize=True
    t_cape_augment=True
    t_cape_glob_loc_scale=[5000.0, 1.0, 1.4]
    t_sparse_self_attn=False
    t_sparse_cross_attn=False
    t_mask_type="diag"
    t_mask_random_seed=42
    t_sparse_attn_window=500
    t_global_window=100
    t_sparsity=0.95
    t_auto_sparsity=False
    # ------ Particuliar parameters
    t_cross_first=False

    # Metadata
    segment: int = 10
    use_train_segment: bool = True

    @capture_init
    def __init__(self, config, *inputs, freq_emb=0.2, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        
        self.freq_emb = None
        self.hop_length = self.nfft // 4

        assert self.wiener_iters == self.end_iters

        self.tencoder = nn.ModuleList()
        self.tdecoder = nn.ModuleList()

        chin = self.audio_channels
        chin_z = chin  # number of channels for the freq branch
        if self.cac:
            chin_z *= 2
        chout = self.channels_time or self.channels
        chout_z = self.channels
        freqs = self.nfft // 2

        for index in range(self.depth):
            norm = index >= self.norm_starts
            freq = freqs > 1
            stri = self.stride
            ker = self.kernel_size
            if not freq:
                assert freqs == 1
                ker = self.time_stride * 2
                stri = self.time_stride

            pad = True
            last_freq = False
            if freq and freqs <= self.kernel_size:
                ker = freqs
                pad = False
                last_freq = True

            kw = {
                "kernel_size": ker,
                "stride": stri,
                "freq": freq,
                "pad": pad,
                "norm": norm,
                "rewrite": self.rewrite,
                "norm_groups": self.norm_groups,
                "dconv_kw": {
                    "depth": self.dconv_depth,
                    "compress": self.dconv_comp,
                    "init": self.dconv_init,
                    "gelu": True,
                },
            }
            kwt = dict(kw)
            kwt["freq"] = 0
            kwt["kernel_size"] = self.kernel_size
            kwt["stride"] = self.stride
            kwt["pad"] = True
            kw_dec = dict(kw)
            multi = False

            if self.multi_freqs and index < self.multi_freqs_depth:
                multi = True
                kw_dec["context_freq"] = False

            if last_freq:
                chout_z = max(chout, chout_z)
                chout = chout_z

            enc = HEncLayer(
                chin_z, chout_z,
                dconv=self.dconv_mode & 1,
                context=self.context_enc,
                **kw
            )
            if freq:
                tenc = HEncLayer(
                    chin,
                    chout,
                    dconv=self.dconv_mode & 1,
                    context=self.context_enc,
                    empty=last_freq,
                    **kwt
                )
                self.tencoder.append(tenc)

            if multi:
                enc = MultiWrap(enc, self.multi_freqs)
            
            self.encoder.append(enc)

            if index == 0:
                chin = self.audio_channels * len(self.sources)
                chin_z = chin
                if self.cac:
                    chin_z *= 2
            
            dec = HDecLayer(
                chout_z,
                chin_z,
                dconv=self.dconv_mode & 2,
                last=index == 0,
                context=self.context,
                **kw_dec
            )
            if multi:
                dec = MultiWrap(dec, self.multi_freqs)
            if freq:
                tdec = HDecLayer(
                    chout,
                    chin,
                    dconv=self.dconv_mode & 2,
                    empty=last_freq,
                    last=index == 0,
                    context=self.context,
                    **kwt
                )
                self.tdecoder.insert(0, tdec)
            
            self.decoder.insert(0, dec)

            chin = chout
            chin_z = chout_z
            chout = int(self.growth * chout)
            chout_z = int(self.growth * chout_z)
            if freq:
                if freqs <= self.kernel_size:
                    freqs = 1
                else:
                    freqs //= self.stride
            if index == 0 and freq_emb:
                self.freq_emb_scale = freq_emb
                self.freq_emb = ScaledEmbedding(
                    freqs, chin_z,
                    smooth=self.emb_smooth,
                    scale=self.emb_scale
                )

        if self.rescale:
            rescale_module(self, reference=self.rescale)

        transformer_channels = self.channels * self.growth ** (self.depth - 1)
        if self.bottom_channels:
            self.channel_upsampler = nn.Conv1d(
                transformer_channels,
                self.bottom_channels,
                1
            )
            self.channel_downsampler = nn.Conv1d(
                self.bottom_channels, transformer_channels, 1
            )
            self.channel_upsampler_t = nn.Conv1d(
                transformer_channels, self.bottom_channels, 1
            )
            self.channel_downsampler_t = nn.Conv1d(
                self.bottom_channels, transformer_channels, 1
            )

            transformer_channels = self.bottom_channels

        if self.t_layers > 0:
            self.crosstransformer = CrossTransformerEncoder(
                dim=transformer_channels,
                emb=self.t_emb,
                hidden_scale=self.t_hidden_scale,
                num_heads=self.t_heads,
                num_layers=self.t_layers,
                cross_first=self.t_cross_first,
                dropout=self.t_dropout,
                max_positions=self.t_max_positions,
                norm_in=self.t_norm_in,
                norm_in_group=self.t_norm_in_group,
                group_norm=self.t_group_norm,
                norm_first=self.t_norm_first,
                norm_out=self.t_norm_out,
                max_period=self.t_max_period,
                weight_decay=self.t_weight_decay,
                lr=self.t_lr,
                layer_scale=self.t_layer_scale,
                gelu=self.t_gelu,
                sin_random_shift=self.t_sin_random_shift,
                weight_pos_embed=self.t_weight_pos_embed,
                cape_mean_normalize=self.t_cape_mean_normalize,
                cape_augment=self.t_cape_augment,
                cape_glob_loc_scale=self.t_cape_glob_loc_scale,
                sparse_self_attn=self.t_sparse_self_attn,
                sparse_cross_attn=self.t_sparse_cross_attn,
                mask_type=self.t_mask_type,
                mask_random_seed=self.t_mask_random_seed,
                sparse_attn_window=self.t_sparse_attn_window,
                global_window=self.t_global_window,
                sparsity=self.t_sparsity,
                auto_sparsity=self.t_auto_sparsity,
            )
        else:
            self.crosstransformer = None

    
    def _spec(self, x):
        hl = self.hop_length
        nfft = self.nfft
        x0 = x  # noqa

        # We re-pad the signal in order to keep the property
        # that the size of the output is exactly the size of the input
        # divided by the stride (here hop_length), when divisible.
        # This is achieved by padding by 1/4th of the kernel size (here nfft).
        # which is not supported by torch.stft.
        # Having all convolution operations follow this convention allow to easily
        # align the time and frequency branches later on.
        assert hl == nfft // 4
        le = int(math.ceil(x.shape[-1] / hl))
        pad = hl // 2 * 3
        x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")

        z = spectro(x, nfft, hl)[..., :-1, :]
        assert z.shape[-1] == le + 4, (z.shape, x.shape, le)
        z = z[..., 2: 2 + le]
        return z

    def _ispec(self, z, length=None, scale=0):
        hl = self.hop_length // (4**scale)
        z = F.pad(z, (0, 0, 0, 1))
        z = F.pad(z, (2, 2))
        pad = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad
        x = ispectro(z, hl, length=le)
        x = x[..., pad: pad + length]
        return x

    def _magnitude(self, z):
        # return the magnitude of the spectrogram, except when cac is True,
        # in which case we just move the complex dimension to the channel one.
        if self.cac:
            B, C, Fr, T = z.shape
            m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
            m = m.reshape(B, C * 2, Fr, T)
        else:
            m = z.abs()
        return m

    def _mask(self, z, m):
        # Apply masking given the mixture spectrogram `z` and the estimated mask `m`.
        # If `cac` is True, `m` is actually a full spectrogram and `z` is ignored.
        niters = self.wiener_iters
        if self.cac:
            B, S, C, Fr, T = m.shape
            out = m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
            out = torch.view_as_complex(out.contiguous())
            return out
        if self.training:
            niters = self.end_iters
        if niters < 0:
            z = z[:, None]
            return z / (1e-8 + z.abs()) * m
        else:
            return self._wiener(m, z, niters)

    def _wiener(self, mag_out, mix_stft, niters):
        # apply wiener filtering from OpenUnmix.
        init = mix_stft.dtype
        wiener_win_len = 300
        residual = self.wiener_residual

        B, S, C, Fq, T = mag_out.shape
        mag_out = mag_out.permute(0, 4, 3, 2, 1)
        mix_stft = torch.view_as_real(mix_stft.permute(0, 3, 2, 1))

        outs = []
        for sample in range(B):
            pos = 0
            out = []
            for pos in range(0, T, wiener_win_len):
                frame = slice(pos, pos + wiener_win_len)
                z_out = wiener(
                    mag_out[sample, frame],
                    mix_stft[sample, frame],
                    niters,
                    residual=residual,
                )
                out.append(z_out.transpose(-1, -2))
            outs.append(torch.cat(out, dim=0))
        out = torch.view_as_complex(torch.stack(outs, 0))
        out = out.permute(0, 4, 3, 2, 1).contiguous()
        if residual:
            out = out[:, :-1]
        assert list(out.shape) == [B, S, C, Fq, T]
        return out.to(init)

    def valid_length(self, length: int):
        """
        Return a length that is appropriate for evaluation.
        In our case, always return the training length, unless
        it is smaller than the given length, in which case this
        raises an error.
        """
        if not self.use_train_segment:
            return length
        training_length = int(self.segment * self.samplerate)
        if training_length < length:
            raise ValueError(
                    f"Given length {length} is longer than "
                    f"training length {training_length}")
        return training_length

    def forward(self, mix):
        length = mix.shape[-1]
        length_pre_pad = None
        if self.use_train_segment:
            if self.training:
                self.segment = Fraction(mix.shape[-1], self.samplerate)
            else:
                training_length = int(self.segment * self.samplerate)
                if mix.shape[-1] < training_length:
                    length_pre_pad = mix.shape[-1]
                    mix = F.pad(mix, (0, training_length - length_pre_pad))
        z = self._spec(mix)
        mag = self._magnitude(z).to(mix.device)
        x = mag

        B, C, Fq, T = x.shape

        # unlike previous Demucs, we always normalize because it is easier.
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        # x will be the freq. branch input.

        # Prepare the time branch input.
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        # okay, this is a giant mess I know...
        saved = []  # skip connections, freq.
        saved_t = []  # skip connections, time.
        lengths = []  # saved lengths to properly remove padding, freq branch.
        lengths_t = []  # saved lengths for time branch.
        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(self.tencoder):
                # we have not yet merged branches.
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    # save for skip connection
                    saved_t.append(xt)
                else:
                    # tenc contains just the first conv., so that now time and freq.
                    # branches have the same shape and can be merged.
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                # add frequency embedding to allow for non equivariant convolutions
                # over the frequency axis.
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb

            saved.append(x)
        if self.crosstransformer:
            if self.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.channel_upsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.channel_upsampler_t(xt)

            x, xt = self.crosstransformer(x, xt)

            if self.bottom_channels:
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.channel_downsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.channel_downsampler_t(xt)

        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            # `pre` contains the output just before final transposed convolution,
            # which is used when the freq. and time branch separate.

            offset = self.depth - len(self.tdecoder)
            if idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    assert pre.shape[2] == 1, pre.shape
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        # Let's make sure we used all stored skip connections.
        assert len(saved) == 0
        assert len(lengths_t) == 0
        assert len(saved_t) == 0

        S = len(self.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        # to cpu as mps doesnt support complex numbers
        # demucs issue #435 ##432
        # NOTE: in this case z already is on cpu
        # TODO: remove this when mps supports complex numbers
        x_is_mps_xpu = x.device.type in ["mps", "xpu"]
        x_device = x.device
        if x_is_mps_xpu:
            x = x.cpu()

        zout = self._mask(z, x)
        if self.use_train_segment:
            if self.training:
                x = self._ispec(zout, length)
            else:
                x = self._ispec(zout, training_length)
        else:
            x = self._ispec(zout, length)

        # back to mps device
        if x_is_mps_xpu:
            x = x.to(x_device)

        if self.use_train_segment:
            if self.training:
                xt = xt.view(B, S, -1, length)
            else:
                xt = xt.view(B, S, -1, training_length)
        else:
            xt = xt.view(B, S, -1, length)
        xt = xt * stdt[:, None] + meant[:, None]
        x = xt + x
        if length_pre_pad:
            x = x[..., :length_pre_pad]
        return x


class DemucsBagOfModel(DemucsPreTrainedModel):
    def __init__(self, config: DemucsConfig, *inputs, **kwargs):
        from iantirta.models.vendor.huggingface_hub import hf_hub_download

        models = [
            load_safetensors_model(
                hf_hub_download(config.name_or_path, f"{sig}.safetensors"),
                config=config,
            )
            for sig in config.bag_models
        ]

        super().__init__(config, *inputs, **kwargs)

        assert len(models) > 0
        first = models[0]
        for other in models:
            assert other.sources == first.sources
            assert other.samplerate == first.samplerate
            assert other.audio_channels == first.audio_channels
            if config.segment is not None:  # noqa: SIM102
                if not isinstance(other, DemucsHTDemucs) or config.segment <= other.segment:
                    other.segment = config.segment

        self.audio_channels = first.audio_channels
        self.samplerate = first.samplerate
        self.sources = first.sources
        self.models = nn.ModuleList(models)

        if config.weights is None:
            config.weights = [[1. for _ in first.sources] for _ in models]
        else:
            assert len(config.weights) == len(models)
            for weight in config.weights:
                assert len(weight) == len(first.sources)
        self.weights = config.weights

    @property
    def max_allowed_segment(self) -> float:
        max_allowed_segment = float('inf')
        for model in self.models:
            if isinstance(model, DemucsHTDemucs):
                max_allowed_segment = min(max_allowed_segment, float(model.segment))
        return max_allowed_segment

    def forward(self, x):
        raise NotImplementedError("Call `apply_model` on this.")


def _decode_json(value):
    """Decode the json encoding of the model init arguments, in particular
    fractions (e.g. the `segment` param of HTDemucs)."""
    if isinstance(value, dict):
        if value.get("_type") == "fraction":
            return Fraction(value["numerator"], value["denominator"])
        return {key: _decode_json(item) for key, item in value.items()}
    elif isinstance(value, list):
        return [_decode_json(item) for item in value]
    return value


def _unflatten_state(tensors: dict[str, Any], structure):
    """Rebuild a nested model state (e.g. diffq packed states) from the flat
    safetensors tensor dict and the json structure stored in its metadata.
    Inverse of `_flatten_state` in `tools/export_hf.py`."""
    def unflatten(node):
        if isinstance(node, dict):
            if "_tensor" in node:
                return tensors[node["_tensor"]]
            elif "_dict" in node:
                return {key: unflatten(item) for key, item in node["_dict"]}
            elif "_list" in node:
                return [unflatten(item) for item in node["_list"]]
            elif "_tuple" in node:
                return tuple(unflatten(item) for item in node["_tuple"])
            elif "_class" in node:
                module, name = node["_class"].rsplit(".", 1)
                return getattr(importlib.import_module(module), name)
            else:
                raise ValueError(f"Invalid structure node {node}.")
        return node
    return unflatten(structure)


DEMUCS_MODEL_REGISTRY = {
    "Demucs": DemucsDemucs,
    "HDemucs": DemucsHDemucs,
    "HTDemucs": DemucsHTDemucs,
}


def load_safetensors_model(path: str | Path, config) -> DemucsModel:
    """Load a single model from a safetensors file, with the model class and init
    arguments stored as json in the safetensors metadata."""
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as file:
        metadata = file.metadata()
        tensors = {
            key: file.get_tensor(key)
            for key in file.keys()
        }
    state: Any
    if 'structure' in metadata:
        state = _unflatten_state(
            tensors,
            json.loads(metadata['structure'])
        )
    else:
        state = tensors
    module, name = metadata['klass'].rsplit(".", 1)
    if module.startswith("demucs."):
        module = module.removeprefix("demucs.")
    klass = DEMUCS_MODEL_REGISTRY[name]
    args = _decode_json(json.loads(metadata['args']))
    args.insert(0, config)
    kwargs = _decode_json(json.loads(metadata['kwargs']))
    return load_model({'klass': klass, 'args': args, 'kwargs': kwargs, 'state': state}, strict=True)


__all__ = [
    "DemucsBagOfModel",
    "DemucsDemucs",
    "DemucsHDemucs",
    "DemucsHTDemucs",
    "DemucsModel",
    "DemucsPreTrainedModel",
]
