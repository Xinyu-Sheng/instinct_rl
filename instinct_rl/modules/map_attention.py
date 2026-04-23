import math
import os
from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from instinct_rl.utils.utils import (
    get_obs_slice,
    get_subobs_by_components,
    get_subobs_size,
)


class MapAttentionBlock(nn.Module):
    """Single map-attention block.

    - Extracts the latest depth frame from a map-like observation (e.g. `depth_image` with history)
    - Runs two Conv2d layers on the z-channel, flattens to LW x (d-3)
    - Concatenates per-point coords (x,y,z) to obtain LW x d tokens
    - Projects proprioceptive inputs to a query vector (1 x d)
    - Runs cross-attention (query from proprio, KV from tokens) and returns map_encoding (1 x d)
    - Also returns proprio embedding (1 x d) for actor input if requested by encoder
    """

    def __init__(
        self,
        input_segments: Dict[str, tuple],
        map_component_names: List[str],
        proprio_component_names: List[str] | None = None,
        d: int = 64,
        num_heads: int = 16,
        conv_channels: List[int] | None = None,
        kernel_sizes: List[int] | None = None,
        paddings: List[int] | None = None,
        nonlinearity: str = "ReLU",
    ) -> None:
        super().__init__()
        assert (
            len(map_component_names) == 1
        ), "MapAttentionBlock currently supports a single map component"
        self.input_segments = input_segments
        self.map_comp = map_component_names[0]
        self.proprio_comp_names = proprio_component_names
        self.d = d
        self.num_heads = num_heads

        if conv_channels is None:
            conv_channels = [16, d - 3]
        if kernel_sizes is None:
            kernel_sizes = [5, 5]
        if paddings is None:
            paddings = [(k - 1) // 2 for k in kernel_sizes]

        assert (
            conv_channels[-1] == d - 3
        ), "Last conv out channels must be d-3 to concat 3 coords"

        act_cls = (
            getattr(nn, nonlinearity) if isinstance(nonlinearity, str) else nonlinearity
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                1, conv_channels[0], kernel_size=kernel_sizes[0], padding=paddings[0]
            ),
            act_cls(),
            nn.Conv2d(
                conv_channels[0],
                conv_channels[1],
                kernel_size=kernel_sizes[1],
                padding=paddings[1],
            ),
            act_cls(),
        )

        # compute proprio input size from input_segments if provided
        if self.proprio_comp_names is None:
            # default: use all components except the map component
            self.proprio_comp_names = [
                k for k in input_segments.keys() if k != self.map_comp
            ]

        self.proprio_dim = get_subobs_size(
            {k: v for k, v in input_segments.items()},
            component_names=self.proprio_comp_names,
        )

        self.proprio_proj = nn.Sequential(nn.Linear(self.proprio_dim, d), act_cls())

        # cross-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=num_heads, batch_first=True
        )

        # prepare positional coords (will be created lazily in forward)
        self.register_buffer("_cached_grid", torch.tensor([0.0]))

    def _build_coords(self, device, H: int, W: int, z_values: torch.Tensor):
        # z_values: (B, H, W)
        # create normalized grid x,y in [-1,1]
        ys = torch.linspace(-1.0, 1.0, H, device=device)
        xs = torch.linspace(-1.0, 1.0, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        # (H, W) -> expand to (B, H, W)
        grid_x = grid_x.unsqueeze(0).expand(z_values.shape[0], -1, -1)
        grid_y = grid_y.unsqueeze(0).expand(z_values.shape[0], -1, -1)
        coords = torch.stack([grid_x, grid_y, z_values], dim=-1)  # (B, H, W, 3)
        return coords

    def forward(self, flat_input: torch.Tensor, return_attn_weights: bool = False):
        # flat_input: (..., obs_dim)
        leading_dim = flat_input.shape[:-1]
        B = int(np.prod(leading_dim)) if len(leading_dim) > 0 else 1

        # extract map component and reshape
        obs_slice, obs_shape = get_obs_slice(self.input_segments, self.map_comp)
        map_tensor = flat_input[..., obs_slice].reshape(
            *flat_input.shape[:-1], *obs_shape
        )
        # map_tensor shape: (B, num_frames, H, W) typically
        if map_tensor.ndim == 4:
            # (B, num_frames, H, W) -> take last frame
            z = map_tensor[..., -1, :, :]
        elif map_tensor.ndim == 3:
            # (B, H, W)
            z = map_tensor
        else:
            raise RuntimeError(f"Unsupported map tensor shape: {map_tensor.shape}")

        # ensure contiguous batch dim
        batch_size = z.shape[0]
        H = z.shape[-2]
        W = z.shape[-1]

        x = z.reshape(batch_size, 1, H, W)  # (B,1,H,W)
        conv_out = self.conv(x)  # (B, d-3, H, W)
        conv_out = conv_out.permute(0, 2, 3, 1).reshape(
            batch_size, H * W, -1
        )  # (B, LW, d-3)

        coords = self._build_coords(z.device, H, W, z)  # (B,H,W,3)
        coords = coords.reshape(batch_size, H * W, 3)

        tokens = torch.cat([conv_out, coords], dim=-1)  # (B, LW, d)

        # proprio
        proprio_vec = get_subobs_by_components(
            flat_input, self.proprio_comp_names, self.input_segments, cat=True
        )
        proprio_emb = self.proprio_proj(proprio_vec)

        # attention: query shape (B, 1, d), key/value shape (B, LW, d)
        q = proprio_emb.unsqueeze(1)
        attn_out, attn_weights = self.attn(q, tokens, tokens)

        map_encoding = attn_out.squeeze(1)  # (B, d)
        if return_attn_weights:
            # attn_weights from torch.nn.MultiheadAttention is averaged over heads by default
            attn_weights = attn_weights.squeeze(1).reshape(batch_size, H, W)
            return map_encoding, proprio_emb, attn_weights

        return map_encoding, proprio_emb


class MapAttentionEncoder(nn.Module):
    """Top-level encoder compatible with EncoderActorCriticMixin.

    Expects constructor signature: (input_segments, block_configs, sequential_idx=0)
    and produces `output_segment` attribute similar to ParallelLayer.
    """

    def __init__(
        self,
        input_segments: Dict[str, tuple],
        block_configs: Dict[str, dict],
        sequential_idx: int = 0,
    ):
        super().__init__()
        self.input_segments = deepcopy(input_segments)
        self.block_configs = deepcopy(block_configs)
        self._sequential_idx = sequential_idx
        self._output_component_name_prefix = f"parallel_latent_{self._sequential_idx}_"

        # build blocks
        self._blocks = nn.ModuleDict()
        for block_name, cfg in self.block_configs.items():
            # skip meta key if present
            if block_name == "class_name":
                continue
            # cfg may be a dict, a dataclass instance, or a shorthand (str/list)
            if isinstance(cfg, dict):
                cfg_dict = deepcopy(cfg)
            elif hasattr(cfg, "__dict__"):
                cfg_dict = deepcopy(cfg.__dict__)
            elif isinstance(cfg, str):
                cfg_dict = {"component_names": [cfg]}
            elif isinstance(cfg, (list, tuple)):
                cfg_dict = {"component_names": list(cfg)}
            else:
                raise TypeError(
                    f"Unsupported block config type for '{block_name}': {type(cfg)}"
                )

            map_comp_names = cfg_dict.get("component_names")
            if map_comp_names is None:
                # fallback to block_name as single component
                map_comp_names = [block_name]
            proprio_comp_names = cfg_dict.get("proprio_component_names", None)
            d = cfg_dict.get("d", cfg_dict.get("output_size", 64))
            num_heads = cfg_dict.get("num_heads", 16)
            conv_channels = cfg_dict.get("conv_channels", [16, d - 3])
            kernel_sizes = cfg_dict.get("kernel_sizes", [5, 5])
            paddings = cfg_dict.get("paddings", [(k - 1) // 2 for k in kernel_sizes])
            nonlinearity = cfg_dict.get("nonlinearity", "ReLU")

            block = MapAttentionBlock(
                input_segments=self.input_segments,
                map_component_names=map_comp_names,
                proprio_component_names=proprio_comp_names,
                d=d,
                num_heads=num_heads,
                conv_channels=conv_channels,
                kernel_sizes=kernel_sizes,
                paddings=paddings,
                nonlinearity=nonlinearity,
            )
            self._blocks[block_name] = block

        self.build_output_segment()

    def build_output_segment(self):
        self.output_segment = deepcopy(self.input_segments)
        for block_name, cfg in self.block_configs.items():
            # skip meta key if present
            if block_name == "class_name":
                continue
            if isinstance(cfg, dict):
                cfg_dict = deepcopy(cfg)
            elif hasattr(cfg, "__dict__"):
                cfg_dict = deepcopy(cfg.__dict__)
            elif isinstance(cfg, str):
                cfg_dict = {"component_names": [cfg]}
            elif isinstance(cfg, (list, tuple)):
                cfg_dict = {"component_names": list(cfg)}
            else:
                raise TypeError(
                    f"Unsupported block config type for '{block_name}': {type(cfg)}"
                )

            d = cfg_dict.get("d", cfg_dict.get("output_size", 64))
            embed_proprio = cfg_dict.get("embed_proprio", True)
            self.output_segment[self._output_component_name_prefix + block_name] = (d,)
            if embed_proprio:
                self.output_segment[
                    self._output_component_name_prefix + block_name + "_proprio"
                ] = (d,)
            if cfg_dict.get("takeout_input_components", True):
                comp_names = list(cfg_dict.get("component_names", [])) + list(
                    cfg_dict.get("proprio_component_names", []) or []
                )
                for comp in comp_names:
                    if comp in self.output_segment:
                        del self.output_segment[comp]
        self.numel_output = get_subobs_size(self.output_segment)
        return self.output_segment

    def forward(
        self,
        flat_input: torch.Tensor,
        return_attn_weights: bool = False,
    ) -> torch.Tensor:
        leading_dim = flat_input.shape[:-1]
        blocks_outputs = {}
        blocks_attn = {}
        for block_name, block in self._blocks.items():
            if return_attn_weights:
                map_enc, proprio_emb, attn_weights = block(
                    flat_input, return_attn_weights=True
                )
                blocks_attn[block_name] = attn_weights
            else:
                map_enc, proprio_emb = block(flat_input)
            blocks_outputs[self._output_component_name_prefix + block_name] = (
                map_enc.reshape(*leading_dim, -1)
            )
            blocks_outputs[
                self._output_component_name_prefix + block_name + "_proprio"
            ] = proprio_emb.reshape(*leading_dim, -1)

        outputs = []
        for output_component_name, output_shape in self.output_segment.items():
            if output_component_name.startswith(self._output_component_name_prefix):
                outputs.append(
                    blocks_outputs[output_component_name].reshape(*leading_dim, -1)
                )
            else:
                outputs.append(
                    get_subobs_by_components(
                        flat_input, [output_component_name], self.input_segments
                    ).reshape(*leading_dim, -1)
                )
        encoded = torch.cat(outputs, dim=-1)
        if return_attn_weights:
            return encoded, blocks_attn
        return encoded

    def __str__(self):
        return f"MapAttentionEncoder(blocks={list(self._blocks.keys())})"

    def export_as_onnx(
        self,
        flat_input: torch.Tensor,
        filedir: str,
        block_as_seperate_files: bool = True,
    ):
        """Export encoder as ONNX.

        The attention encoder consumes the full flattened policy observation and emits
        encoded features for the actor network. For compatibility with the existing
        split-export path, the default output name follows the same ``<idx>-<block>``
        pattern when a single block is used.
        """
        self.eval()
        with torch.no_grad():
            if block_as_seperate_files:
                if len(self._blocks) != 1:
                    raise NotImplementedError(
                        "MapAttentionEncoder split export currently supports exactly one block."
                    )
                block_name = next(iter(self._blocks.keys()))
                save_path = os.path.join(
                    filedir, f"{self._sequential_idx}-{block_name}.onnx"
                )
            else:
                save_path = os.path.join(filedir, "map_attention_encoder.onnx")

            exported_program = torch.onnx.export(
                self,
                flat_input,
                "/tmp/map_attention_encoder.onnx",
                input_names=["input"],
                output_names=["output"],
                dynamo=True,
                opset_version=18,
            )
            exported_program.save(save_path)
            print(f"Exported map_attention encoder to {save_path}")
