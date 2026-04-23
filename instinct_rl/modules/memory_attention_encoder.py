from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from instinct_rl.modules.conv2d import Conv2dHeadModel
from instinct_rl.utils import unpad_trajectories
from instinct_rl.utils.utils import (
    get_obs_slice,
    get_subobs_by_components,
    get_subobs_size,
)


class MemoryAttentionEncoder(nn.Module):
    """Encoder for parkour policy/value that mixes depth latent, memory and terrain attention.

    Output order is fixed to:
    1) memory latent [128]
    2) attention latent [8]
    3) depth latent [128]
    4) all non-depth raw observations
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

        self.hidden_states: torch.Tensor | None = None
        self._last_memory_consistency_mse: torch.Tensor | None = None

        self._build_from_cfg()
        self.build_output_segment()

    def _cfg_to_dict(self, cfg):
        if isinstance(cfg, dict):
            return deepcopy(cfg)
        if hasattr(cfg, "__dict__"):
            cfg_dict = {
                k: v
                for k, v in deepcopy(cfg.__dict__).items()
                if not k.startswith("__") and not callable(v)
            }
            return cfg_dict
        raise TypeError(f"Unsupported block config type: {type(cfg)}")

    def _build_from_cfg(self):
        candidate_blocks = [
            (name, cfg)
            for name, cfg in self.block_configs.items()
            if name != "class_name"
        ]
        if len(candidate_blocks) != 1:
            raise ValueError(
                "MemoryAttentionEncoder expects exactly one block config (e.g. depth_encoder), "
                f"but got {len(candidate_blocks)}"
            )

        self.depth_block_name, cfg = candidate_blocks[0]
        cfg = self._cfg_to_dict(cfg)

        self.depth_component_names = cfg.get("component_names", ["depth_image"])
        if len(self.depth_component_names) != 1:
            raise ValueError(
                "MemoryAttentionEncoder currently supports exactly one depth component."
            )
        self.depth_component_name = self.depth_component_names[0]

        self.depth_latent_name = (
            self._output_component_name_prefix + self.depth_block_name
        )
        self.memory_latent_name = self._output_component_name_prefix + "memory_t"
        self.attention_latent_name = self._output_component_name_prefix + "attention"

        self.depth_obs_slice, self.depth_obs_shape = get_obs_slice(
            self.input_segments, self.depth_component_name
        )
        if len(self.depth_obs_shape) != 3:
            raise ValueError(
                f"Expected depth observation shape [frames,H,W], got {self.depth_obs_shape}"
            )

        # Depth encoder: Conv2dHeadModel with output latent 128 by default.
        self.depth_latent_size = int(cfg.get("output_size", 128))
        depth_hidden_sizes = list(cfg.get("hidden_sizes", [])) + [
            self.depth_latent_size
        ]
        self.depth_encoder = Conv2dHeadModel(
            image_shape=self.depth_obs_shape,
            channels=cfg["channels"],
            kernel_sizes=cfg["kernel_sizes"],
            strides=cfg["strides"],
            hidden_sizes=depth_hidden_sizes,
            output_size=None,
            paddings=cfg.get("paddings", None),
            nonlinearity=cfg.get("nonlinearity", "ReLU"),
            use_maxpool=cfg.get("use_maxpool", False),
        )

        self.non_depth_component_names = [
            name
            for name in self.input_segments.keys()
            if name != self.depth_component_name
        ]
        self.non_depth_size = get_subobs_size(
            self.input_segments,
            component_names=self.non_depth_component_names,
        )

        self.memory_hidden_size = int(cfg.get("memory_hidden_size", 128))
        self.memory_state_proj_size = int(cfg.get("memory_state_proj_size", 128))
        self.non_depth_to_memory = nn.Linear(
            self.non_depth_size, self.memory_state_proj_size
        )
        self.memory_gru = nn.GRUCell(
            input_size=self.memory_state_proj_size + self.depth_latent_size,
            hidden_size=self.memory_hidden_size,
        )
        self.memory_target_head = nn.Sequential(
            nn.Linear(self.memory_hidden_size, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, self.depth_latent_size),
        )

        self.attention_embed_dim = int(cfg.get("attention_embed_dim", 8))
        self.attention_num_heads = int(cfg.get("attention_num_heads", 2))
        if self.depth_obs_shape[0] != self.attention_embed_dim:
            raise ValueError(
                "Attention embed dim must match depth frame count. "
                f"Got depth frames={self.depth_obs_shape[0]}, attention_embed_dim={self.attention_embed_dim}."
            )
        self.non_depth_to_query = nn.Linear(
            self.non_depth_size, self.attention_embed_dim
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=self.attention_embed_dim,
            num_heads=self.attention_num_heads,
            batch_first=True,
        )

    def _reshape_depth(self, flat_input: torch.Tensor):
        leading_dims = flat_input.shape[:-1]
        batch_size = int(np.prod(leading_dims)) if len(leading_dims) > 0 else 1
        depth = flat_input[..., self.depth_obs_slice].reshape(
            *leading_dims, *self.depth_obs_shape
        )
        depth = depth.reshape(batch_size, *self.depth_obs_shape)
        return depth, leading_dims, batch_size

    def _run_memory_gru(
        self,
        gru_input: torch.Tensor,
        leading_dims,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # In batch mode, hidden_states is from rollout storage and should not modify module state.
        batch_mode = hidden_states is not None

        if len(leading_dims) == 1:
            if batch_mode:
                h = hidden_states[0] if hidden_states.dim() == 3 else hidden_states
            else:
                batch_size = leading_dims[0]
                if (
                    self.hidden_states is None
                    or self.hidden_states.shape[1] != batch_size
                    or self.hidden_states.device != gru_input.device
                ):
                    self.hidden_states = torch.zeros(
                        1,
                        batch_size,
                        self.memory_hidden_size,
                        device=gru_input.device,
                        dtype=gru_input.dtype,
                    )
                h = self.hidden_states[0]

            h = self.memory_gru(gru_input, h)

            if not batch_mode:
                self.hidden_states = h.unsqueeze(0)
            return h

        if len(leading_dims) != 2:
            raise RuntimeError(
                f"MemoryAttentionEncoder expects obs with 1 or 2 leading dims, got shape {gru_input.shape}"
            )

        time_dim, batch_dim = leading_dims
        gru_input = gru_input.reshape(time_dim, batch_dim, -1)

        if batch_mode:
            h = hidden_states[0] if hidden_states.dim() == 3 else hidden_states
        else:
            if (
                self.hidden_states is None
                or self.hidden_states.shape[1] != batch_dim
                or self.hidden_states.device != gru_input.device
            ):
                self.hidden_states = torch.zeros(
                    1,
                    batch_dim,
                    self.memory_hidden_size,
                    device=gru_input.device,
                    dtype=gru_input.dtype,
                )
            h = self.hidden_states[0]

        out_seq = []
        for t in range(time_dim):
            h = self.memory_gru(gru_input[t], h)
            out_seq.append(h)
        out_seq = torch.stack(out_seq, dim=0)

        if not batch_mode:
            self.hidden_states = h.unsqueeze(0)
        return out_seq

    def build_output_segment(self):
        # Output order is fixed by design request.
        output_segment = OrderedDict()
        output_segment[self.memory_latent_name] = (self.memory_hidden_size,)
        output_segment[self.attention_latent_name] = (self.attention_embed_dim,)
        output_segment[self.depth_latent_name] = (self.depth_latent_size,)
        for name, shape in self.input_segments.items():
            if name != self.depth_component_name:
                output_segment[name] = shape
        self.output_segment = output_segment
        self.numel_output = get_subobs_size(self.output_segment)
        return self.output_segment

    def forward(
        self,
        flat_input: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        depth_obs, leading_dims, flat_batch = self._reshape_depth(flat_input)

        # [N, 128]
        depth_latent = self.depth_encoder(depth_obs)

        # [*, non_depth_size]
        non_depth_raw = get_subobs_by_components(
            flat_input,
            self.non_depth_component_names,
            self.input_segments,
        )
        non_depth_raw_batch = non_depth_raw.reshape(flat_batch, -1)

        # Memory branch: [N, 128] + detach(depth latent) -> [N, 256] -> GRU hidden [N, 128]
        state_proj = self.non_depth_to_memory(non_depth_raw_batch)
        memory_input = torch.cat([state_proj, depth_latent.detach()], dim=-1)
        memory_t = self._run_memory_gru(
            memory_input,
            leading_dims,
            hidden_states=hidden_states,
        )

        if len(leading_dims) == 1:
            depth_latent = depth_latent.reshape(leading_dims[0], -1)
            non_depth_raw = non_depth_raw.reshape(leading_dims[0], -1)
        else:
            depth_latent = depth_latent.reshape(*leading_dims, -1)
            non_depth_raw = non_depth_raw.reshape(*leading_dims, -1)

        # Attention branch: depth [N,8,18,32] -> [N,18*32,8], Q from non-depth [N,8]
        kv = depth_obs.permute(0, 2, 3, 1).reshape(
            flat_batch, -1, self.attention_embed_dim
        )
        query = self.non_depth_to_query(non_depth_raw_batch).unsqueeze(1)
        attn_out, _ = self.attention(query, kv, kv)
        attn_out = attn_out.squeeze(1)
        if len(leading_dims) == 1:
            attn_out = attn_out.reshape(leading_dims[0], -1)
        else:
            attn_out = attn_out.reshape(*leading_dims, -1)

        # Consistency loss: norm(MLP(memory_t)) vs norm(detach(depth_latent))
        memory_target = self.memory_target_head(
            memory_t.reshape(-1, self.memory_hidden_size)
        )
        memory_target = memory_target.reshape(
            *memory_t.shape[:-1], self.depth_latent_size
        )

        if masks is not None and len(memory_target.shape) == 3:
            memory_target_for_loss = unpad_trajectories(memory_target, masks)
            depth_for_loss = unpad_trajectories(depth_latent, masks)
        else:
            memory_target_for_loss = memory_target
            depth_for_loss = depth_latent

        self._last_memory_consistency_mse = F.mse_loss(
            F.normalize(memory_target_for_loss, dim=-1),
            F.normalize(depth_for_loss.detach(), dim=-1),
        )

        encoded = torch.cat([memory_t, attn_out, depth_latent, non_depth_raw], dim=-1)
        if masks is not None and len(encoded.shape) == 3:
            encoded = unpad_trajectories(encoded, masks)
        return encoded

    def get_hidden_states(self):
        return self.hidden_states

    def get_memory_consistency_loss(self):
        if self._last_memory_consistency_mse is not None:
            return self._last_memory_consistency_mse
        device = next(self.parameters()).device
        return torch.zeros((), device=device)

    def reset(self, dones: torch.Tensor | None = None):
        if self.hidden_states is None:
            return
        if dones is None:
            self.hidden_states.zero_()
            return
        dones = dones.to(torch.bool)
        self.hidden_states[:, dones, :] = 0.0

    def __str__(self):
        return (
            "MemoryAttentionEncoder("
            f"depth={self.depth_component_name}, "
            f"memory={self.memory_hidden_size}, "
            f"attn_d={self.attention_embed_dim}, heads={self.attention_num_heads})"
        )

    def export_as_onnx(
        self, flat_input, filedir: str, block_as_seperate_files: bool = True
    ):
        raise NotImplementedError(
            "ONNX export for MemoryAttentionEncoder is not implemented yet."
        )
