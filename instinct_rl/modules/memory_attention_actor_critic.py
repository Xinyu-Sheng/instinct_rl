from __future__ import annotations

import torch

from instinct_rl.modules.actor_critic_recurrent import ActorCriticHiddenState
from instinct_rl.modules.all_mixer import EncoderMoEActorCritic
from instinct_rl.modules.moe_actor_critic import MoEActorCritic


class MemoryAttentionEncoderMoEActorCritic(EncoderMoEActorCritic):
    """Recurrent-aware encoder MoE policy for memory-attention encoder blocks.

    This class keeps MoE actor/critic heads and delegates temporal memory state to encoders.
    """

    is_recurrent = True

    def _get_actor_hidden(self, hidden_states):
        if hidden_states is None:
            return None
        if hasattr(hidden_states, "actor"):
            return hidden_states.actor
        return hidden_states

    def _get_critic_hidden(self, hidden_states):
        if hidden_states is None:
            return None
        if hasattr(hidden_states, "critic"):
            return hidden_states.critic
        return hidden_states

    def act(self, observations, masks=None, hidden_states=None):
        actor_hidden_states = self._get_actor_hidden(hidden_states)
        obs = self.encoders(
            observations,
            masks=masks,
            hidden_states=actor_hidden_states,
        )
        return MoEActorCritic.act(self, obs)

    def act_inference(self, observations):
        obs = self.encoders(observations)
        return MoEActorCritic.act_inference(self, obs)

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        critic_hidden_states = self._get_critic_hidden(hidden_states)
        obs = (
            self.critic_encoders(
                critic_observations,
                masks=masks,
                hidden_states=critic_hidden_states,
            )
            if self.critic_encoders is not None
            else critic_observations
        )
        return MoEActorCritic.evaluate(self, obs)

    def backbone_act(self, flatten_observations, **kwargs):
        return MoEActorCritic.act(self, flatten_observations)

    def backbone_evaluate(self, flatten_observations, masks=None, hidden_states=None):
        return MoEActorCritic.evaluate(self, flatten_observations)

    def reset(self, dones=None):
        if hasattr(self.encoders, "reset"):
            self.encoders.reset(dones)
        if self.critic_encoders is None or self.critic_encoders is self.encoders:
            return
        if hasattr(self.critic_encoders, "reset"):
            self.critic_encoders.reset(dones)

    def get_hidden_states(self):
        actor_hidden = (
            self.encoders.get_hidden_states()
            if hasattr(self.encoders, "get_hidden_states")
            else None
        )
        if self.critic_encoders is None:
            critic_hidden = None
        elif self.critic_encoders is self.encoders:
            critic_hidden = actor_hidden
        else:
            critic_hidden = (
                self.critic_encoders.get_hidden_states()
                if hasattr(self.critic_encoders, "get_hidden_states")
                else None
            )
        return ActorCriticHiddenState(actor_hidden, critic_hidden)

    def get_memory_consistency_loss(self):
        device = self.std.device
        actor_loss = (
            self.encoders.get_memory_consistency_loss()
            if hasattr(self.encoders, "get_memory_consistency_loss")
            else torch.zeros((), device=device)
        )

        if self.critic_encoders is None:
            critic_loss = torch.zeros((), device=device)
            total_loss = actor_loss
        elif self.critic_encoders is self.encoders:
            critic_loss = actor_loss
            total_loss = actor_loss
        else:
            critic_loss = (
                self.critic_encoders.get_memory_consistency_loss()
                if hasattr(self.critic_encoders, "get_memory_consistency_loss")
                else torch.zeros((), device=device)
            )
            total_loss = 0.5 * (actor_loss + critic_loss)

        return total_loss, actor_loss, critic_loss
