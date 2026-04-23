from __future__ import annotations

from instinct_rl.algorithms.wasabi import WasabiPPO


class WasabiMemoryAttentionPPO(WasabiPPO):
    """WasabiPPO + memory-attention auxiliary loss from actor-critic encoders."""

    def compute_losses(self, minibatch):
        losses, inter_vars, stats = super().compute_losses(minibatch)

        if hasattr(self.actor_critic, "get_memory_consistency_loss"):
            memory_loss, actor_memory_loss, critic_memory_loss = (
                self.actor_critic.get_memory_consistency_loss()
            )
            losses["memory_consistency_mse"] = memory_loss
            stats["memory_consistency_mse_actor"] = actor_memory_loss.detach()
            stats["memory_consistency_mse_critic"] = critic_memory_loss.detach()

        return losses, inter_vars, stats
