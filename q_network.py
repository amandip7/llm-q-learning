"""
Q-Network implementation for LLM post-training.
Uses Double DQN with a learned Q-value head on top of the LLM.

Key Insight:
- States (s) = sequence of tokens (prefix)
- Actions (a) = next token from vocabulary
- Q(s, a) = learned projection from LLM hidden states to Q-values
- Next state s' = s + a (prefix concatenated with action)

Architecture:
- LLM produces hidden states (hidden_size)
- Q-value projection layer: Linear(hidden_size -> vocab_size)
- Initialized with small random weights (std=0.01) for near-zero initial Q-values
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Tuple, Optional
import copy


class QValueHead(nn.Module):
    """
    Learnable projection layer that maps LLM hidden states to Q-values.
    Initialized with small random weights so initial Q-values are near zero,
    analogous to standard Q-function initialization in RL.
    """

    def __init__(self, hidden_size: int, vocab_size: int, init_std: float = 0.01):
        super().__init__()
        self.projection = nn.Linear(hidden_size, vocab_size)

        # Initialize with small random weights for near-zero initial Q-values
        # (but not exactly zero to preserve gradient flow)
        nn.init.normal_(self.projection.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.projection.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project LLM hidden states to Q-values.

        Args:
            hidden_states: [batch_size, seq_length, hidden_size] from LLM

        Returns:
            q_values: [batch_size, seq_length, vocab_size]
        """
        return self.projection(hidden_states)


class QLearningLLM(nn.Module):
    """
    Wrapper around a causal LM with a Q-value projection head.
    Implements Double DQN with online and target networks.

    Architecture: LLM → logits → QValueHead → Q-values
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda",
        tau: float = 0.005,
        q_head_init_std: float = 0.01
    ):
        super().__init__()

        self.device = device if torch.cuda.is_available() else "cpu"
        self.tau = tau

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Online network (the one we train)
        self.online_network = AutoModelForCausalLM.from_pretrained(model_name)
        self.online_network.to(self.device)

        # Target network (frozen, updated slowly)
        self.target_network = AutoModelForCausalLM.from_pretrained(model_name)
        self.target_network.to(self.device)
        self.target_network.eval()

        # Freeze target network LLM parameters
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.vocab_size = self.online_network.config.vocab_size
        self.hidden_size = self.online_network.config.hidden_size

        # Q-value projection heads (one for online, one for target)
        # Maps hidden_size -> vocab_size (much smaller than vocab_size -> vocab_size)
        self.online_q_head = QValueHead(self.hidden_size, self.vocab_size, init_std=q_head_init_std)
        self.online_q_head.to(self.device)

        self.target_q_head = QValueHead(self.hidden_size, self.vocab_size, init_std=q_head_init_std)
        self.target_q_head.to(self.device)
        self.target_q_head.eval()

        # Freeze target Q-head
        for param in self.target_q_head.parameters():
            param.requires_grad = False

        # Initialize target Q-head to match online Q-head
        self.target_q_head.load_state_dict(self.online_q_head.state_dict())
    
    def get_q_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_target: bool = False
    ) -> torch.Tensor:
        """
        Get Q-values for all possible next tokens.

        Architecture: LLM → hidden_states → QValueHead → Q-values

        Args:
            input_ids: [batch_size, seq_length]
            attention_mask: [batch_size, seq_length]
            use_target: Whether to use target network

        Returns:
            q_values: [batch_size, seq_length, vocab_size]
                     Q(s_t, a) for each position t and action a
        """
        network = self.target_network if use_target else self.online_network
        q_head = self.target_q_head if use_target else self.online_q_head

        with torch.set_grad_enabled(not use_target):
            outputs = network(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            # Get final layer hidden states and project through Q-value head
            hidden_states = outputs.hidden_states[-1]  # [batch, seq, hidden_size]
            q_values = q_head(hidden_states)
            return q_values
    
    def get_action_q_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_target: bool = False
    ) -> torch.Tensor:
        """
        Get Q-values for the actions that were actually taken.
        
        For sequence [t1, t2, t3, t4]:
        - Q(s0, t1) from position 0
        - Q(s1, t2) from position 1  
        - etc.
        
        Returns:
            action_q_values: [batch_size, seq_length-1]
        """
        q_values = self.get_q_values(input_ids, attention_mask, use_target)
        
        # Shift: Q-values at position t predict token at position t+1
        q_values = q_values[:, :-1, :]  # [batch, seq-1, vocab]
        actions = input_ids[:, 1:]  # [batch, seq-1]
        
        # Gather Q-values for taken actions
        action_q_values = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        
        return action_q_values
    
    def compute_target_q_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rewards: torch.Tensor,
        gamma: float = 0.99,
        terminal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute target Q-values using Double DQN.
        
        Double DQN formula:
        Q_target(s, a) = r + γ * Q_target(s', argmax_a' Q_online(s', a'))
        
        Args:
            input_ids: [batch_size, seq_length]
            attention_mask: [batch_size, seq_length]  
            rewards: [batch_size, seq_length] token-level rewards
            gamma: Discount factor
            terminal_mask: [batch_size, seq_length] 1 for terminal states
        """
        with torch.no_grad():
            # Get Q-values from both networks
            online_q = self.get_q_values(input_ids, attention_mask, use_target=False)
            target_q = self.get_q_values(input_ids, attention_mask, use_target=True)
            
            # Double DQN: use online network to select actions
            # Use target network to evaluate them
            best_actions = online_q.argmax(dim=-1)  # [batch, seq]
            
            # Get target Q-values for best actions
            next_q_values = target_q.gather(2, best_actions.unsqueeze(-1)).squeeze(-1)
            
            # Shift for next state values
            # Q(s_t, a_t) = r_t + gamma * Q(s_{t+1}, a*)
            next_q_values = next_q_values[:, 1:]  # [batch, seq-1]
            current_rewards = rewards[:, :-1]  # [batch, seq-1]
            
            # Terminal states: no future reward
            if terminal_mask is not None:
                terminal = terminal_mask[:, :-1]
                next_q_values = next_q_values * (1 - terminal)
            
            # Bellman target
            target_q_values = current_rewards + gamma * next_q_values
            
            return target_q_values
    
    def soft_update_target(self):
        """
        Soft update target network and Q-head:
        θ_target = τ*θ_online + (1-τ)*θ_target
        """
        # Update target LLM
        for target_param, online_param in zip(
            self.target_network.parameters(),
            self.online_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

        # Update target Q-head
        for target_param, online_param in zip(
            self.target_q_head.parameters(),
            self.online_q_head.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

    def hard_update_target(self):
        """Hard update: copy online network and Q-head to target."""
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_q_head.load_state_dict(self.online_q_head.state_dict())

    def get_trainable_parameters(self):
        """
        Get all trainable parameters (LLM + Q-head).
        Use this to set up the optimizer.
        """
        return list(self.online_network.parameters()) + list(self.online_q_head.parameters())

