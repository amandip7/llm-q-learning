"""
Q-Network implementation for LLM post-training.
Uses Double DQN with the LLM's logits as Q-values.

Key Insight:
- States (s) = sequence of tokens (prefix)
- Actions (a) = next token from vocabulary
- Q(s, a) = logit for token a given prefix s
- Next state s' = s + a (prefix concatenated with action)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Tuple, Optional
import copy


class QLearningLLM(nn.Module):
    """
    Wrapper around a causal LM that treats logits as Q-values.
    Implements Double DQN with online and target networks.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda",
        tau: float = 0.005
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
        
        # Freeze target network
        for param in self.target_network.parameters():
            param.requires_grad = False
        
        self.vocab_size = self.online_network.config.vocab_size
    
    def get_q_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_target: bool = False
    ) -> torch.Tensor:
        """
        Get Q-values (logits) for all possible next tokens.
        
        Args:
            input_ids: [batch_size, seq_length]
            attention_mask: [batch_size, seq_length]
            use_target: Whether to use target network
            
        Returns:
            q_values: [batch_size, seq_length, vocab_size]
                     Q(s_t, a) for each position t and action a
        """
        network = self.target_network if use_target else self.online_network
        
        with torch.set_grad_enabled(not use_target):
            outputs = network(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            # Logits are our Q-values: Q(s_t, a) for all actions a
            return outputs.logits
    
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
        """Soft update target network: θ_target = τ*θ_online + (1-τ)*θ_target"""
        for target_param, online_param in zip(
            self.target_network.parameters(),
            self.online_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def hard_update_target(self):
        """Hard update: copy online network to target network."""
        self.target_network.load_state_dict(self.online_network.state_dict())

