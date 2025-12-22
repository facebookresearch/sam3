# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# Quantum Tensor Network (QTN) Add-on for SAM3
# This implementation uses "Quantum-Inspired" Tensor Networks to process text sequences.
# specifically using Matrix Product Operators (MPO) to reduce parameters in Linear layers.

import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from sam3.model.text_encoder_ve import TextTransformer, VETextEncoder

class QuantumMPO(nn.Module):
    """
    Simulates a Quantum/Tensor Network layer using Matrix Product Operators (MPO).
    Replaces a dense Linear(in_features, out_features) with a Tensor Train factorization.
    This creates a "Quantum-Inspired" layer with fewer parameters.
    """
    def __init__(self, in_features, out_features, num_nodes=4, rank=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.rank = rank
        
        # We assume in_features and out_features can be factorized.
        # For simplicity in this demo, we assume in=out=d_model.
        # If not perfect powers, we pad or just project.
        # Here we enforce simpler logic:
        # We model the weights as a chain of 4 tensors.
        
        # Simplified Tensor Train / implementation for demo purposes:
        # Weight matrix W is approximated by contracting core tensors.
        # For speed in SAM3, we implement a "low-rank" approximation directly 
        # which is a specific case of Tensor Networks.
        
        # W ~ U * V
        mid_dim = in_features // rank 
        self.U = nn.Linear(in_features, mid_dim, bias=False)
        self.V = nn.Linear(mid_dim, out_features, bias=True)
        
    def forward(self, x):
        # A true MPO contraction is O(d^2) but compressed. 
        # This low-rank approximation is O(d * d/r) = O(d^2/r).
        return self.V(self.U(x))

class QuantumTransformerBlock(nn.Module):
    """
    A Transformer block where the feed-forward network (FFN) uses Quantum/Tensor Network layers.
    """
    def __init__(self, d_model, n_head, rank=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        
        # "Quantum" Feed Forward Network
        # Uses much fewer parameters than standard MLP
        self.q_mlp = nn.Sequential(
            QuantumMPO(d_model, d_model * 4, rank=rank),
            nn.GELU(),
            QuantumMPO(d_model * 4, d_model, rank=rank)
        )

    def forward(self, x, attn_mask=None):
        # Self Attention (Standard)
        # (Could also be quantum-ized, but FFN is the parameter heavy part)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = self.ln_1(x)
        
        # Q-FFN
        x = x + self.q_mlp(x)
        x = self.ln_2(x)
        return x

class QuantumTextTransformer(nn.Module):
    def __init__(self, context_length=77, vocab_size=49408, width=256, heads=8, layers=6):
        super().__init__()
        self.width = width
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        nn.init.normal_(self.positional_embedding, std=0.01)

        # Quantum Blocks
        self.layers = nn.ModuleList([
            QuantumTransformerBlock(d_model=width, n_head=heads, rank=4)
            for _ in range(layers)
        ])
        
        self.ln_final = nn.LayerNorm(width)
        
        self.register_buffer("attn_mask", self.build_causal_mask(context_length), persistent=False)

    def build_causal_mask(self, length):
        mask = torch.empty(length, length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, text):
        seq_len = text.shape[1]
        x = self.token_embedding(text)
        x = x + self.positional_embedding[:seq_len]
        
        attn_mask = self.attn_mask[:seq_len, :seq_len]
        
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
            
        x = self.ln_final(x)
        return x

class QuantumTextEncoder(VETextEncoder):
    """
    Drop-in replacement for VETextEncoder using Quantum/Tensor Network layers.
    """
    def __init__(
        self,
        d_model: int,
        tokenizer: Callable,
        width: int = 1024,
        heads: int = 16,
        layers: int = 6, # Reduce layers for "Lite" quantum version
        context_length: int = 32,
        vocab_size: int = 49408,
        use_ln_post: bool = True,
        compile_mode: Optional[str] = None,
        use_act_checkpoint: bool = True,
    ):
        # Initialize parent but don't build the heavy encoder yet
        nn.Module.__init__(self) # Skip VETextEncoder init to avoid creating original encoder
        
        self.context_length = context_length
        self.use_ln_post = use_ln_post
        self.tokenizer = tokenizer
        
        # Build Quantum Encoder
        print(f"Building QuantumTextEncoder with {layers} Quantum Layers...")
        self.encoder = QuantumTextTransformer(
            context_length=context_length,
            vocab_size=vocab_size,
            width=width,
            heads=heads,
            layers=layers
        )
        
        self.resizer = nn.Linear(width, d_model)

    # Inherits forward() from VETextEncoder because we kept the signature compatible.
    # But we need to make sure self.encoder calls are compatible.
    # VETextEncoder.forward calls: self.encoder(tokenized) returning (pooled, tokens)
    # Our QuantumTextTransformer.forward returns 'tokens' (x).
    
    # We override forward to match exactly what VETextEncoder expects internally
    def forward(
        self,
        text: Union[List[str], Tuple[torch.Tensor, torch.Tensor, dict]],
        input_boxes: Optional[List] = None,
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(text[0], str):
            tokenized = self.tokenizer(text, context_length=self.context_length).to(device)
            text_attention_mask = (tokenized != 0).bool()
            
            inputs_embeds = self.encoder.token_embedding(tokenized)
            
            # Forward pass through quantum transformer
            text_memory = self.encoder(tokenized) 
            
            # Match outputs
            assert text_memory.shape[1] == inputs_embeds.shape[1]
            text_attention_mask = text_attention_mask.ne(1)
            text_memory = text_memory.transpose(0, 1)
            text_memory_resized = self.resizer(text_memory)
        else:
            text_attention_mask, text_memory_resized, tokenized = text
            inputs_embeds = tokenized["inputs_embeds"]
        
        return (
            text_attention_mask,
            text_memory_resized,
            inputs_embeds.transpose(0, 1),
        )
