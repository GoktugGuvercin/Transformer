import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, x):
        batch_size, seq_len = x.size()[:2]
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, _, seq_len = x.size()[:3]
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # Use boolean mask directly
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        q = self.split_heads(self.q_proj(query))
        k = self.split_heads(self.k_proj(key))
        v = self.split_heads(self.v_proj(value))
        
        attn_output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)
        attn_output = self.combine_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=64, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, tgt_mask=None):
        # Self attention with mask
        attn_output, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout(attn_output)
        tgt = self.norm1(tgt)
        
        # Encoder-decoder attention
        attn_output, _ = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout(attn_output)
        tgt = self.norm2(tgt)
        
        # Feed forward
        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout(ff_output)
        tgt = self.norm3(tgt)
        
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, nhead) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def create_mask(self, sz):
        """Create boolean causal mask (True = positions to mask)"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    def forward(self, tgt, memory):
        tgt = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        tgt = self.pos_encoder(tgt)
        
        # Create boolean mask
        tgt_mask = self.create_mask(tgt.size(1)).to(tgt.device)
        
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask)
            
        return self.fc_out(tgt)

if __name__ == "__main__":
    vocab_size = 10000
    d_model = 64
    nhead = 8
    num_layers = 6
    
    decoder = TransformerDecoder(vocab_size, d_model, nhead, num_layers)
    
    tgt = torch.LongTensor([[1, 2, 3, 4, 5]])  # (batch_size, seq_len)
    memory = torch.randn(1, 5, d_model)         # Encoder output
    
    output = decoder(tgt, memory)
    print("Output shape:", output.shape)  # Should be (1, 5, vocab_size)
