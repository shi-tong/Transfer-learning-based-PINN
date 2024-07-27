import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightAttentionModule(nn.Module):
    def __init__(self, input_dim):
        super(LightweightAttentionModule, self).__init__()
        self.attention_weights = nn.Parameter(torch.zeros(input_dim))  # 将注意力权重初始化为接近零的值

    def forward(self, inputs):
        attention_scores = F.softmax(self.attention_weights, dim=0)
        weighted_inputs = inputs * attention_scores.unsqueeze(0)
        return weighted_inputs

class FNN(nn.Module):
    def __init__(self, layers, activation, in_tf=None, out_tf=None):
        super().__init__()
        self.activation = activation
        self.linears = nn.ModuleList()
        self.in_tf = in_tf
        self.out_tf = out_tf

        # Lightweight Attention Module
        self.attention_module = LightweightAttentionModule(layers[0])

        # Weight initialization
        for i in range(1, len(layers)):
            self.linears.append(nn.Linear(layers[i-1], layers[i]))
            nn.init.xavier_uniform_(self.linears[-1].weight)
            nn.init.zeros_(self.linears[-1].bias)

    def forward(self, inputs):
        X = inputs

        # Apply Lightweight Attention Module
        X = self.attention_module(X)

        # Input transformation
        if self.in_tf:
            X = self.in_tf(X)

        # Linear layers with activation
        for i, linear in enumerate(self.linears[:-2]):
            X = linear(X)
            if self.activation:
                X = self.activation(X)

        # Residual connection before the last layer
        X_res = X

        # Apply activation to the second last layer
        if self.activation:
            X = self.activation(X)

        # Last layer without activation
        X = self.linears[-2](X)

        # Add residual connection
        X += X_res

        # Apply activation to the last layer if provided
        if self.activation and len(self.linears) > 2:
            X = self.activation(X)

        # Last layer, no activation
        X = self.linears[-1](X)

        # Output transformation
        if self.out_tf:
            X = self.out_tf(X)

        return X



