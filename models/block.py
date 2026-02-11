import torch
from torch import nn



class MultiHeadSABlock(nn.Module):
    def __init__(self, in_features, num_classes, heads=1, activation="linear", temperature=2.0):
        super().__init__()
        self.heads = heads
        self.activation = activation
        self.temperature = torch.tensor(float(temperature))

        self.att = nn.Conv2d(in_features, num_classes * heads, kernel_size=(1, 1), bias=True)
        self.cla = nn.Conv2d(in_features, num_classes * heads, kernel_size=1, bias=True)
        self.head_weights = nn.Parameter(torch.ones(self.heads))

    def nonlinearity(self, x):
        activations = {
            "linear": lambda t: t,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
        }
        if self.activation not in activations:
            raise ValueError(f"Unsupported activation: {self.activation}")
        return activations[self.activation](x)

    def forward(self, x, crop):
        # x: (B, C, F, T)
        if crop > 0:
            x = x[:, :, :, crop: -crop]

        B, C, Fq, T = x.size()

        att_map = self.att(x) / self.temperature
        att_map = att_map.view(B, -1, self.heads, Fq, T)

        norm_att = torch.softmax(att_map.view(B, -1, self.heads, Fq * T), dim=-1)
        norm_att = norm_att.view(B, -1, self.heads, Fq, T)


        cla_feat = self.nonlinearity(self.cla(x))
        cla_feat = cla_feat.view(B, -1, self.heads, Fq, T)
        weighted = norm_att * cla_feat
        weighted = weighted.sum(dim=-1).sum(dim=-1)

        final_features = (weighted * self.head_weights.view(1, 1, -1)).sum(dim=-1) / self.head_weights.sum()

        return final_features, norm_att, cla_feat



class GatedAttention(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.att = nn.Linear(in_dim, n_classes)
        self.gate = nn.Linear(in_dim, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: (B, T, D)
        attn_logits = torch.tanh(self.att(x)) * torch.sigmoid(self.gate(x))
        attn_weights = self.softmax(attn_logits)  # (B, T, n_classes)
        return attn_weights