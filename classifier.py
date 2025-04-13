import torch.nn as nn

class CLS(nn.Module):
    def __init__(self, nb_classes, sz_embed):
        super(CLS, self).__init__()
        self.theta1 = nn.Linear(sz_embed, nb_classes)

    def forward(self, X):
        out = self.theta1(X)
        return out