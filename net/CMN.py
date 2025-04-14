import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.models import resnet50


class CMN(nn.Module):
    def __init__(self, embedding_size, concept_dim, pretrained=True, is_norm=True, bn_freeze = True):
        super(CMN, self).__init__()

        self.model = resnet50(pretrained)
        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.concept_dim = concept_dim
        self.num_ftrs = self.model.fc.in_features
        
        self.MLP1 = nn.Linear(self.num_ftrs, self.num_ftrs)
        self.lnorm1 = nn.LayerNorm(self.num_ftrs, elementwise_affine=True).cuda()
        self.MLP2 = nn.Linear(concept_dim, self.num_ftrs, bias=False)
        self.MLP3 = nn.Linear(concept_dim, self.num_ftrs, bias=False)
        self.lnorm2 = nn.LayerNorm(concept_dim, elementwise_affine=True).cuda()
        self.lnorm3 = nn.LayerNorm(concept_dim, elementwise_affine=True).cuda()
        self._initialize_weights()
        
        self.concept_v = nn.Parameter(torch.randn(self.embedding_size, self.concept_dim).cuda())
        nn.init.kaiming_normal_(self.concept_v, mode='fan_out')

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = (x.view(x.size(0), x.size(1), -1)).transpose(-1, -2)
        x = self.MLP1(self.lnorm1(x)).transpose(-1, -2)

        concept_v1 = self.MLP2(self.lnorm2(self.concept_v))

        A = torch.einsum('if, bfr->bir', concept_v1, x)
        A = nn.Softmax(dim = -1)(F.normalize(A)*A.shape[-1])
        F_p = torch.einsum('bir, bfr->bif', A, x)     # compute concept related visual features
        concept_v2 = self.MLP3(self.lnorm3(self.concept_v))
        Pred_concept = torch.einsum('if, bif->bi', F.normalize(concept_v2), F.normalize(F_p))

        if self.is_norm:
            Pred_concept = self.l2_norm(Pred_concept)
        
        return Pred_concept

    def _initialize_weights(self):
        init.kaiming_normal_(self.MLP1.weight, mode='fan_out')
        init.constant_(self.MLP1.bias, 0)
        init.kaiming_normal_(self.MLP2.weight, mode='fan_out')
        init.kaiming_normal_(self.MLP3.weight, mode='fan_out')
    
    def Concept_Separation_Loss(self):
        concept_v = F.normalize(self.concept_v)
        correlation = torch.matmul(concept_v, concept_v.T)
        correlation = torch.einsum('ij, ij->ij', correlation, 1-torch.eye(correlation.shape[0]).cuda())
        loss = torch.log(1+abs(correlation)).mean()*correlation.shape[1]
        return loss
