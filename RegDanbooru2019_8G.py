
import torch
import torch.nn as nn
import torch.nn.functional as F

from RegNetY_8G import build_model

class RegDanbooru2019(nn.Module) :
    def __init__(self) :
        super(RegDanbooru2019, self).__init__()
        self.backbone = build_model()
        num_p = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print( 'Backbone has %d parameters' % num_p )
        self.head_danbooru = nn.Linear(2016, 4096)

    def forward_train_head(self, images) :
        """
        images of shape [N, 3, 512, 512]
        """
        with torch.no_grad() :
            feats = self.backbone(images)
            feats = F.adaptive_avg_pool2d(feats, 1).view(-1, 2016)
        danbooru_logits = self.head_danbooru(feats) # [N, 4096]
        return danbooru_logits

    def forward(self, images) :
        """
        images of shape [N, 3, 512, 512]
        """
        feats = self.backbone(images)
        feats = F.adaptive_avg_pool2d(feats, 1).view(-1, 2016)
        danbooru_logits = self.head_danbooru(feats) # [N, 4096]
        return danbooru_logits
