from torch import nn
import torch

class CT25D(nn.Module):
    def __init__(self):
        super(CT25D, self).__init__()
                            
        self.cnn = torch.hub.load('facebookresearch/dinov2', f'dinov2_vitb14')
        self.hidden_dim = 768
        
        self.lstm = nn.LSTM(768, 768, 2, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(            
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 1))
        self.fc[-1].weight.data.normal_(mean=0.0, std=0.01)
        
    def forward(self, imgs):                
        bs, n_img, channel, w, h = imgs.shape
        imgs = imgs.reshape(bs * n_img, channel, w, h)
        img_feats = self.cnn(imgs)        
        img_feats = img_feats.reshape(bs, n_img, -1)        
                
        img_feats, _ = self.lstm(img_feats)
        img_feats = img_feats.mean(1)

        pred_os = self.fc(img_feats)
        return pred_os
    def get_features(self, imgs):
        bs, n_img, channel, w, h = imgs.shape
        imgs = imgs.reshape(bs * n_img, channel, w, h)
        img_feats = self.cnn(imgs)        
        img_feats = img_feats.reshape(bs, n_img, -1)        
                
        img_feats = self.lstm(img_feats)
        img_feats = img_feats.mean(1)        
        return img_feats    
    