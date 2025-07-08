from torch import nn
import torch
    

class CT25D(nn.Module):
    def __init__(self):
        super(CT25D, self).__init__()
                
        self.cnn = torch.hub.load('facebookresearch/dinov2', f'dinov2_vitb14')
        self.hidden_dim = 768
        self.lstm = nn.LSTM(768, 768, 2, batch_first=True, bidirectional=True)

        self.proj = nn.Sequential(                        
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim//2, 128),  
            )


    def forward(self, imgs):                
        bs, n_img, channel, w, h = imgs.shape
        imgs = imgs.reshape(bs * n_img, channel, w, h)
        img_feats = self.cnn(imgs)        
        img_feats = img_feats.reshape(bs, n_img, -1)        

        img_feats = self.lstm(img_feats)
        img_feats = img_feats.mean(1)        
        return img_feats
    