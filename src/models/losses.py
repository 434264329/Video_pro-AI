import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGLoss(nn.Module):
    """基于VGG19的感知损失"""
    def __init__(self):
        super(VGGLoss, self).__init__()
        try:
            import torchvision.models as models
            vgg = models.vgg19(pretrained=True).features
            self.vgg = nn.Sequential(*list(vgg.children())[:35]).eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        except:
            print("Warning: VGG19 not available, using L1 loss only")
            self.vgg = None

    def forward(self, x, y):
        if self.vgg is None:
            return F.l1_loss(x, y)

        # 归一化到 [0, 1]
        x = (x + 1) / 2
        y = (y + 1) / 2

        vgg_x = self.vgg(x)
        vgg_y = self.vgg(y)
        return F.mse_loss(vgg_x, vgg_y)

class CombinedLoss(nn.Module):
    """组合损失函数"""
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1, gan_weight=0.01):
        super(CombinedLoss, self).__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.gan_weight = gan_weight
        
        self.l1_loss = nn.L1Loss()
        self.vgg_loss = VGGLoss()
        # 使用 BCEWithLogitsLoss 替代 BCELoss，这样在 autocast 下是安全的
        self.gan_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, sr_imgs, hr_imgs, fake_pred, real_labels):
        """
        计算组合损失
        Args:
            sr_imgs: 生成的超分辨率图像
            hr_imgs: 真实高分辨率图像
            fake_pred: 判别器对生成图像的预测（logits，未经过sigmoid）
            real_labels: 真实标签
        """
        # L1损失
        l1_loss = self.l1_loss(sr_imgs, hr_imgs)
        
        # 感知损失
        perceptual_loss = self.vgg_loss(sr_imgs, hr_imgs)
        
        # GAN损失 - 现在使用 logits
        gan_loss = self.gan_loss(fake_pred, real_labels)
        
        # 总损失
        total_loss = (self.l1_weight * l1_loss + 
                     self.perceptual_weight * perceptual_loss + 
                     self.gan_weight * gan_loss)
        
        return total_loss, {
            'l1_loss': l1_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'gan_loss': gan_loss.item(),
            'total_loss': total_loss.item()
        }
