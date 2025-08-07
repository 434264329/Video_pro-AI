import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseBlock(nn.Module):
    """密集连接块"""
    def __init__(self, in_channels, growth_rate=32):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_rate, growth_rate, 3, padding=1)
        self.conv4 = nn.Conv2d(in_channels + 3 * growth_rate, growth_rate, 3, padding=1)
        self.conv5 = nn.Conv2d(in_channels + 4 * growth_rate, in_channels, 3, padding=1)
        
    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x), 0.2)
        x2 = F.leaky_relu(self.conv2(torch.cat([x, x1], 1)), 0.2)
        x3 = F.leaky_relu(self.conv3(torch.cat([x, x1, x2], 1)), 0.2)
        x4 = F.leaky_relu(self.conv4(torch.cat([x, x1, x2, x3], 1)), 0.2)
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual-in-Residual Dense Block"""
    def __init__(self, in_channels):
        super(RRDB, self).__init__()
        self.dense1 = DenseBlock(in_channels)
        self.dense2 = DenseBlock(in_channels)
        self.dense3 = DenseBlock(in_channels)
        
    def forward(self, x):
        out = self.dense1(x)
        out = self.dense2(out)
        out = self.dense3(out)
        return out * 0.2 + x

class PixelShuffle2x(nn.Module):
    """2倍上采样模块"""
    def __init__(self, in_channels):
        super(PixelShuffle2x, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * 4, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return F.leaky_relu(x, 0.2)

class LiteRealESRGAN(nn.Module):
    """轻量化 Real-ESRGAN 模型"""
    def __init__(self, num_blocks=8, num_features=64):
        super(LiteRealESRGAN, self).__init__()
        
        # 特征提取
        self.conv_first = nn.Conv2d(3, num_features, 3, padding=1)
        
        # RRDB 主干网络
        self.rrdb_blocks = nn.Sequential(*[RRDB(num_features) for _ in range(num_blocks)])
        
        # 特征融合
        self.conv_body = nn.Conv2d(num_features, num_features, 3, padding=1)
        
        # 上采样 (2x)
        self.upsampler = PixelShuffle2x(num_features)
        
        # 输出层
        self.conv_last = nn.Conv2d(num_features, 3, 3, padding=1)
        
    def forward(self, x):
        # 特征提取
        fea = F.leaky_relu(self.conv_first(x), 0.2)
        
        # 主干特征提取
        trunk = self.rrdb_blocks(fea)
        trunk = self.conv_body(trunk)
        fea = fea + trunk
        
        # 上采样
        fea = self.upsampler(fea)
        
        # 输出
        out = self.conv_last(fea)
        return out

class Discriminator(nn.Module):
    """判别器网络"""
    def __init__(self, input_size=128):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(3, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
            # 移除 nn.Sigmoid()，因为使用 BCEWithLogitsLoss
        )
        
    def forward(self, img):
        return self.model(img)

if __name__ == "__main__":
    # 测试模型 - 支持CPU和GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    generator = LiteRealESRGAN(num_blocks=6, num_features=32).to(device)
    discriminator = Discriminator().to(device)
    
    # 测试输入
    test_input = torch.randn(1, 3, 64, 64).to(device)
    
    # 前向传播
    with torch.no_grad():
        sr_output = generator(test_input)
        d_output = discriminator(sr_output)
    
    print(f"Input shape: {test_input.shape}")
    print(f"SR output shape: {sr_output.shape}")
    print(f"Discriminator output shape: {d_output.shape}")
    print("Model test completed successfully!")