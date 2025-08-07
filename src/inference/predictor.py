import os
import time
import torch
from PIL import Image
from torchvision import transforms

from ..models.esrgan import LiteRealESRGAN

class ImageSuperResolution:
    """图像超分辨率处理类 - GPU专用版"""
    
    def __init__(self, model_path, device='cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用，此推理器仅支持GPU运行")
        
        self.device = torch.device('cuda')
        print(f"Using device: {self.device}")
        
        # 加载模型
        self.model = LiteRealESRGAN(num_blocks=8).to(self.device)
        self.load_model(model_path)
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def load_model(self, model_path):
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 兼容不同的保存格式
        if 'generator_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['generator_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'best_psnr' in checkpoint:
                print(f"Best PSNR: {checkpoint['best_psnr']:.2f} dB")
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"Model loaded successfully from {model_path}")
    
    def process_image(self, image):
        """处理PIL图像对象"""
        # 如果输入是路径，先加载图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # 确保图像尺寸是偶数（便于2倍上采样）
        w, h = image.size
        if w % 2 != 0:
            w -= 1
        if h % 2 != 0:
            h -= 1
        image = image.resize((w, h), Image.BICUBIC)
        
        # 转换为tensor
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        
        # 后处理
        output_image = self.postprocess_tensor(output_tensor)
        
        return output_image
    
    def postprocess_tensor(self, tensor):
        """后处理输出tensor"""
        # 反归一化到[0,1]
        tensor = (tensor + 1) / 2
        tensor = torch.clamp(tensor, 0, 1)
        
        # 转换为PIL图像 - 移动到CPU进行图像转换
        tensor = tensor.squeeze(0).detach().cpu()
        image = transforms.ToPILImage()(tensor)
        return image
