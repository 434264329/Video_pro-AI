import os
import time
import math
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from ..models.esrgan import LiteRealESRGAN

class EnhancedSuperResolutionPredictor:
    """增强版超分辨率预测器 - GPU专用版，支持大图片分块处理和模型属性自动读取"""
    
    def __init__(self, model_path, device='cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用，此预测器仅支持GPU运行")
        
        self.device = torch.device('cuda')
        print(f"Using device: {self.device}")
        
        # 模型属性
        self.model_info = {}
        self.scale_factor = 2  # 默认2倍上采样
        self.max_tile_size = 480  # 最大分块尺寸
        self.tile_overlap = 32  # 分块重叠像素
        
        # 加载模型并读取属性
        self.model = None
        self.load_model_with_info(model_path)
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 内存管理
        self.enable_memory_optimization()
        
    def enable_memory_optimization(self):
        """启用内存优化"""
        # 设置CUDA内存分配策略
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
            
    def load_model_with_info(self, model_path):
        """加载模型并读取模型信息"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 读取模型信息
        self.model_info = {
            'model_path': model_path,
            'file_size': os.path.getsize(model_path) / (1024 * 1024),  # MB
            'creation_time': os.path.getctime(model_path)
        }
        
        # 从checkpoint读取训练信息
        if isinstance(checkpoint, dict):
            if 'generator_state_dict' in checkpoint:
                # 训练保存的格式
                state_dict = checkpoint['generator_state_dict']
                self.model_info.update({
                    'epoch': checkpoint.get('epoch', 'unknown'),
                    'best_psnr': checkpoint.get('best_psnr', 0),
                    'best_ssim': checkpoint.get('best_ssim', 0),
                    'training_loss': checkpoint.get('training_loss', 0),
                    'validation_loss': checkpoint.get('validation_loss', 0),
                    'learning_rate': checkpoint.get('learning_rate', 0),
                    'optimizer_state': 'optimizer_state_dict' in checkpoint
                })
            else:
                # 直接保存的state_dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        # 分析模型结构
        self._analyze_model_structure(state_dict)
        
        # 创建并加载模型
        num_blocks = self.model_info.get('num_blocks', 8)
        num_features = self.model_info.get('num_features', 64)
        
        self.model = LiteRealESRGAN(num_blocks=num_blocks, num_features=num_features).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print(f"Model loaded successfully from {model_path}")
        self._print_model_info()
        
    def _analyze_model_structure(self, state_dict):
        """分析模型结构参数"""
        # 分析RRDB块数量
        rrdb_keys = [k for k in state_dict.keys() if 'rrdb_blocks' in k]
        if rrdb_keys:
            max_block_idx = max([int(k.split('.')[1]) for k in rrdb_keys if k.split('.')[1].isdigit()])
            self.model_info['num_blocks'] = max_block_idx + 1
        else:
            self.model_info['num_blocks'] = 8
            
        # 分析特征通道数
        conv_first_weight = state_dict.get('conv_first.weight')
        if conv_first_weight is not None:
            self.model_info['num_features'] = conv_first_weight.shape[0]
        else:
            self.model_info['num_features'] = 64
            
        # 计算模型参数量
        total_params = sum(p.numel() for p in state_dict.values())
        self.model_info['total_parameters'] = total_params
        self.model_info['model_size_mb'] = total_params * 4 / (1024 * 1024)  # 假设float32
        
        # 分析上采样倍数（通过PixelShuffle层）
        upsampler_keys = [k for k in state_dict.keys() if 'upsampler' in k and 'conv.weight' in k]
        if upsampler_keys:
            upsampler_weight = state_dict[upsampler_keys[0]]
            in_channels = upsampler_weight.shape[1]
            out_channels = upsampler_weight.shape[0]
            self.scale_factor = int(math.sqrt(out_channels // in_channels))
        
        self.model_info['scale_factor'] = self.scale_factor
        
    def _print_model_info(self):
        """打印模型信息"""
        print("\n=== 模型信息 ===")
        print(f"文件大小: {self.model_info.get('file_size', 0):.2f} MB")
        print(f"模型参数量: {self.model_info.get('total_parameters', 0):,}")
        print(f"模型大小: {self.model_info.get('model_size_mb', 0):.2f} MB")
        print(f"RRDB块数量: {self.model_info.get('num_blocks', 8)}")
        print(f"特征通道数: {self.model_info.get('num_features', 64)}")
        print(f"上采样倍数: {self.model_info.get('scale_factor', 2)}x")
        
        if 'epoch' in self.model_info:
            print(f"训练轮次: {self.model_info['epoch']}")
        if 'best_psnr' in self.model_info and self.model_info['best_psnr'] > 0:
            print(f"最佳PSNR: {self.model_info['best_psnr']:.2f} dB")
        if 'best_ssim' in self.model_info and self.model_info['best_ssim'] > 0:
            print(f"最佳SSIM: {self.model_info['best_ssim']:.4f}")
        print("================\n")
        
    def get_model_info(self):
        """获取模型信息"""
        return self.model_info.copy()
        
    def set_tile_size(self, tile_size):
        """设置分块大小"""
        self.max_tile_size = tile_size
        print(f"分块大小设置为: {tile_size}x{tile_size}")
        
    def predict(self, image):
        """预测单张图像 - 主要接口方法"""
        return self.process_image_with_tiling(image)
        
    def process_image_with_tiling(self, image):
        """使用分块处理大图像"""
        # 如果输入是路径，先加载图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            
        w, h = image.size
        print(f"处理图像: {w}x{h}")
        
        # 判断是否需要分块处理
        if w <= self.max_tile_size and h <= self.max_tile_size:
            print("图像尺寸较小，使用直接处理")
            return self._process_single_tile(image)
        else:
            print(f"图像尺寸较大，使用分块处理 (分块大小: {self.max_tile_size}x{self.max_tile_size})")
            return self._process_with_tiles(image)
            
    def _process_single_tile(self, image):
        """处理单个分块"""
        # 确保图像尺寸是偶数
        w, h = image.size
        if w % 2 != 0:
            w -= 1
        if h % 2 != 0:
            h -= 1
        image = image.resize((w, h), Image.Resampling.LANCZOS)
        
        # 转换为tensor
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            torch.cuda.empty_cache()  # 清理缓存
            output_tensor = self.model(input_tensor)
            torch.cuda.empty_cache()  # 清理缓存
        
        # 后处理
        output_image = self._postprocess_tensor(output_tensor)
        
        # 清理内存
        del input_tensor, output_tensor
        torch.cuda.empty_cache()
        
        return output_image
        
    def _process_with_tiles(self, image):
        """分块处理大图像"""
        w, h = image.size
        tile_size = self.max_tile_size
        overlap = self.tile_overlap
        
        # 计算分块数量
        tiles_x = math.ceil(w / (tile_size - overlap))
        tiles_y = math.ceil(h / (tile_size - overlap))
        
        print(f"分块数量: {tiles_x}x{tiles_y} = {tiles_x * tiles_y}个分块")
        
        # 创建输出图像
        output_w = w * self.scale_factor
        output_h = h * self.scale_factor
        output_image = Image.new('RGB', (output_w, output_h))
        
        # 权重图，用于处理重叠区域
        weight_map = np.zeros((output_h, output_w), dtype=np.float32)
        result_array = np.zeros((output_h, output_w, 3), dtype=np.float32)
        
        for tile_y in range(tiles_y):
            for tile_x in range(tiles_x):
                # 计算分块位置
                start_x = tile_x * (tile_size - overlap)
                start_y = tile_y * (tile_size - overlap)
                end_x = min(start_x + tile_size, w)
                end_y = min(start_y + tile_size, h)
                
                # 提取分块
                tile = image.crop((start_x, start_y, end_x, end_y))
                
                print(f"处理分块 [{tile_y+1}/{tiles_y}][{tile_x+1}/{tiles_x}]: "
                      f"{start_x},{start_y} -> {end_x},{end_y} ({tile.size[0]}x{tile.size[1]})")
                
                # 处理分块
                try:
                    sr_tile = self._process_single_tile(tile)
                    
                    # 计算输出位置
                    out_start_x = start_x * self.scale_factor
                    out_start_y = start_y * self.scale_factor
                    out_end_x = out_start_x + sr_tile.size[0]
                    out_end_y = out_start_y + sr_tile.size[1]
                    
                    # 转换为numpy数组
                    sr_array = np.array(sr_tile, dtype=np.float32)
                    
                    # 创建权重（中心权重高，边缘权重低）
                    tile_h, tile_w = sr_array.shape[:2]
                    tile_weight = self._create_tile_weight(tile_w, tile_h, overlap * self.scale_factor)
                    
                    # 累加到结果中
                    result_array[out_start_y:out_end_y, out_start_x:out_end_x] += sr_array * tile_weight[:, :, np.newaxis]
                    weight_map[out_start_y:out_end_y, out_start_x:out_end_x] += tile_weight
                    
                except Exception as e:
                    print(f"处理分块时出错: {e}")
                    continue
        
        # 归一化结果
        weight_map[weight_map == 0] = 1  # 避免除零
        result_array = result_array / weight_map[:, :, np.newaxis]
        result_array = np.clip(result_array, 0, 255).astype(np.uint8)
        
        # 转换回PIL图像
        output_image = Image.fromarray(result_array)
        
        print(f"分块处理完成，输出尺寸: {output_image.size}")
        return output_image
        
    def _create_tile_weight(self, width, height, overlap):
        """创建分块权重图"""
        weight = np.ones((height, width), dtype=np.float32)
        
        if overlap > 0:
            # 创建渐变权重
            fade = np.linspace(0, 1, overlap)
            
            # 左边缘
            if width > overlap:
                weight[:, :overlap] *= fade[np.newaxis, :]
            
            # 右边缘
            if width > overlap:
                weight[:, -overlap:] *= fade[np.newaxis, ::-1]
            
            # 上边缘
            if height > overlap:
                weight[:overlap, :] *= fade[:, np.newaxis]
            
            # 下边缘
            if height > overlap:
                weight[-overlap:, :] *= fade[::-1, np.newaxis]
        
        return weight
        
    def _postprocess_tensor(self, tensor):
        """后处理输出tensor"""
        # 反归一化到[0,1]
        tensor = (tensor + 1) / 2
        tensor = torch.clamp(tensor, 0, 1)
        
        # 转换为PIL图像 - 移动到CPU进行图像转换
        tensor = tensor.squeeze(0).detach().cpu()
        image = transforms.ToPILImage()(tensor)
        return image
        
    def process_batch(self, images, batch_size=2):
        """批量处理图像（降低batch_size以节省内存）"""
        results = []
        for i, image in enumerate(images):
            print(f"处理图像 {i+1}/{len(images)}")
            result = self.process_image_with_tiling(image)
            results.append(result)
            
            # 定期清理内存
            if (i + 1) % batch_size == 0:
                torch.cuda.empty_cache()
        
        return results
        
    def estimate_memory_usage(self, image_size):
        """估算内存使用量"""
        w, h = image_size
        
        # 输入图像内存 (RGB, float32)
        input_memory = w * h * 3 * 4
        
        # 输出图像内存
        output_memory = w * h * self.scale_factor * self.scale_factor * 3 * 4
        
        # 模型内存
        model_memory = self.model_info.get('model_size_mb', 0) * 1024 * 1024
        
        # 中间特征图内存（估算）
        feature_memory = w * h * self.model_info.get('num_features', 64) * 4 * 2
        
        total_memory = input_memory + output_memory + model_memory + feature_memory
        
        return {
            'input_mb': input_memory / (1024 * 1024),
            'output_mb': output_memory / (1024 * 1024),
            'model_mb': model_memory / (1024 * 1024),
            'feature_mb': feature_memory / (1024 * 1024),
            'total_mb': total_memory / (1024 * 1024),
            'recommended_tile_size': self._recommend_tile_size(total_memory)
        }
        
    def _recommend_tile_size(self, estimated_memory):
        """根据内存使用量推荐分块大小"""
        # 获取可用GPU内存
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = gpu_memory * 0.8  # 使用80%的GPU内存
            
        if estimated_memory > available_memory:
            # 需要分块处理
            ratio = estimated_memory / available_memory
            recommended_size = int(self.max_tile_size / math.sqrt(ratio))
            return max(256, recommended_size)  # 最小256
        else:
            return self.max_tile_size