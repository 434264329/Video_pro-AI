import os
import glob
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from PIL import Image
import random
import numpy as np
import gc

class MemoryEfficientPairedDataset(Dataset):
    """内存高效的配对数据集加载器 - 用于超分辨率训练"""
    
    def __init__(self, lr_dir, hr_dir, transform=None, augment=True, crop_size=None, 
                 max_cache_size=100, preload_batch_size=50, 
                 auto_resize=True, target_scale=2, max_lr_size=512, 
                 enforce_divisible=8, min_size_threshold=64):
        """
        初始化配对数据集
        
        Args:
            lr_dir: 低分辨率图像目录
            hr_dir: 高分辨率图像目录
            transform: 图像变换
            augment: 是否进行数据增强
            crop_size: 裁剪尺寸 (如果为None则不裁剪)
            max_cache_size: 最大缓存图像数量 (减小到100)
            preload_batch_size: 预加载批次大小 (减小到50)
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.augment = augment
        self.crop_size = crop_size or (256, 256)  # 默认裁剪到256x256
        self.max_cache_size = max_cache_size
        self.preload_batch_size = preload_batch_size
        
        # 图像缓存 - 使用LRU策略
        self.cache = {}
        self.cache_order = []  # 用于LRU缓存管理
        
        # 支持的图像格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        
        # 获取所有图像文件路径
        self.lr_paths = []
        self.hr_paths = []
        
        for ext in image_extensions:
            self.lr_paths.extend(glob.glob(os.path.join(lr_dir, ext)))
            self.lr_paths.extend(glob.glob(os.path.join(lr_dir, ext.upper())))
            self.hr_paths.extend(glob.glob(os.path.join(hr_dir, ext)))
            self.hr_paths.extend(glob.glob(os.path.join(hr_dir, ext.upper())))
        
        # 排序确保对应关系
        self.lr_paths = sorted(self.lr_paths)
        self.hr_paths = sorted(self.hr_paths)
        
        # 验证数据集
        self._validate_dataset()
        
        # 设置默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1, 1]
            ])
        else:
            self.transform = transform
            
        print(f"内存高效数据集初始化完成: {len(self.lr_paths)} 对图像")
        print(f"缓存设置: 最大缓存 {max_cache_size} 张图像, 预加载批次 {preload_batch_size}")
        print(f"图像将被裁剪到: {self.crop_size}")
    
    def _validate_dataset(self):
        """验证数据集的有效性"""
        # 检查目录是否存在
        if not os.path.exists(self.lr_dir):
            raise ValueError(f"低分辨率图像目录不存在: {self.lr_dir}")
        
        if not os.path.exists(self.hr_dir):
            raise ValueError(f"高分辨率图像目录不存在: {self.hr_dir}")
        
        # 检查是否找到图像文件
        if len(self.lr_paths) == 0:
            raise ValueError(f"在低分辨率目录中未找到图像文件: {self.lr_dir}")
        
        if len(self.hr_paths) == 0:
            raise ValueError(f"在高分辨率目录中未找到图像文件: {self.hr_dir}")
        
        # 检查图像数量是否匹配
        if len(self.lr_paths) != len(self.hr_paths):
            print(f"警告: LR图像数量 ({len(self.lr_paths)}) 与 HR图像数量 ({len(self.hr_paths)}) 不匹配")
            # 取较小的数量
            min_count = min(len(self.lr_paths), len(self.hr_paths))
            self.lr_paths = self.lr_paths[:min_count]
            self.hr_paths = self.hr_paths[:min_count]
            print(f"已调整为使用 {min_count} 对图像")
        
        # 验证文件名对应关系（可选）
        lr_names = [os.path.splitext(os.path.basename(p))[0] for p in self.lr_paths]
        hr_names = [os.path.splitext(os.path.basename(p))[0] for p in self.hr_paths]
        
        mismatched = []
        for i, (lr_name, hr_name) in enumerate(zip(lr_names, hr_names)):
            if lr_name != hr_name:
                mismatched.append((i, lr_name, hr_name))
        
        if mismatched:
            print(f"警告: 发现 {len(mismatched)} 对文件名不匹配的图像:")
            for i, lr_name, hr_name in mismatched[:5]:  # 只显示前5个
                print(f"  索引 {i}: LR='{lr_name}' vs HR='{hr_name}'")
            if len(mismatched) > 5:
                print(f"  ... 还有 {len(mismatched) - 5} 个不匹配")
        
        print(f"数据集验证完成: {len(self.lr_paths)} 对有效图像")
        
    def __getitem__(self, idx):
        """获取一对图像 - 动态加载优化版本"""
        if idx not in self.cache:
            if len(self.cache) >= self.max_cache_size:
                # 移除最久未使用的项
                oldest_idx = self.cache_order.pop(0)
                del self.cache[oldest_idx]
                # 强制垃圾回收
                gc.collect()
            # 动态加载图像
            self.cache[idx] = self._load_image_pair(idx)
            self.cache_order.append(idx)
        else:
            # 更新LRU顺序
            self.cache_order.remove(idx)
            self.cache_order.append(idx)
            
        return self.cache[idx]
    
    def _load_image_pair(self, idx):
        """加载并处理图像对"""
        try:
            # 加载图像
            lr_img = Image.open(self.lr_paths[idx]).convert("RGB")
            hr_img = Image.open(self.hr_paths[idx]).convert("RGB")
            
            # 确保图像尺寸不超过512x512（对4GB显存友好）
            max_size = 512
            if lr_img.size[0] > max_size or lr_img.size[1] > max_size:
                lr_img.thumbnail((max_size, max_size), Image.LANCZOS)
            if hr_img.size[0] > max_size*2 or hr_img.size[1] > max_size*2:
                hr_img.thumbnail((max_size*2, max_size*2), Image.LANCZOS)
            
            # 验证和调整尺寸
            lr_img, hr_img = self._process_image_pair(lr_img, hr_img)
            
            # 数据增强
            if self.augment:
                lr_img, hr_img = self._augment_images(lr_img, hr_img)
            
            # 裁剪到固定尺寸
            if self.crop_size:
                lr_img, hr_img = self._crop_images(lr_img, hr_img)
            
            # 应用变换
            lr_tensor = self.transform(lr_img)
            hr_tensor = self.transform(hr_img)
            
            return lr_tensor, hr_tensor
            
        except Exception as e:
            print(f"加载图像时出错 (索引 {idx}): {e}")
            # 返回下一个有效的图像
            return self._load_image_pair((idx + 1) % len(self))
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.cache_order.clear()
        gc.collect()
        print("图像缓存已清空")
    
    def preload_batch(self, start_idx, batch_size=None):
        """预加载一批图像"""
        if batch_size is None:
            batch_size = self.preload_batch_size
        
        end_idx = min(start_idx + batch_size, len(self))
        
        print(f"预加载图像 {start_idx} 到 {end_idx-1}...")
        
        for idx in range(start_idx, end_idx):
            if idx not in self.cache:
                self._load_image_pair(idx)
        
        print(f"预加载完成，当前缓存: {len(self.cache)} 张图像")
    
    def __len__(self):
        return len(self.lr_paths)
    
    def _process_image_pair(self, lr_img, hr_img):
        """处理图像对，确保尺寸关系正确"""
        lr_w, lr_h = lr_img.size
        hr_w, hr_h = hr_img.size
        
        # 期望HR是LR的2倍
        expected_hr_w = lr_w * 2
        expected_hr_h = lr_h * 2
        
        # 如果HR尺寸不正确，调整它
        if hr_w != expected_hr_w or hr_h != expected_hr_h:
            hr_img = hr_img.resize((expected_hr_w, expected_hr_h), Image.BICUBIC)
        
        # 确保尺寸是8的倍数（便于网络处理）
        lr_w_new = (lr_w // 8) * 8
        lr_h_new = (lr_h // 8) * 8
        
        if lr_w_new != lr_w or lr_h_new != lr_h:
            lr_img = lr_img.resize((lr_w_new, lr_h_new), Image.BICUBIC)
            hr_img = hr_img.resize((lr_w_new * 2, lr_h_new * 2), Image.BICUBIC)
        
        return lr_img, hr_img
    
    def _augment_images(self, lr_img, hr_img):
        """数据增强"""
        # 随机水平翻转
        if random.random() > 0.5:
            lr_img = lr_img.transpose(Image.FLIP_LEFT_RIGHT)
            hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 随机垂直翻转
        if random.random() > 0.5:
            lr_img = lr_img.transpose(Image.FLIP_TOP_BOTTOM)
            hr_img = hr_img.transpose(Image.FLIP_TOP_BOTTOM)
        
        # 随机旋转90度
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            lr_img = lr_img.rotate(angle)
            hr_img = hr_img.rotate(angle)
        
        return lr_img, hr_img
    
    def _crop_images(self, lr_img, hr_img):
        """随机裁剪图像"""
        lr_w, lr_h = lr_img.size
        
        # 确保crop_size是整数，如果是元组则取对应的值
        if isinstance(self.crop_size, tuple):
            crop_w, crop_h = self.crop_size
        else:
            crop_w = crop_h = self.crop_size
        
        if lr_w < crop_w or lr_h < crop_h:
            # 如果图像太小，先放大
            scale = max(crop_w / lr_w, crop_h / lr_h) * 1.1
            new_w = int(lr_w * scale)
            new_h = int(lr_h * scale)
            lr_img = lr_img.resize((new_w, new_h), Image.BICUBIC)
            hr_img = hr_img.resize((new_w * 2, new_h * 2), Image.BICUBIC)
            lr_w, lr_h = new_w, new_h
        
        # 随机裁剪位置
        x = random.randint(0, lr_w - crop_w)
        y = random.randint(0, lr_h - crop_h)
        
        # 裁剪LR图像
        lr_img = lr_img.crop((x, y, x + crop_w, y + crop_h))
        
        # 裁剪对应的HR区域
        hr_img = hr_img.crop((x * 2, y * 2, (x + crop_w) * 2, (y + crop_h) * 2))
        
        return lr_img, hr_img


class BatchSampler(Sampler):
    """分批采样器 - 用于内存管理"""
    
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.current_batch_start = 0
        
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            random.shuffle(indices)
        
        # 分批处理
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            if len(batch_indices) < self.batch_size and self.drop_last:
                continue
                
            # 预加载这一批的图像
            if hasattr(self.dataset, 'preload_batch'):
                min_idx = min(batch_indices)
                max_idx = max(batch_indices)
                self.dataset.preload_batch(min_idx, max_idx - min_idx + 1)
            
            yield batch_indices
    
    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


# 保持向后兼容性
PairedDataset = MemoryEfficientPairedDataset


class SingleImageDataset(Dataset):
    """单图像数据集 - 用于推理"""
    
    def __init__(self, image_dir, transform=None):
        """
        初始化单图像数据集
        
        Args:
            image_dir: 图像目录
            transform: 图像变换
        """
        self.image_dir = image_dir
        
        # 支持的图像格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        
        # 获取所有图像文件路径
        self.image_paths = []
        for ext in image_extensions:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))
        
        self.image_paths = sorted(self.image_paths)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"在目录 {image_dir} 中未找到图像文件")
        
        # 设置默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
            
        print(f"单图像数据集初始化完成: {len(self.image_paths)} 张图像")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """获取单张图像"""
        try:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert("RGB")
            
            # 确保尺寸是8的倍数
            w, h = image.size
            w_new = (w // 8) * 8
            h_new = (h // 8) * 8
            
            if w_new != w or h_new != h:
                image = image.resize((w_new, h_new), Image.BICUBIC)
            
            tensor = self.transform(image)
            return tensor, image_path
            
        except Exception as e:
            print(f"加载图像时出错 (索引 {idx}): {e}")
            return self.__getitem__((idx + 1) % len(self))


def create_data_loaders(train_lr_dir, train_hr_dir, val_lr_dir, val_hr_dir, 
                       batch_size=4, num_workers=2, max_cache_size=500):
    """创建数据加载器"""
    
    # 创建数据集
    train_dataset = MemoryEfficientPairedDataset(
        train_lr_dir, train_hr_dir, 
        max_cache_size=max_cache_size,
        preload_batch_size=batch_size * 10  # 预加载10个批次的数据
    )
    
    val_dataset = MemoryEfficientPairedDataset(
        val_lr_dir, val_hr_dir,
        augment=False,  # 验证时不进行数据增强
        max_cache_size=max_cache_size // 2,
        preload_batch_size=batch_size * 5
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,  # 验证时使用较小的批次
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# 测试代码
if __name__ == "__main__":
    # 测试内存高效数据集
    try:
        dataset = MemoryEfficientPairedDataset(
            "data/train/lr", 
            "data/train/hr",
            max_cache_size=100,
            preload_batch_size=50
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        # 测试加载几个样本
        for i in range(min(5, len(dataset))):
            lr, hr = dataset[i]
            print(f"样本 {i}: LR shape: {lr.shape}, HR shape: {hr.shape}")
        
        # 测试缓存清理
        dataset.clear_cache()
        
        print("数据集测试完成!")
        
    except Exception as e:
        print(f"数据集测试失败: {e}")