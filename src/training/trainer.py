import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import gc
import psutil

# 🚨 紧急内存优化配置 - 在所有操作之前执行
torch.cuda.empty_cache()
# 限制PyTorch内存分配策略，减少内存碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

from ..models.esrgan import LiteRealESRGAN, Discriminator
from ..models.losses import VGGLoss
from ..data.dataset import MemoryEfficientPairedDataset, create_data_loaders
from ..utils.metrics import calculate_psnr

class MemoryEfficientTrainer:
    def __init__(self, train_lr_dir, train_hr_dir, val_lr_dir, val_hr_dir, 
                 batch_size=2, lr=1e-4, device='cuda', max_cache_size=100,
                 gradient_accumulation_steps=4, mixed_precision=True, 
                 num_blocks=6, num_features=64):
        
        # 🚨 额外的内存优化设置
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 强制检查CUDA可用性
        if not torch.cuda.is_available():
            raise RuntimeError("此版本仅支持GPU训练，但未检测到CUDA设备！")
        
        # 设置显存使用限制
        torch.cuda.set_per_process_memory_fraction(0.85)  # 只使用85%显存
        
        self.device = torch.device('cuda')
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        
        # 保存模型配置
        self.num_blocks = num_blocks
        self.num_features = num_features
        
        # 检查GPU内存
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"检测到GPU显存: {gpu_memory:.1f}GB")
        print(f"🔧 内存优化配置已启用: max_split_size_mb=128, 显存限制=85%")
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        
        print(f"Using device: {self.device}")
        print(f"模型配置: {num_blocks}块, {num_features}特征")
        print(f"内存优化设置: 缓存大小 {max_cache_size}, 梯度累积步数 {gradient_accumulation_steps}")
        print(f"混合精度训练: {'启用' if mixed_precision else '禁用'}")
        
        # 数据加载 - 使用内存高效的数据加载器
        self.train_loader, self.val_loader = create_data_loaders(
            train_lr_dir, train_hr_dir, val_lr_dir, val_hr_dir,
            batch_size=batch_size, max_cache_size=max_cache_size, num_workers=0
        )
        
        # 模型初始化 - 使用配置参数
        self.generator = LiteRealESRGAN(num_blocks=num_blocks, num_features=num_features).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # 优化器
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.9, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.9, 0.999))
        
        # 学习率调度器
        self.g_scheduler = optim.lr_scheduler.StepLR(self.g_optimizer, step_size=30, gamma=0.5)
        self.d_scheduler = optim.lr_scheduler.StepLR(self.d_optimizer, step_size=30, gamma=0.5)
        
        # 损失函数
        self.l1_loss = nn.L1Loss()
        self.vgg_loss = VGGLoss().to(self.device)
        self.gan_loss = nn.BCEWithLogitsLoss()
        
        # 训练状态
        self.best_psnr = 0
        self.epoch = 0
        
        # 内存管理
        self.memory_cleanup_frequency = 20

    def get_system_memory_usage(self):
        """获取系统内存使用情况"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / 1024**3,  # GB
            'available': memory.available / 1024**3,  # GB
            'used': memory.used / 1024**3,  # GB
            'percent': memory.percent
        }

    def clear_memory(self):
        """更彻底的内存清理"""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    def train_epoch(self, epoch=None, total_epochs=None, progress_callback=None):
        """训练一个epoch - GPU专用版本"""
        self.generator.train()
        self.discriminator.train()
        
        total_g_loss = 0
        total_d_loss = 0
        
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(self.train_loader):
            # 将数据移动到GPU
            lr_imgs = lr_imgs.to(self.device, non_blocking=True)
            hr_imgs = hr_imgs.to(self.device, non_blocking=True)
            batch_size = lr_imgs.size(0)
            
            # 训练判别器
            if batch_idx % self.gradient_accumulation_steps == 0:
                self.d_optimizer.zero_grad()
            
            try:
                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    # 生成超分辨率图像
                    sr_imgs = self.generator(lr_imgs)
                    
                    # 真实图像
                    real_labels = torch.ones(batch_size, 1, device=self.device)
                    real_pred = self.discriminator(hr_imgs)
                    d_real_loss = self.gan_loss(real_pred, real_labels)
                    
                    # 生成图像
                    fake_labels = torch.zeros(batch_size, 1, device=self.device)
                    fake_pred = self.discriminator(sr_imgs.detach())
                    d_fake_loss = self.gan_loss(fake_pred, fake_labels)
                    
                    d_loss = (d_real_loss + d_fake_loss) / (2 * self.gradient_accumulation_steps)
                
                if self.scaler and self.mixed_precision:
                    self.scaler.scale(d_loss).backward()
                else:
                    d_loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.scaler and self.mixed_precision:
                        self.scaler.step(self.d_optimizer)
                        self.scaler.update()
                    else:
                        self.d_optimizer.step()
                    self.d_optimizer.zero_grad()
                
                # 训练生成器
                if batch_idx % self.gradient_accumulation_steps == 0:
                    self.g_optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    # 重新生成
                    sr_imgs = self.generator(lr_imgs)
                    
                    # L1损失
                    l1_loss = self.l1_loss(sr_imgs, hr_imgs)
                    
                    # 感知损失
                    perceptual_loss = self.vgg_loss(sr_imgs, hr_imgs)
                    
                    # GAN损失
                    fake_pred = self.discriminator(sr_imgs)
                    gan_loss = self.gan_loss(fake_pred, real_labels)
                    
                    # 总生成器损失
                    g_loss = (l1_loss + 0.1 * perceptual_loss + 0.01 * gan_loss) / self.gradient_accumulation_steps
                
                if self.scaler and self.mixed_precision:
                    self.scaler.scale(g_loss).backward()
                else:
                    g_loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.scaler and self.mixed_precision:
                        self.scaler.step(self.g_optimizer)
                        self.scaler.update()
                    else:
                        self.g_optimizer.step()
                    self.g_optimizer.zero_grad()
                
                total_g_loss += g_loss.item() * self.gradient_accumulation_steps
                total_d_loss += d_loss.item() * self.gradient_accumulation_steps
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"❌ GPU内存不足: {str(e)}")
                    self.clear_memory()
                    raise e
                else:
                    raise e
            
            # 定期清理内存
            if batch_idx % self.memory_cleanup_frequency == 0:
                self.clear_memory()
            
            # 删除不需要的张量
            try:
                del lr_imgs, hr_imgs, sr_imgs, real_pred, fake_pred
                del real_labels, fake_labels, l1_loss, perceptual_loss, gan_loss
                del d_real_loss, d_fake_loss
            except:
                pass
            
            # 进度回调
            if progress_callback and batch_idx % 10 == 0:
                progress_info = {
                    'epoch': epoch or self.epoch,
                    'total_epochs': total_epochs or 1,
                    'batch': batch_idx + 1,
                    'total_batches': len(self.train_loader),
                    'g_loss': total_g_loss/(batch_idx+1) if batch_idx > 0 else 0,
                    'd_loss': total_d_loss/(batch_idx+1) if batch_idx > 0 else 0,
                    'progress': (batch_idx + 1) / len(self.train_loader)
                }
                progress_callback(progress_info)
            
            # 详细输出
            if batch_idx % 10 == 0:
                current_g_loss = total_g_loss/(batch_idx+1) if batch_idx > 0 else 0
                current_d_loss = total_d_loss/(batch_idx+1) if batch_idx > 0 else 0
                
                print(f"[Epoch {epoch or self.epoch}/{total_epochs or 1}] "
                      f"Batch {batch_idx+1}/{len(self.train_loader)} [GPU] - "
                      f"G_Loss: {current_g_loss:.6f}, D_Loss: {current_d_loss:.6f}")
                
                # 显示GPU内存使用情况
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_cached = torch.cuda.memory_reserved() / 1024**3
                print(f"GPU内存: 使用 {memory_used:.2f}GB, 缓存 {memory_cached:.2f}GB")
        
        # 最终清理
        self.clear_memory()
        
        if len(self.train_loader) > 0:
            return total_g_loss / len(self.train_loader), total_d_loss / len(self.train_loader)
        else:
            return 0, 0

    def get_memory_info(self):
        """获取GPU内存信息"""
        return {
            'gpu_allocated': torch.cuda.memory_allocated() / 1024**3,
            'gpu_cached': torch.cuda.memory_reserved() / 1024**3,
            'gpu_total': torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
    
    def validate(self):
        """验证模型"""
        self.generator.eval()
        total_psnr = 0
        
        with torch.no_grad():
            for batch_idx, (lr_imgs, hr_imgs) in enumerate(self.val_loader):
                lr_imgs = lr_imgs.to(self.device, non_blocking=True)
                hr_imgs = hr_imgs.to(self.device, non_blocking=True)
                
                sr_imgs = self.generator(lr_imgs)
                
                # 计算PSNR (转换到[0,1]范围)
                sr_imgs_norm = (sr_imgs + 1) / 2
                hr_imgs_norm = (hr_imgs + 1) / 2
                
                psnr = calculate_psnr(sr_imgs_norm, hr_imgs_norm)
                total_psnr += psnr.item()
                
                # 删除张量释放内存
                del lr_imgs, hr_imgs, sr_imgs, sr_imgs_norm, hr_imgs_norm
                
                # 定期清理内存
                if batch_idx % 20 == 0:
                    self.clear_memory()
        
        # 清理验证后的内存
        self.clear_memory()
        
        avg_psnr = total_psnr / len(self.val_loader)
        return avg_psnr
    
    def save_checkpoint(self, filepath, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_scheduler_state_dict': self.g_scheduler.state_dict(),
            'd_scheduler_state_dict': self.d_scheduler.state_dict(),
            'best_psnr': self.best_psnr
        }
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = filepath.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")
    
    def load_checkpoint(self, filepath):
        """加载检查点"""
        try:
            print(f"正在加载检查点: {filepath}")
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            
            # 检查检查点格式
            required_keys = ['generator_state_dict', 'discriminator_state_dict']
            
            for key in required_keys:
                if key not in checkpoint:
                    raise KeyError(f"检查点缺少必要的键: {key}")
            
            # 加载生成器状态
            try:
                self.generator.load_state_dict(checkpoint['generator_state_dict'])
                print("✅ 生成器状态加载成功")
            except Exception as e:
                print(f"❌ 生成器状态加载失败: {e}")
                raise
            
            # 加载判别器状态
            try:
                self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                print("✅ 判别器状态加载成功")
            except Exception as e:
                print(f"❌ 判别器状态加载失败: {e}")
                raise
            
            # 检查模型结构是否匹配
            model_structure_changed = self._check_model_structure_change(checkpoint)
            
            # 加载优化器状态（仅在模型结构未变化时）
            if not model_structure_changed and 'g_optimizer_state_dict' in checkpoint and 'd_optimizer_state_dict' in checkpoint:
                try:
                    self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
                    self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
                    print("✅ 优化器状态加载成功")
                except Exception as e:
                    print(f"⚠️  优化器状态加载失败，将使用默认状态: {e}")
                    print("🔄 优化器将从默认状态开始")
            else:
                if model_structure_changed:
                    print("⚠️  检测到模型结构变化，跳过优化器状态加载")
                    print("🔄 优化器将从默认状态开始，这是正常的")
                else:
                    print("⚠️  检查点中缺少优化器状态，将使用默认状态")
            
            # 加载学习率调度器状态（仅在模型结构未变化时）
            if not model_structure_changed and 'g_scheduler_state_dict' in checkpoint and 'd_scheduler_state_dict' in checkpoint:
                try:
                    self.g_scheduler.load_state_dict(checkpoint['g_scheduler_state_dict'])
                    self.d_scheduler.load_state_dict(checkpoint['d_scheduler_state_dict'])
                    print("✅ 学习率调度器状态加载成功")
                except Exception as e:
                    print(f"⚠️  学习率调度器状态加载失败，将使用默认状态: {e}")
            else:
                if model_structure_changed:
                    print("⚠️  模型结构变化，学习率调度器将从默认状态开始")
            
            # 加载训练信息
            self.epoch = checkpoint.get('epoch', 0)
            self.best_psnr = checkpoint.get('best_psnr', 0.0)
            
            print(f"✅ 检查点加载完成")
            print(f"   - 起始epoch: {self.epoch}")
            print(f"   - 最佳PSNR: {self.best_psnr:.2f} dB")
            
            if model_structure_changed:
                print("🔄 由于模型结构变化，优化器和调度器已重置")
                print("💡 建议降低学习率以获得更稳定的训练")
            
            return True
            
        except Exception as e:
            print(f"❌ 检查点加载失败: {e}")
            print(f"   文件路径: {filepath}")
            import traceback
            traceback.print_exc()
            return False
    
    def _check_model_structure_change(self, checkpoint):
        """检查模型结构是否发生变化"""
        try:
            # 检查生成器参数形状
            current_state = self.generator.state_dict()
            checkpoint_state = checkpoint['generator_state_dict']
            
            for key in current_state.keys():
                if key in checkpoint_state:
                    if current_state[key].shape != checkpoint_state[key].shape:
                        print(f"🔍 检测到参数形状变化: {key}")
                        print(f"   检查点: {checkpoint_state[key].shape}")
                        print(f"   当前模型: {current_state[key].shape}")
                        return True
                else:
                    print(f"🔍 检测到新增参数: {key}")
                    return True
            
            # 检查是否有参数被删除
            for key in checkpoint_state.keys():
                if key not in current_state:
                    print(f"🔍 检测到删除的参数: {key}")
                    return True
            
            return False
            
        except Exception as e:
            print(f"⚠️  模型结构检查失败: {e}")
            # 如果检查失败，为了安全起见，假设结构发生了变化
            return True
    
    def train(self, num_epochs, save_dir='checkpoints', save_frequency=5):
        """完整训练流程"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"开始GPU训练，共 {num_epochs} 个epoch...")
        print(f"训练数据集大小: {len(self.train_loader.dataset)}")
        print(f"验证数据集大小: {len(self.val_loader.dataset)}")
        
        # 记录起始epoch，避免覆盖从检查点加载的epoch
        start_epoch = self.epoch
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            # 不要重新设置self.epoch，保持从检查点加载的值
            start_time = time.time()
            
            print(f"\n=== Epoch {epoch+1}/{start_epoch + num_epochs} ===")
            
            # 清理数据集缓存
            if hasattr(self.train_loader.dataset, 'clear_cache'):
                if epoch % 10 == 0:  # 每10个epoch清理一次缓存
                    self.train_loader.dataset.clear_cache()
                    self.val_loader.dataset.clear_cache()
                    print("数据集缓存已清理")
            
            # 训练
            print("开始训练...")
            g_loss, d_loss = self.train_epoch()
            
            # 验证
            print("开始验证...")
            val_psnr = self.validate()
            
            # 更新学习率
            self.g_scheduler.step()
            self.d_scheduler.step()
            
            epoch_time = time.time() - start_time
            
            print(f"\nEpoch {epoch+1} 完成:")
            print(f"G_Loss: {g_loss:.4f}, D_Loss: {d_loss:.4f}")
            print(f"Val PSNR: {val_psnr:.2f} dB")
            print(f"学习率: G={self.g_scheduler.get_last_lr()[0]:.6f}, D={self.d_scheduler.get_last_lr()[0]:.6f}")
            print(f"用时: {epoch_time:.2f}s")
            
            # 显示GPU内存使用情况
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_cached = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU内存: 使用 {memory_used:.2f}GB, 缓存 {memory_cached:.2f}GB")
            
            print("-" * 60)
            
            # 保存检查点
            is_best = val_psnr > self.best_psnr
            if is_best:
                self.best_psnr = val_psnr
                print(f"🎉 新的最佳PSNR: {self.best_psnr:.2f} dB")
            
            if (epoch + 1) % save_frequency == 0 or is_best:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                # 更新self.epoch为当前epoch，以便正确保存到检查点
                self.epoch = epoch
                self.save_checkpoint(checkpoint_path, is_best)
                print(f"检查点已保存: {checkpoint_path}")
        
        # 训练结束后更新最终的epoch
        self.epoch = start_epoch + num_epochs - 1
        print(f"\n🎊 训练完成! 最佳PSNR: {self.best_psnr:.2f} dB")


# 保持向后兼容性
Trainer = MemoryEfficientTrainer


def main():
    # 配置参数
    config = {
        'train_lr_dir': 'data/train/lr',  # 低分辨率训练图像目录
        'train_hr_dir': 'data/train/hr',  # 高分辨率训练图像目录
        'val_lr_dir': 'data/val/lr',      # 低分辨率验证图像目录
        'val_hr_dir': 'data/val/hr',      # 高分辨率验证图像目录
        'batch_size': 4,                  # GPU训练可以使用更大的批次
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'device': 'cuda',
        'max_cache_size': 500,            # GPU训练可以使用更大的缓存
        'gradient_accumulation_steps': 2   # 梯度累积步数
    }
    
    # 检查数据目录
    for dir_path in [config['train_lr_dir'], config['train_hr_dir'], 
                     config['val_lr_dir'], config['val_hr_dir']]:
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist!")
            print("Please create the data directories and add your training images.")
            return
    
    # 创建训练器
    trainer = MemoryEfficientTrainer(
        train_lr_dir=config['train_lr_dir'],
        train_hr_dir=config['train_hr_dir'],
        val_lr_dir=config['val_lr_dir'],
        val_hr_dir=config['val_hr_dir'],
        batch_size=config['batch_size'],
        lr=config['learning_rate'],
        device=config['device'],
        max_cache_size=config['max_cache_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps']
    )
    
    # 开始训练
    trainer.train(num_epochs=config['num_epochs'])

if __name__ == "__main__":
    main()