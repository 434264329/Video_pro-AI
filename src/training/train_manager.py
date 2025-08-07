import os
import time
import json
import threading
from datetime import datetime
import torch
import matplotlib.pyplot as plt
import gc

# 🚨 紧急内存优化配置 - 在所有操作之前执行
torch.cuda.empty_cache()
# 限制PyTorch内存分配策略，减少内存碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

from .trainer import MemoryEfficientTrainer
from ..utils.metrics import calculate_psnr

class MemoryOptimizedTrainingManager:
    """内存优化的训练管理器 - 负责训练流程控制和监控"""
    
    def __init__(self, config):
        # 🚨 初始化时立即执行内存优化
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            # 设置显存使用限制
            torch.cuda.set_per_process_memory_fraction(0.85)
            print("🔧 训练管理器内存优化配置已启用")
        
        self.config = config
        self.trainer = None
        self.is_training = False
        self.training_thread = None
        self.progress_callback = None
        self.is_incremental = False  # 新增：标记是否为增量训练
        self.checkpoint_path = None  # 新增：预训练模型路径
        self.callbacks = {
            'on_epoch_start': [],
            'on_epoch_end': [],
            'on_training_start': [],
            'on_training_end': [],
            'on_progress_update': [],
            'on_memory_status': [],
            'on_error': []
        }
        
        # 训练历史
        self.training_history = {
            'epochs': [],
            'train_g_loss': [],
            'train_d_loss': [],
            'val_psnr': [],
            'learning_rates': [],
            'times': [],
            'memory_usage': []
        }
        
        # 内存监控
        self.memory_monitor_enabled = True
        
    def add_callback(self, event, callback):
        """添加回调函数"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def _trigger_callback(self, event, *args, **kwargs):
        """触发回调函数"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                print(f"回调函数执行错误: {e}")
    
    def get_memory_info(self):
        """获取内存使用信息"""
        memory_info = {}
        
        if torch.cuda.is_available():
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_info['gpu_cached'] = torch.cuda.memory_reserved() / 1024**3  # GB
            memory_info['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        # 可以添加系统内存监控
        try:
            import psutil
            memory_info['system_memory'] = psutil.virtual_memory().percent
        except ImportError:
            memory_info['system_memory'] = 0
        
        return memory_info
    
    def prepare_training(self, checkpoint_path=None):
        """准备训练"""
        try:
            print(f"🔧 正在准备训练环境...")
            
            # 验证数据目录
            required_dirs = [
                self.config['data']['train_lr_dir'],
                self.config['data']['train_hr_dir'],
                self.config['data']['val_lr_dir'],
                self.config['data']['val_hr_dir']
            ]
            
            missing_dirs = []
            for dir_path in required_dirs:
                if not os.path.exists(dir_path):
                    missing_dirs.append(dir_path)
            
            if missing_dirs:
                error_msg = f"以下数据目录不存在: {', '.join(missing_dirs)}"
                print(f"❌ {error_msg}")
                self._trigger_callback('on_error', error_msg)
                return False
            
            print("✅ 数据目录验证通过")
            
            # 创建输出目录
            os.makedirs(self.config['paths']['save_dir'], exist_ok=True)
            os.makedirs(self.config['paths']['log_dir'], exist_ok=True)
            print("✅ 输出目录创建完成")
            
            # 内存优化配置
            memory_config = self.config.get('memory', {})
            max_cache_size = memory_config.get('max_cache_size', 500)
            gradient_accumulation_steps = memory_config.get('gradient_accumulation_steps', 1)
            
            print(f"📊 内存配置: 缓存大小={max_cache_size}, 梯度累积步数={gradient_accumulation_steps}")
            
            # 创建内存优化训练器
            print("🚀 正在创建训练器...")
            self.trainer = MemoryEfficientTrainer(
                train_lr_dir=self.config['data']['train_lr_dir'],
                train_hr_dir=self.config['data']['train_hr_dir'],
                val_lr_dir=self.config['data']['val_lr_dir'],
                val_hr_dir=self.config['data']['val_hr_dir'],
                batch_size=self.config['training']['batch_size'],
                lr=self.config['training']['learning_rate'],
                device=self.config['training']['device'],
                max_cache_size=max_cache_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                num_blocks=self.config['model']['num_blocks'],
                num_features=self.config['model']['num_features']
            )
            print("✅ 训练器创建成功")
            
            # 如果提供了检查点路径，加载预训练模型
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"📂 正在加载检查点: {os.path.basename(checkpoint_path)}")
                
                # 使用改进的加载方法
                if self.trainer.load_checkpoint(checkpoint_path):
                    self.is_incremental = True
                    self.checkpoint_path = checkpoint_path
                    print(f"✅ 预训练模型加载成功")
                    print(f"📈 将从第 {self.trainer.epoch + 1} 个epoch开始增量训练")
                else:
                    error_msg = f"检查点加载失败: {checkpoint_path}"
                    print(f"❌ {error_msg}")
                    self._trigger_callback('on_error', error_msg)
                    return False
            
            print("🎉 训练准备完成")
            return True
            
        except Exception as e:
            error_msg = f"训练准备失败: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            self._trigger_callback('on_error', error_msg)
            return False
    
    def start_training(self, progress_callback=None, checkpoint_path=None):
        """开始训练"""
        if self.is_training:
            return False
            
        if not self.trainer:
            if not self.prepare_training(checkpoint_path):
                return False
        
        # 保存进度回调函数
        self.progress_callback = progress_callback
        
        self.is_training = True
        self.training_thread = threading.Thread(target=self._training_loop)
        self.training_thread.daemon = True
        self.training_thread.start()
        
        return True
    
    def start_incremental_training(self, checkpoint_path, progress_callback=None):
        """开始增量训练"""
        if not os.path.exists(checkpoint_path):
            self._trigger_callback('on_error', f"检查点文件不存在: {checkpoint_path}")
            return False
        
        # 为增量训练调整配置
        original_lr = self.config['training']['learning_rate']
        self.config['training']['learning_rate'] = original_lr * 0.1
        print(f"增量训练学习率调整为: {self.config['training']['learning_rate']}")
        print(f"增量训练将进行 {self.config['training']['num_epochs']} 个epoch")
        
        return self.start_training(progress_callback, checkpoint_path)
    
    def stop_training(self):
        """停止训练"""
        self.is_training = False
        if self.training_thread:
            self.training_thread.join(timeout=5)
    
    def _training_loop(self):
        """训练循环"""
        epoch = 0
        num_epochs = 0
        memory_after = {}
        
        try:
            self._trigger_callback('on_training_start')
            
            num_epochs = self.config['training']['num_epochs']
            save_frequency = self.config['training'].get('save_frequency', 5)
            
            # 如果是增量训练，从已有的epoch开始
            start_epoch = self.trainer.epoch if self.is_incremental else 0
            
            for epoch in range(start_epoch, start_epoch + num_epochs):
                if not self.is_training:
                    break
                    
                self._trigger_callback('on_epoch_start', epoch)
                
                start_time = time.time()
                
                # 获取训练前的内存状态
                memory_before = self.get_memory_info()
                
                # 创建进度回调函数，用于更新控制台训练器的batch信息
                def batch_progress_callback(progress_info):
                    # 更新控制台训练器的统计信息
                    if hasattr(self, 'console_trainer'):
                        self.console_trainer.training_stats['current_batch'] = progress_info['batch']
                        self.console_trainer.training_stats['total_batches'] = progress_info['total_batches']
                        self.console_trainer.training_stats['g_loss'] = progress_info['g_loss']
                        self.console_trainer.training_stats['d_loss'] = progress_info['d_loss']
                        self.console_trainer.training_stats['total_batches_processed'] += 1
                
                # 训练一个epoch，传递进度回调
                g_loss, d_loss = self.trainer.train_epoch(
                    epoch=epoch + 1, 
                    total_epochs=start_epoch + num_epochs,
                    progress_callback=batch_progress_callback
                )
                
                # 验证
                val_psnr = self.trainer.validate()
                
                # 获取训练后的内存状态
                memory_after = self.get_memory_info()
                
                epoch_time = time.time() - start_time
                
                # 记录历史
                self.training_history['epochs'].append(epoch + 1)
                self.training_history['train_g_loss'].append(g_loss)
                self.training_history['train_d_loss'].append(d_loss)
                self.training_history['val_psnr'].append(val_psnr)
                self.training_history['times'].append(epoch_time)
                self.training_history['memory_usage'].append(memory_after)
                
                # 保存检查点
                # 保存检查点
                is_best = val_psnr > self.trainer.best_psnr
                if is_best:
                    self.trainer.best_psnr = val_psnr
                
                if (epoch + 1) % save_frequency == 0 or is_best:
                    checkpoint_name = f'checkpoint_epoch_{epoch+1}.pth'
                    if self.is_incremental:
                        checkpoint_name = f'incremental_checkpoint_epoch_{epoch+1}.pth'
                    
                    checkpoint_path = os.path.join(
                        self.config['paths']['save_dir'], 
                        checkpoint_name
                    )
                    # 确保保存时epoch是正确的
                    self.trainer.epoch = epoch + 1
                    self.trainer.save_checkpoint(checkpoint_path, is_best)
                
                # 触发epoch结束回调
                epoch_info = {
                    'epoch': epoch + 1,
                    'total_epochs': start_epoch + num_epochs,
                    'g_loss': g_loss,
                    'd_loss': d_loss,
                    'val_psnr': val_psnr,
                    'time': epoch_time,
                    'is_best': is_best,
                    'memory_before': memory_before,
                    'memory_after': memory_after,
                    'is_incremental': self.is_incremental
                }
                
                self._trigger_callback('on_epoch_end', epoch_info)
                
                # 内存清理
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # 显示训练信息
                training_type = "增量训练" if self.is_incremental else "训练"
                print(f"{training_type} Epoch {epoch+1}/{start_epoch + num_epochs} - G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}, Val PSNR: {val_psnr:.2f}")
                
                # 触发进度更新回调
                if num_epochs > 0:
                    progress_percent = (epoch - start_epoch + 1) / num_epochs * 100
                    self._trigger_callback('on_progress_update', progress_percent)
                
                # 触发内存状态回调
                if self.memory_monitor_enabled:
                    self._trigger_callback('on_memory_status', memory_after)
                
                # 定期清理内存
                if (epoch + 1) % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # 训练完成后的回调
            self._trigger_callback('on_training_end', self.training_history)
            
        except Exception as e:
            self._trigger_callback('on_error', f"训练过程中发生错误: {str(e)}")
            print(f"训练错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_training = False
    
    def get_training_status(self):
        """获取训练状态"""
        # 计算当前实际的epoch
        current_epoch = 0
        total_epochs = 0
        
        if self.trainer and self.config:
            if self.is_incremental:
                # 增量训练：当前epoch = 检查点epoch + 已完成的新epoch数
                start_epoch = self.trainer.epoch
                completed_new_epochs = len(self.training_history['epochs'])
                current_epoch = start_epoch + completed_new_epochs
                total_epochs = start_epoch + self.config['training']['num_epochs']
            else:
                # 普通训练
                current_epoch = len(self.training_history['epochs'])
                total_epochs = self.config['training']['num_epochs']
        
        status = {
            'is_training': self.is_training,
            'current_epoch': current_epoch,
            'total_epochs': total_epochs,
            'best_psnr': self.trainer.best_psnr if self.trainer else 0,
            'history': self.training_history,
            'is_incremental': self.is_incremental,
            'start_epoch': self.trainer.epoch if self.trainer and self.is_incremental else 0
        }
        
        # 添加内存信息
        if self.memory_monitor_enabled:
            status['memory_info'] = self.get_memory_info()
        
        return status
    
    def save_training_log(self, filepath):
        """保存训练日志"""
        log_data = {
            'config': self.config,
            'history': self.training_history,
            'timestamp': datetime.now().isoformat(),
            'final_best_psnr': self.trainer.best_psnr if self.trainer else 0,
            'memory_optimization': True
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    def plot_training_curves(self, save_path=None):
        """绘制训练曲线"""
        if not self.training_history['epochs']:
            return None
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = self.training_history['epochs']
        
        # 生成器和判别器损失
        ax1.plot(epochs, self.training_history['train_g_loss'], 'b-', label='Generator Loss')
        ax1.plot(epochs, self.training_history['train_d_loss'], 'r-', label='Discriminator Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Losses')
        ax1.legend()
        ax1.grid(True)
        
        # 验证PSNR
        ax2.plot(epochs, self.training_history['val_psnr'], 'g-', label='Validation PSNR')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('PSNR (dB)')
        ax2.set_title('Validation PSNR')
        ax2.legend()
        ax2.grid(True)
        
        # 训练时间
        ax3.plot(epochs, self.training_history['times'], 'm-', label='Epoch Time')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Training Time per Epoch')
        ax3.legend()
        ax3.grid(True)
        
        # 内存使用情况
        if self.training_history['memory_usage']:
            gpu_memory = [mem.get('gpu_allocated', 0) for mem in self.training_history['memory_usage']]
            ax4.plot(epochs, gpu_memory, 'c-', label='GPU Memory (GB)')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Memory (GB)')
            ax4.set_title('GPU Memory Usage')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线已保存到: {save_path}")
        
        return fig


# 保持向后兼容性
TrainingManager = MemoryOptimizedTrainingManager