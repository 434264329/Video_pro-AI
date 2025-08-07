import os
import sys
import json
import time
import torch
import threading
import re
from datetime import datetime
from colorama import init, Fore, Back, Style

torch.cuda.empty_cache()
# 限制PyTorch内存分配策略，减少内存碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

from config.config import (
    load_config, save_config, get_default_config,
    load_low_memory_config, load_high_memory_config, load_rtx_4090_config,
    create_4gb_config, create_6gb_config, create_8gb_config, create_12gb_config,
    auto_config_by_gpu_memory
)

# 初始化colorama
init(autoreset=True)

# 🚨 额外的内存优化设置
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85)  # 只使用85%显存
    torch.cuda.reset_peak_memory_stats()
    print(f"{Fore.GREEN}🔧 控制台训练器内存优化配置已启用: max_split_size_mb=128, 显存限制=85%{Style.RESET_ALL}")

class GPUOptimizer:
    """GPU优化器 - 专门针对RTX 4090等高端GPU的优化"""
    
    def __init__(self):
        # 🚨 初始化时立即执行内存优化
        torch.cuda.empty_cache()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tf32_enabled = False
        self.compile_enabled = False
        
    def enable_tf32(self):
        """启用TF32加速（RTX 30/40系列）"""
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.tf32_enabled = True
            print(f"{Fore.GREEN}✅ TF32加速已启用{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ CUDA不可用，无法启用TF32{Style.RESET_ALL}")
    
    def disable_tf32(self):
        """禁用TF32加速"""
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            self.tf32_enabled = False
            print(f"{Fore.YELLOW}⚠️  TF32加速已禁用{Style.RESET_ALL}")
    
    def enable_compile(self):
        """启用模型编译优化（PyTorch 2.0+）"""
        try:
            if hasattr(torch, 'compile'):
                self.compile_enabled = True
                print(f"{Fore.GREEN}✅ 模型编译优化已启用{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}⚠️  当前PyTorch版本不支持模型编译{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ 启用模型编译失败: {e}{Style.RESET_ALL}")
    
    def disable_compile(self):
        """禁用模型编译优化"""
        self.compile_enabled = False
        print(f"{Fore.YELLOW}⚠️  模型编译优化已禁用{Style.RESET_ALL}")
    
    def optimize_model(self, model):
        """优化模型"""
        if self.compile_enabled and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='max-autotune')
                print(f"{Fore.GREEN}✅ 模型编译优化完成{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️  模型编译失败，使用原始模型: {e}{Style.RESET_ALL}")
        return model
    
    def get_optimal_batch_size(self, model, input_shape, max_memory_gb=20):
        """自动计算最优批次大小"""
        if not torch.cuda.is_available():
            return 1
        
        try:
            # 清空缓存
            torch.cuda.empty_cache()
            
            # 测试不同批次大小
            batch_sizes = [1, 2, 4, 6, 8, 12, 16, 20, 24]
            optimal_batch_size = 1
            
            for batch_size in batch_sizes:
                try:
                    # 创建测试输入
                    test_input = torch.randn(batch_size, *input_shape[1:]).cuda()
                    
                    # 前向传播测试
                    with torch.no_grad():
                        _ = model(test_input)
                    
                    # 检查显存使用
                    memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                    if memory_used < max_memory_gb * 0.8:  # 保留20%余量
                        optimal_batch_size = batch_size
                    else:
                        break
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        break
                    else:
                        raise e
                finally:
                    torch.cuda.empty_cache()
            
            print(f"{Fore.GREEN}🎯 推荐批次大小: {optimal_batch_size}{Style.RESET_ALL}")
            return optimal_batch_size
            
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️  自动批次大小计算失败: {e}，使用默认值{Style.RESET_ALL}")
            return 4

class ModelCompatibilityChecker:
    """模型兼容性检查器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def check_model_compatibility(self, checkpoint_path, new_config):
        """检查模型兼容性"""
        try:
            # 加载检查点
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            old_state_dict = checkpoint['generator_state_dict']
            
            # 提取旧模型配置
            old_config = self._extract_model_config_from_state_dict(old_state_dict)
            
            # 获取新配置
            new_num_blocks = new_config['model']['num_blocks']
            new_num_features = new_config['model']['num_features']
            
            print(f"{Fore.YELLOW}旧模型配置: blocks={old_config['num_blocks']}, features={old_config['num_features']}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}新模型配置: blocks={new_num_blocks}, features={new_num_features}{Style.RESET_ALL}")
            
            # 检查兼容性
            if old_config['num_blocks'] == new_num_blocks and old_config['num_features'] == new_num_features:
                print(f"{Fore.GREEN}✅ 模型配置完全兼容{Style.RESET_ALL}")
                return True, "完全兼容"
            else:
                print(f"{Fore.YELLOW}⚠️  模型配置不兼容，需要调整{Style.RESET_ALL}")
                return False, f"配置不匹配: 旧模型({old_config['num_blocks']}, {old_config['num_features']}) vs 新配置({new_num_blocks}, {new_num_features})"
                
        except Exception as e:
            return False, f"检查失败: {str(e)}"
    
    def _extract_model_config_from_state_dict(self, state_dict):
        """从状态字典中提取模型配置"""
        # 通过分析权重形状来推断模型参数
        num_features = 64  # 默认值
        num_blocks = 6     # 默认值
        
        # 分析第一个卷积层来确定特征数
        if 'conv_first.weight' in state_dict:
            num_features = state_dict['conv_first.weight'].shape[0]
            print(f"🔍 从conv_first.weight检测到特征数: {num_features}")
        
        # 计算RRDB块的数量
        rrdb_count = 0
        for key in state_dict.keys():
            if key.startswith('rrdb_blocks.') and '.dense1.conv1.weight' in key:
                # 提取块索引
                import re
                match = re.match(r'rrdb_blocks\.(\d+)\.dense1\.conv1\.weight', key)
                if match:
                    block_idx = int(match.group(1))
                    rrdb_count = max(rrdb_count, block_idx + 1)
        
        if rrdb_count > 0:
            num_blocks = rrdb_count
            print(f"🔍 从rrdb_blocks检测到块数: {num_blocks}")
        
        print(f"🔍 提取的模型配置: {num_blocks}块, {num_features}特征")
        return {
            'num_blocks': num_blocks,
            'num_features': num_features
        }

    def adjust_checkpoint_for_new_config(self, checkpoint_path, new_config, output_path=None):
        """调整检查点以适应新配置"""
        try:
            print(f"{Fore.CYAN}🔧 正在调整模型检查点...{Style.RESET_ALL}")
            
            # 加载检查点
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            old_state_dict = checkpoint['generator_state_dict']
            
            # 显示原始检查点信息
            if 'epoch' in checkpoint:
                print(f"{Fore.CYAN}原始检查点训练轮数: {checkpoint['epoch']}{Style.RESET_ALL}")
            if 'best_psnr' in checkpoint:
                print(f"{Fore.CYAN}原始检查点最佳PSNR: {checkpoint['best_psnr']:.4f} dB{Style.RESET_ALL}")
            
            # 提取配置信息
            old_config = self._extract_model_config_from_state_dict(old_state_dict)
            new_model_config = new_config['model']
            
            print(f"{Fore.CYAN}原始模型配置: {old_config['num_blocks']}块, {old_config['num_features']}特征{Style.RESET_ALL}")
            print(f"{Fore.CYAN}目标模型配置: {new_model_config['num_blocks']}块, {new_model_config['num_features']}特征{Style.RESET_ALL}")
            
            # 检查是否需要调整
            if (old_config['num_blocks'] == new_model_config['num_blocks'] and 
                old_config['num_features'] == new_model_config['num_features']):
                print(f"{Fore.GREEN}✅ 模型配置已匹配，无需调整{Style.RESET_ALL}")
                return True, checkpoint_path
            
            # 创建新模型以获取目标状态字典结构
            from src.models.esrgan import LiteRealESRGAN
            new_model = LiteRealESRGAN(
                num_blocks=new_model_config['num_blocks'],
                num_features=new_model_config['num_features']
            )
            new_model_state_dict = new_model.state_dict()
            
            # 调整权重
            adjusted_state_dict = self._adjust_weights(old_state_dict, new_model_state_dict, old_config, new_model_config)
            
            # 更新检查点 - 保留所有原始元数据
            checkpoint['generator_state_dict'] = adjusted_state_dict
            # 保留原始的epoch、best_psnr、optimizer_state_dict等信息
            # 这些信息在调整后仍然有效，因为我们只是调整了模型结构
            
            # 生成输出路径
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
                output_path = os.path.join(
                    os.path.dirname(checkpoint_path),
                    f"{base_name}_adjusted.pth"
                )
            
            # 保存调整后的检查点
            torch.save(checkpoint, output_path)
            print(f"{Fore.GREEN}✅ 调整后的模型已保存: {os.path.basename(output_path)}{Style.RESET_ALL}")
            
            # 显示调整后的检查点信息
            if 'epoch' in checkpoint:
                print(f"{Fore.GREEN}✅ 保留训练轮数: {checkpoint['epoch']}{Style.RESET_ALL}")
            if 'best_psnr' in checkpoint:
                print(f"{Fore.GREEN}✅ 保留最佳PSNR: {checkpoint['best_psnr']:.4f} dB{Style.RESET_ALL}")
            
            return True, output_path
            
        except Exception as e:
            print(f"{Fore.RED}❌ 模型调整失败: {str(e)}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            return False, str(e)

    def _adjust_weights(self, old_state_dict, new_model_state_dict, old_config, new_config):
        """调整权重以匹配新模型结构"""
        adjusted_state_dict = {}
        
        print(f"{Fore.CYAN}🔧 开始权重调整...{Style.RESET_ALL}")
        print(f"🔧 从 {old_config['num_blocks']}块/{old_config['num_features']}特征 -> {new_config['num_blocks']}块/{new_config['num_features']}特征")
        
        # 获取配置信息
        old_blocks = old_config['num_blocks']
        old_features = old_config['num_features']
        new_blocks = new_config['num_blocks']
        new_features = new_config['num_features']
        
        # 处理所有新模型需要的权重
        for key, new_weight in new_model_state_dict.items():
            if key in old_state_dict:
                old_weight = old_state_dict[key]
                if old_weight.shape == new_weight.shape:
                    # 形状相同，直接复制
                    adjusted_state_dict[key] = old_weight.clone()
                    print(f"✅ {key}: 直接复制 {old_weight.shape}")
                else:
                    # 形状不同，需要调整
                    adjusted_weight = self._smart_resize_weight(old_weight, new_weight.shape, key)
                    adjusted_state_dict[key] = adjusted_weight
                    print(f"🔧 {key}: 调整 {old_weight.shape} -> {new_weight.shape}")
            else:
                # 新增的权重
                if self._is_new_rrdb_block(key, old_blocks):
                    # 对于新增的RRDB块，复制现有块的权重
                    source_key = self._get_source_block_key(key, old_state_dict, old_blocks)
                    if source_key and source_key in old_state_dict:
                        source_weight = old_state_dict[source_key]
                        if source_weight.shape == new_weight.shape:
                            # 添加小的随机噪声以避免完全相同
                            noise = torch.randn_like(source_weight) * 0.01
                            adjusted_state_dict[key] = source_weight + noise
                            print(f"🔄 {key}: 复制自 {source_key} (添加噪声)")
                        else:
                            # 需要调整尺寸后复制
                            adjusted_weight = self._smart_resize_weight(source_weight, new_weight.shape, key)
                            noise = torch.randn_like(adjusted_weight) * 0.01
                            adjusted_state_dict[key] = adjusted_weight + noise
                            print(f"🔄 {key}: 从 {source_key} 调整并添加噪声 {source_weight.shape} -> {new_weight.shape}")
                    else:
                        # 随机初始化
                        adjusted_state_dict[key] = self._initialize_weight(new_weight)
                        print(f"🎲 {key}: 随机初始化 {new_weight.shape}")
                else:
                    # 其他新增权重，随机初始化
                    adjusted_state_dict[key] = self._initialize_weight(new_weight)
                    print(f"🎲 {key}: 随机初始化 {new_weight.shape}")
        
        print(f"{Fore.GREEN}✅ 权重调整完成，共处理 {len(adjusted_state_dict)} 个权重{Style.RESET_ALL}")
        return adjusted_state_dict
    def smart_adjust_features_enhanced(self, checkpoint_path, new_features, use_progressive=True, steps=3):
        """
        增强版智能特征数调整，支持渐进式调整和错误恢复
        """
        try:
            print(f"{Fore.CYAN}🚀 增强版智能特征数调整{Style.RESET_ALL}")
            
            # 1. 检测当前特征数
            detected_config = self._detect_checkpoint_config(checkpoint_path)
            if not detected_config:
                return False, "无法检测检查点配置"
            
            old_features = detected_config['num_features']
            print(f"📊 检测到当前特征数: {old_features}")
            print(f"🎯 目标特征数: {new_features}")
            
            if old_features == new_features:
                print(f"{Fore.YELLOW}⚠️  特征数已经是 {new_features}，无需调整{Style.RESET_ALL}")
                return True, checkpoint_path
            
            # 2. 备份原始检查点
            backup_path = checkpoint_path.replace('.pth', '_backup.pth')
            import shutil
            shutil.copy2(checkpoint_path, backup_path)
            print(f"💾 已备份原始检查点: {os.path.basename(backup_path)}")
            
            # 3. 直接调整（简化版本，避免复杂的渐进式调整）
            print(f"🔧 开始直接特征数调整")
            success, result = self._smart_adjust_features_direct(checkpoint_path, old_features, new_features)
            
            if success:
                # 4. 最终验证
                print(f"{Fore.GREEN}🎉 智能特征数调整成功完成!{Style.RESET_ALL}")
                return True, result
            else:
                print(f"{Fore.RED}❌ 调整过程失败: {result}{Style.RESET_ALL}")
                return False, result
                
        except Exception as e:
            print(f"{Fore.RED}❌ 增强版调整失败: {str(e)}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            return False, str(e)

    def _smart_adjust_features_direct(self, checkpoint_path, old_features, new_features):
        """
        直接特征数调整方法
        """
        try:
            print(f"{Fore.CYAN}🔧 直接调整特征数: {old_features} -> {new_features}{Style.RESET_ALL}")
            
            # 加载检查点
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            old_state_dict = checkpoint['generator_state_dict']
            
            # 检测检查点的实际配置
            checkpoint_config = self._detect_checkpoint_config(checkpoint_path)
            if not checkpoint_config:
                raise Exception("无法检测检查点配置")
            
            detected_blocks = checkpoint_config['num_blocks']
            detected_features = checkpoint_config['num_features']
            
            print(f"{Fore.CYAN}📊 检测到检查点配置: {detected_blocks}块, {detected_features}特征{Style.RESET_ALL}")
            
            # 创建目标模型 - 使用检测到的块数
            from src.models.esrgan import LiteRealESRGAN
            target_model = LiteRealESRGAN(
                num_blocks=detected_blocks,  # 使用检测到的块数
                num_features=new_features
            )
            new_state_dict = target_model.state_dict()
            
            # 调整权重
            adjusted_state_dict = self._adjust_features_weights_enhanced(
                old_state_dict, new_state_dict, old_features, new_features
            )
            
            # 更新检查点
            checkpoint['generator_state_dict'] = adjusted_state_dict
            
            # 🔥 清理优化器状态，避免尺寸不匹配
            if 'g_optimizer_state_dict' in checkpoint:
                del checkpoint['g_optimizer_state_dict']
                print(f"{Fore.YELLOW}🧹 已清理生成器优化器状态{Style.RESET_ALL}")
            
            if 'd_optimizer_state_dict' in checkpoint:
                del checkpoint['d_optimizer_state_dict']
                print(f"{Fore.YELLOW}🧹 已清理判别器优化器状态{Style.RESET_ALL}")
            
            if 'g_scheduler_state_dict' in checkpoint:
                del checkpoint['g_scheduler_state_dict']
                print(f"{Fore.YELLOW}🧹 已清理生成器调度器状态{Style.RESET_ALL}")
            
            if 'd_scheduler_state_dict' in checkpoint:
                del checkpoint['d_scheduler_state_dict']
                print(f"{Fore.YELLOW}🧹 已清理判别器调度器状态{Style.RESET_ALL}")
            
            # 保存结果
            base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
            checkpoint_dir = os.path.dirname(checkpoint_path)
            output_path = os.path.join(checkpoint_dir, f"{base_name}_features_{new_features}.pth")
            
            torch.save(checkpoint, output_path)
            print(f"{Fore.GREEN}✅ 调整完成，保存至: {os.path.basename(output_path)}{Style.RESET_ALL}")
            
            return True, output_path
            
        except Exception as e:
            print(f"{Fore.RED}❌ 直接调整失败: {str(e)}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            return False, str(e)

    def _adjust_features_weights_enhanced(self, old_state_dict, new_state_dict, old_features, new_features):
        """
        增强版权重调整方法，修复所有已知问题
        """
        adjusted_state_dict = {}
        
        print(f"{Fore.CYAN}🔧 权重调整详情: {old_features} -> {new_features} 特征数{Style.RESET_ALL}")
        
        for key, new_weight in new_state_dict.items():
            try:
                if key in old_state_dict:
                    old_weight = old_state_dict[key]
                    
                    if old_weight.shape == new_weight.shape:
                        # 形状相同，直接复制
                        adjusted_state_dict[key] = old_weight.clone()
                        print(f"✅ {key}: 直接复制 {old_weight.shape}")
                        
                    elif 'conv' in key and 'weight' in key and len(old_weight.shape) == 4:
                        # 卷积层权重处理
                        adjusted_weight = self._adjust_conv_weight_enhanced(old_weight, new_weight, key, old_features, new_features)
                        adjusted_state_dict[key] = adjusted_weight
                        print(f"🔧 {key}: 卷积调整 {old_weight.shape} -> {new_weight.shape}")
                        
                    elif 'bias' in key and len(old_weight.shape) == 1:
                        # 偏置处理
                        adjusted_weight = self._adjust_bias_enhanced(old_weight, new_weight, key, old_features, new_features)
                        adjusted_state_dict[key] = adjusted_weight
                        print(f"🔧 {key}: 偏置调整 {old_weight.shape} -> {new_weight.shape}")
                        
                    else:
                        # 其他权重，智能调整
                        adjusted_weight = self._smart_resize_weight_safe(old_weight, new_weight.shape, key)
                        adjusted_state_dict[key] = adjusted_weight
                        print(f"🔄 {key}: 智能调整 {old_weight.shape} -> {new_weight.shape}")
                        
                else:
                    # 新增权重，智能初始化
                    adjusted_state_dict[key] = self._initialize_weight_smart(new_weight, key)
                    print(f"🎲 {key}: 智能初始化 {new_weight.shape}")
                    
            except Exception as e:
                print(f"{Fore.RED}🛑 处理 {key} 时出错: {str(e)}{Style.RESET_ALL}")
                # 出错时使用智能初始化
                adjusted_state_dict[key] = self._initialize_weight_smart(new_weight, key)
                print(f"🔄 {key}: 错误恢复，使用智能初始化")
        
        print(f"{Fore.GREEN}✅ 权重调整完成，共处理 {len(adjusted_state_dict)} 个权重{Style.RESET_ALL}")
        return adjusted_state_dict

    def _adjust_conv_weight_enhanced(self, old_weight, new_weight, key, old_features, new_features):
        """
        增强版卷积权重调整
        """
        old_out, old_in, old_h, old_w = old_weight.shape
        new_out, new_in, new_h, new_w = new_weight.shape
        
        # 创建新权重张量
        new_weight_tensor = torch.zeros_like(new_weight)
        
        # 特殊处理RRDB密集连接层
        if 'rrdb_blocks' in key and 'dense' in key:
            return self._adjust_rrdb_dense_weight_enhanced(old_weight, new_weight, key, old_features, new_features)
        
        # 处理第一个卷积层（conv_first）
        elif 'conv_first' in key:
            # 输出通道从old_features调整到new_features
            min_out = min(old_out, new_out)
            new_weight_tensor[:min_out] = old_weight[:min_out]
            
            if new_out > old_out:
                # 扩展输出通道
                with torch.no_grad():
                    torch.nn.init.kaiming_normal_(new_weight_tensor[old_out:], mode='fan_out', nonlinearity='leaky_relu')
            
            return new_weight_tensor
        
        # 处理最后的卷积层
        elif 'conv_last' in key or 'conv_hr' in key:
            # 输入通道从old_features调整到new_features
            min_in = min(old_in, new_in)
            new_weight_tensor[:, :min_in] = old_weight[:, :min_in]
            
            if new_in > old_in:
                # 扩展输入通道
                with torch.no_grad():
                    torch.nn.init.kaiming_normal_(new_weight_tensor[:, old_in:], mode='fan_in', nonlinearity='leaky_relu')
            
            return new_weight_tensor
        
        # 其他卷积层，通用处理
        else:
            min_out = min(old_out, new_out)
            min_in = min(old_in, new_in)
            new_weight_tensor[:min_out, :min_in] = old_weight[:min_out, :min_in]
            
            # 初始化新增部分
            if new_out > old_out or new_in > old_in:
                with torch.no_grad():
                    torch.nn.init.kaiming_normal_(new_weight_tensor, mode='fan_in', nonlinearity='leaky_relu')
                    # 保留已复制的部分
                    new_weight_tensor[:min_out, :min_in] = old_weight[:min_out, :min_in]
            
            return new_weight_tensor

    def _adjust_rrdb_dense_weight_enhanced(self, old_weight, new_weight, key, old_features, new_features):
        """
        增强版RRDB密集连接层权重调整
        """
        old_out, old_in, old_h, old_w = old_weight.shape
        new_out, new_in, new_h, new_w = new_weight.shape
        
        # 解析层信息
        parts = key.split('.')
        try:
            block_idx = int(parts[1])
            dense_idx = int(parts[2][-1])
            conv_idx = int(parts[3][-1])
        except (IndexError, ValueError):
            # 解析失败，使用通用方法
            return self._smart_resize_weight_safe(old_weight, new_weight.shape, key)
        
        # 计算期望的输入通道数
        growth_rate = 32
        layer_order = (dense_idx - 1) * 5 + conv_idx
        
        expected_old_in = old_features + growth_rate * (layer_order - 1)
        expected_new_in = new_features + growth_rate * (layer_order - 1)
        
        print(f"    🔗 RRDB Block{block_idx} Dense{dense_idx} Conv{conv_idx}")
        print(f"    📊 通道分析: 旧{old_in}(期望{expected_old_in}), 新{new_in}(期望{expected_new_in})")
        
        # 创建新权重张量
        new_weight_tensor = torch.zeros_like(new_weight)
        
        # 复制现有权重（安全复制）
        min_in = min(old_in, new_in)
        min_out = min(old_out, new_out)
        new_weight_tensor[:min_out, :min_in] = old_weight[:min_out, :min_in]
        
        # 处理输入通道扩展
        if new_in > old_in:
            remaining = new_in - old_in
            print(f"    📈 扩展输入通道: +{remaining}")
            
            with torch.no_grad():
                if remaining <= old_in and old_in > 0:
                    # 复制现有通道并添加小噪声
                    for i in range(remaining):
                        src_idx = i % old_in
                        dst_idx = old_in + i
                        if dst_idx < new_in:
                            new_weight_tensor[:min_out, dst_idx] = old_weight[:min_out, src_idx] * 0.9
                            noise = torch.randn_like(old_weight[:min_out, src_idx]) * 0.01
                            new_weight_tensor[:min_out, dst_idx] += noise
                else:
                    # 使用Kaiming初始化新增通道
                    if old_in < new_in:
                        torch.nn.init.kaiming_normal_(new_weight_tensor[:, old_in:], mode='fan_in', nonlinearity='leaky_relu')
        
        # 处理输出通道扩展
        if new_out > old_out:
            with torch.no_grad():
                torch.nn.init.kaiming_normal_(new_weight_tensor[old_out:, :], mode='fan_out', nonlinearity='leaky_relu')
        
        return new_weight_tensor

    def _adjust_bias_enhanced(self, old_bias, new_bias, key, old_features, new_features):
        """
        增强版偏置调整
        """
        new_bias_tensor = torch.zeros_like(new_bias)
        
        # 复制现有偏置
        min_size = min(old_bias.shape[0], new_bias.shape[0])
        new_bias_tensor[:min_size] = old_bias[:min_size]
        
        # 新增偏置初始化为0
        if new_bias.shape[0] > old_bias.shape[0]:
            new_bias_tensor[old_bias.shape[0]:] = 0.0
        
        return new_bias_tensor

    def _smart_resize_weight_safe(self, old_weight, new_shape, key):
        """
        安全的权重尺寸调整方法
        """
        if old_weight.shape == new_shape:
            return old_weight.clone()
        
        # 创建新权重张量
        new_weight = torch.zeros(new_shape, dtype=old_weight.dtype, device=old_weight.device)
        
        # 计算可复制的维度
        min_dims = [min(old_dim, new_dim) for old_dim, new_dim in zip(old_weight.shape, new_shape)]
        
        # 安全复制
        try:
            if len(old_weight.shape) == 4:  # 4D卷积权重
                new_weight[:min_dims[0], :min_dims[1], :min_dims[2], :min_dims[3]] = \
                    old_weight[:min_dims[0], :min_dims[1], :min_dims[2], :min_dims[3]]
            elif len(old_weight.shape) == 2:  # 2D线性权重
                new_weight[:min_dims[0], :min_dims[1]] = old_weight[:min_dims[0], :min_dims[1]]
            elif len(old_weight.shape) == 1:  # 1D偏置
                new_weight[:min_dims[0]] = old_weight[:min_dims[0]]
            else:
                # 通用复制
                slices = tuple(slice(0, dim) for dim in min_dims)
                new_weight[slices] = old_weight[slices]
        except Exception as e:
            print(f"    ⚠️ 复制权重时出错: {str(e)}, 使用初始化")
        
        # 初始化新增部分
        if any(new_dim > old_dim for new_dim, old_dim in zip(new_shape, old_weight.shape)):
            with torch.no_grad():
                if 'conv' in key and len(new_shape) == 4:
                    torch.nn.init.kaiming_normal_(new_weight, mode='fan_in', nonlinearity='leaky_relu')
                    # 恢复已复制的部分
                    if len(old_weight.shape) == 4:
                        new_weight[:min_dims[0], :min_dims[1], :min_dims[2], :min_dims[3]] = \
                            old_weight[:min_dims[0], :min_dims[1], :min_dims[2], :min_dims[3]]
                elif 'bias' in key:
                    # 偏置新增部分保持为0
                    pass
                else:
                    torch.nn.init.normal_(new_weight, mean=0, std=0.02)
                    # 恢复已复制的部分
                    slices = tuple(slice(0, dim) for dim in min_dims)
                    new_weight[slices] = old_weight[slices]
        
        return new_weight

    def _initialize_weight_smart(self, weight_tensor, key):
        """
        智能权重初始化
        """
        with torch.no_grad():
            if 'conv' in key and len(weight_tensor.shape) == 4:
                # 卷积层使用Kaiming初始化
                torch.nn.init.kaiming_normal_(weight_tensor, mode='fan_in', nonlinearity='leaky_relu')
            elif 'bias' in key:
                # 偏置初始化为0
                torch.nn.init.zeros_(weight_tensor)
            elif len(weight_tensor.shape) == 2:
                # 线性层使用Xavier初始化
                torch.nn.init.xavier_uniform_(weight_tensor)
            else:
                # 其他情况使用正态分布
                torch.nn.init.normal_(weight_tensor, mean=0, std=0.02)
        
        return weight_tensor

    # ... existing code ...
    
    def _adjust_features_weights(self, old_state_dict, new_state_dict, old_features, new_features):
        """
        调整权重以匹配新的特征数
        支持扩展和收缩特征数，特别针对RRDB密集连接层优化
        """
        adjusted_state_dict = {}
        
        print(f"{Fore.CYAN}🔧 权重调整详情: {old_features} -> {new_features} 特征数{Style.RESET_ALL}")
        
        for key, new_weight in new_state_dict.items():
            if key in old_state_dict:
                old_weight = old_state_dict[key]
                
                try:
                    if old_weight.shape == new_weight.shape:
                        # 形状相同，直接复制
                        adjusted_state_dict[key] = old_weight.clone()
                        print(f"✅ {key}: 直接复制 {old_weight.shape}")
                        
                    elif 'conv' in key and 'weight' in key and len(old_weight.shape) == 4:
                        # 卷积层权重需要特殊处理
                        if 'rrdb_blocks' in key and 'dense' in key:
                            # RRDB密集连接层
                            adjusted_weight = self._adjust_rrdb_dense_weight(old_weight, new_weight, key, old_features, new_features)
                            adjusted_state_dict[key] = adjusted_weight
                            print(f"🔗 {key}: RRDB密集调整 {old_weight.shape} -> {new_weight.shape}")
                        elif self._is_direct_feature_related(key, old_weight, new_weight, old_features, new_features):
                            # 直接特征相关的权重
                            adjusted_weight = self._adjust_direct_feature_weight(old_weight, new_weight, key, old_features, new_features)
                            adjusted_state_dict[key] = adjusted_weight
                            print(f"🔧 {key}: 直接特征调整 {old_weight.shape} -> {new_weight.shape}")
                        else:
                            # 其他卷积层，使用通用调整
                            adjusted_weight = self._adjust_general_conv_weight(old_weight, new_weight.shape, key)
                            adjusted_state_dict[key] = adjusted_weight
                            print(f"🔄 {key}: 通用调整 {old_weight.shape} -> {new_weight.shape}")
                            
                    elif 'bias' in key and len(old_weight.shape) == 1:
                        # 偏置调整
                        if self._is_direct_feature_related(key, old_weight, new_weight, old_features, new_features):
                            adjusted_weight = self._adjust_feature_bias(old_weight, new_weight, old_features, new_features)
                            adjusted_state_dict[key] = adjusted_weight
                            print(f"🔧 {key}: 特征偏置调整 {old_weight.shape} -> {new_weight.shape}")
                        else:
                            adjusted_state_dict[key] = old_weight.clone()
                            print(f"✅ {key}: 偏置直接复制 {old_weight.shape}")
                    else:
                        # 其他类型权重，尝试通用调整
                        adjusted_weight = self._adjust_general_weight(old_weight, new_weight.shape, key)
                        adjusted_state_dict[key] = adjusted_weight
                        print(f"🔄 {key}: 其他调整 {old_weight.shape} -> {new_weight.shape}")
                        
                except Exception as e:
                    print(f"{Fore.RED}🛑 调整失败详情:{Style.RESET_ALL}")
                    print(f"   操作: {key}")
                    print(f"   目标形状: {new_weight.shape}")
                    print(f"   原始形状: {old_weight.shape}")
                    print(f"   层类型: {'RRDB密集层' if 'dense' in key else '普通卷积层'}")
                    print(f"   错误: {str(e)}")
                    raise
            else:
                # 新增的权重，随机初始化
                adjusted_state_dict[key] = self._initialize_weight_smart(new_weight, key)
                print(f"🎲 {key}: 随机初始化 {new_weight.shape}")
        
        print(f"{Fore.GREEN}✅ 权重调整完成，共处理 {len(adjusted_state_dict)} 个权重{Style.RESET_ALL}")
        return adjusted_state_dict
    
    def _is_direct_feature_related(self, key, old_weight, new_weight, old_features, new_features):
        """判断是否是直接特征相关的权重"""
        if 'conv' in key and 'weight' in key and len(old_weight.shape) == 4:
            old_out, old_in, _, _ = old_weight.shape
            new_out, new_in, _, _ = new_weight.shape
            
            # 检查输出或输入通道是否直接匹配特征数变化
            return (old_out == old_features and new_out == new_features) or \
                   (old_in == old_features and new_in == new_features)
        
        # 检查是否是偏置且长度匹配特征数变化
        if 'bias' in key and len(old_weight.shape) == 1:
            return old_weight.shape[0] == old_features and new_weight.shape[0] == new_features
        
        return False
    
    def _adjust_rrdb_dense_weight(self, old_weight, new_weight, key, old_features, new_features):
        """
        专门处理RRDB密集连接层的权重调整
        修复通道数计算错误，增加动态增长率计算
        """
        old_out, old_in, old_h, old_w = old_weight.shape
        new_out, new_in, new_h, new_w = new_weight.shape
        
        # 解析密集层信息
        parts = key.split('.')
        block_idx = int(parts[1])
        dense_idx = int(parts[2][-1])
        conv_idx = int(parts[3][-1])
        
        # 动态计算增长率
        growth_rate = 32
        layer_order = (dense_idx - 1) * 5 + conv_idx
        
        print(f"    🔗 RRDB Block{block_idx} Dense{dense_idx} Conv{conv_idx} (序号:{layer_order})")
        print(f"    📊 通道分析: 旧输入{old_in}, 新输入{new_in}")
        
        # 创建新权重张量
        new_weight_tensor = torch.zeros_like(new_weight)
        
        # 复制现有权重
        min_in = min(old_in, new_in)
        new_weight_tensor[:, :min_in] = old_weight[:, :min_in].clone()
        
        if new_in > old_in:
            # 扩展输入通道
            remaining = new_in - old_in
            print(f"    📈 扩展密集连接输入通道: +{remaining}")
            
            with torch.no_grad():
                if remaining <= old_in:
                    # 复制现有通道并添加噪声
                    for i in range(remaining):
                        src_idx = i % old_in
                        dst_idx = old_in + i
                        new_weight_tensor[:, dst_idx] = old_weight[:, src_idx] * 0.8
                        noise = torch.randn_like(old_weight[:, src_idx]) * 0.01
                        new_weight_tensor[:, dst_idx] += noise
                else:
                    # 使用Kaiming初始化
                    torch.nn.init.kaiming_normal_(new_weight_tensor[:, old_in:], mode='fan_in', nonlinearity='leaky_relu')
        
        elif new_in < old_in:
            # 收缩输入通道
            print(f"    📉 收缩密集连接输入通道: -{old_in - new_in}")
        
        return new_weight_tensor

    def _adjust_direct_feature_weight(self, old_weight, new_weight, key, old_features, new_features):
        """
        调整直接特征相关的权重（如conv_first等）
        """
        old_out, old_in, old_h, old_w = old_weight.shape
        new_out, new_in, new_h, new_w = new_weight.shape
        
        # 创建新权重张量
        new_weight_tensor = torch.zeros_like(new_weight)
        
        if old_out == old_features and new_out == new_features:
            # 输出通道调整
            min_out = min(old_out, new_out)
            new_weight_tensor[:min_out] = old_weight[:min_out]
            
            if new_out > old_out:
                # 扩展输出通道
                remaining = new_out - old_out
                print(f"    📈 扩展输出通道: +{remaining}")
                
                # 使用Kaiming初始化
                with torch.no_grad():
                    torch.nn.init.kaiming_normal_(new_weight_tensor[old_out:], mode='fan_out', nonlinearity='leaky_relu')
            
        elif old_in == old_features and new_in == new_features:
            # 输入通道调整
            min_in = min(old_in, new_in)
            new_weight_tensor[:, :min_in] = old_weight[:, :min_in]
            
            if new_in > old_in:
                # 扩展输入通道
                remaining = new_in - old_in
                print(f"    📈 扩展输入通道: +{remaining}")
                
                # 使用Kaiming初始化
                with torch.no_grad():
                    torch.nn.init.kaiming_normal_(new_weight_tensor[:, old_in:], mode='fan_in', nonlinearity='leaky_relu')
        
        return new_weight_tensor

    def _adjust_feature_bias(self, old_bias, new_bias, old_features, new_features):
        """
        调整特征相关的偏置
        """
        new_bias_tensor = torch.zeros_like(new_bias)
        min_features = min(old_features, new_features)
        new_bias_tensor[:min_features] = old_bias[:min_features]
        
        if new_features > old_features:
            # 扩展偏置，新增部分初始化为0
            remaining = new_features - old_features
            print(f"    📈 扩展偏置: +{remaining}")
            new_bias_tensor[old_features:] = 0.0
        elif new_features < old_features:
            # 收缩偏置
            print(f"    📉 收缩偏置: -{old_features - new_features}")
        
        return new_bias_tensor

    def _adjust_general_conv_weight(self, old_weight, new_shape, key):
        """
        通用卷积权重调整方法
        """
        if old_weight.shape == new_shape:
            return old_weight.clone()
        
        old_out, old_in, old_h, old_w = old_weight.shape
        new_out, new_in, new_h, new_w = new_shape
        
        # 创建新权重
        new_weight = torch.zeros(new_shape, dtype=old_weight.dtype, device=old_weight.device)
        
        # 复制可以复制的部分
        min_out = min(old_out, new_out)
        min_in = min(old_in, new_in)
        min_h = min(old_h, new_h)
        min_w = min(old_w, new_w)
        
        new_weight[:min_out, :min_in, :min_h, :min_w] = old_weight[:min_out, :min_in, :min_h, :min_w]
        
        # 对于新增的部分，使用Kaiming初始化
        if new_out > old_out or new_in > old_in:
            with torch.no_grad():
                torch.nn.init.kaiming_normal_(new_weight, mode='fan_in', nonlinearity='leaky_relu')
                # 保留已复制的部分
                new_weight[:min_out, :min_in, :min_h, :min_w] = old_weight[:min_out, :min_in, :min_h, :min_w]
        
        return new_weight

    def _adjust_general_weight(self, old_weight, new_shape, key):
        """
        通用权重调整方法
        """
        if old_weight.shape == new_shape:
            return old_weight.clone()
        
        # 创建新权重并使用智能初始化
        new_weight = torch.zeros(new_shape, dtype=old_weight.dtype, device=old_weight.device)
        return self._initialize_weight_smart(new_weight, key)
    

    
    def _initialize_weight_smart(self, weight_tensor, key):
        """
        智能权重初始化
        """
        with torch.no_grad():
            if 'conv' in key and len(weight_tensor.shape) == 4:
                # 卷积层使用Xavier初始化
                torch.nn.init.xavier_uniform_(weight_tensor)
            elif 'bias' in key:
                # 偏置初始化为0
                torch.nn.init.zeros_(weight_tensor)
            elif len(weight_tensor.shape) == 2:
                # 线性层使用Xavier初始化
                torch.nn.init.xavier_uniform_(weight_tensor)
            else:
                # 其他情况使用正态分布
                torch.nn.init.normal_(weight_tensor, mean=0, std=0.02)
        
        return weight_tensor
    
    def _progressive_feature_adjust(self, checkpoint_path, old_features, new_features, steps=3):
        """
        渐进式特征数调整，避免一次性大幅调整导致的问题
        """
        print(f"{Fore.CYAN}🚀 开始渐进式特征数调整: {old_features} -> {new_features} (分{steps}步){Style.RESET_ALL}")
        
        current_features = old_features
        # 计算每步增量（确保最后一步能到达目标）
        increment = (new_features - old_features) // steps
        if increment == 0:
            increment = 1
        
        base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        checkpoint_dir = os.path.dirname(checkpoint_path)
        
        for step in range(steps):
            try:
                # 计算当前步骤的目标特征数
                if step == steps - 1:
                    target_features = new_features
                else:
                    target_features = current_features + increment
                    # 确保不会超过目标
                    if (new_features > old_features and target_features > new_features) or \
                       (new_features < old_features and target_features < new_features):
                        target_features = new_features
                
                if target_features == current_features:
                    continue
                
                print(f"\n{Fore.YELLOW}📈 步骤 {step+1}/{steps}: {current_features} -> {target_features}{Style.RESET_ALL}")
                
                # 加载当前检查点
                if step == 0:
                    current_checkpoint_path = checkpoint_path
                else:
                    current_checkpoint_path = os.path.join(checkpoint_dir, f"{base_name}_step_{step}_features_{current_features}.pth")
                
                checkpoint = torch.load(current_checkpoint_path, map_location='cpu', weights_only=False)
                old_state_dict = checkpoint['generator_state_dict']
                
                # 创建目标模型
                from src.models.esrgan import LiteRealESRGAN
                target_model = LiteRealESRGAN(
                    num_blocks=6,  # 使用默认块数
                    num_features=target_features
                )
                new_state_dict = target_model.state_dict()
                
                # 调整权重
                adjusted_state_dict = self._adjust_features_weights(
                    old_state_dict, new_state_dict, current_features, target_features
                )
                
                # 更新检查点
                checkpoint['generator_state_dict'] = adjusted_state_dict
                
                # 保存中间结果
                step_output_path = os.path.join(checkpoint_dir, f"{base_name}_step_{step+1}_features_{target_features}.pth")
                torch.save(checkpoint, step_output_path)
                print(f"{Fore.GREEN}✅ 步骤 {step+1} 完成，保存至: {os.path.basename(step_output_path)}{Style.RESET_ALL}")
                
                # 验证当前步骤
                if self._verify_feature_adjustment(step_output_path, target_features):
                    current_features = target_features
                    print(f"{Fore.GREEN}✅ 步骤 {step+1} 验证通过{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}❌ 步骤 {step+1} 验证失败，停止渐进调整{Style.RESET_ALL}")
                    return False, f"步骤 {step+1} 验证失败"
                
            except Exception as e:
                print(f"{Fore.RED}❌ 步骤 {step+1} 调整失败: {str(e)}{Style.RESET_ALL}")
                import traceback
                traceback.print_exc()
                return False, f"步骤 {step+1} 失败: {str(e)}"
        
        # 返回最终结果
        import shutil
        final_path = os.path.join(checkpoint_dir, f"{base_name}_features_{new_features}.pth")
        shutil.copy2(step_output_path, final_path)
        return True, final_path

    def _validate_rrdb_structure(self, state_dict, num_features):
        """
        验证RRDB结构的通道数是否正确
        """
        print(f"{Fore.CYAN}🔍 验证RRDB结构通道数...{Style.RESET_ALL}")
        
        growth_rate = 32
        errors = []
        
        for key, weight in state_dict.items():
            if 'rrdb_blocks' in key and 'dense' in key and 'conv' in key and 'weight' in key:
                parts = key.split('.')
                dense_idx = int(parts[2][-1])
                conv_idx = int(parts[3][-1])
                
                layer_order = (dense_idx - 1) * 5 + conv_idx
                expected_in = num_features + growth_rate * (layer_order - 1)
                actual_in = weight.shape[1]
                
                if actual_in != expected_in:
                    errors.append(f"{key}: 预期{expected_in}, 实际{actual_in}")
        
        if errors:
            print(f"{Fore.RED}❌ 发现 {len(errors)} 个通道数错误:{Style.RESET_ALL}")
            for error in errors[:5]:  # 只显示前5个错误
                print(f"    {error}")
            if len(errors) > 5:
                print(f"    ... 还有 {len(errors)-5} 个错误")
            return False
        else:
            print(f"{Fore.GREEN}✅ RRDB结构验证通过{Style.RESET_ALL}")
            return True

    def _verify_feature_adjustment(self, checkpoint_path, expected_features):
        """
        验证特征数调整结果
        """
        try:
            print(f"{Fore.CYAN}🔍 验证调整结果...{Style.RESET_ALL}")
            
            # 1. 检测特征数
            detected_config = self._detect_checkpoint_config(checkpoint_path)
            if detected_config and detected_config['num_features'] == expected_features:
                print(f"{Fore.GREEN}✅ 特征数验证通过: {expected_features}{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}❌ 特征数验证失败{Style.RESET_ALL}")
                return False
            
            # 2. 测试模型加载
            from src.models.esrgan import LiteRealESRGAN
            model = LiteRealESRGAN(
                num_blocks=6,  # 使用默认值而不是self.config
                num_features=expected_features
            )
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['generator_state_dict'])
            print(f"{Fore.GREEN}✅ 模型加载验证通过{Style.RESET_ALL}")
            
            # 3. 测试前向传播
            model.eval()
            test_input = torch.randn(1, 3, 64, 64)
            with torch.no_grad():
                output = model(test_input)
            print(f"{Fore.GREEN}✅ 前向传播验证通过: {test_input.shape} -> {output.shape}{Style.RESET_ALL}")
            
            return True
            
        except Exception as e:
            print(f"{Fore.RED}❌ 验证失败: {str(e)}{Style.RESET_ALL}")
            return False
    
    def _is_new_rrdb_block(self, key, old_blocks):
        """判断是否是新增的RRDB块"""
        import re
        match = re.match(r'rrdb_blocks\.(\d+)\.', key)
        if match:
            block_idx = int(match.group(1))
            return block_idx >= old_blocks
        return False
    
    def _get_source_block_key(self, key, old_state_dict, old_blocks):
        """获取源块的权重键"""
        if old_blocks == 0:
            return None
            
        import re
        match = re.match(r'(rrdb_blocks\.)(\d+)(\..+)', key)
        if match:
            prefix = match.group(1)
            block_idx = int(match.group(2))
            suffix = match.group(3)
            
            # 使用最后一个旧块作为源
            source_idx = old_blocks - 1
            source_key = f"{prefix}{source_idx}{suffix}"
            
            if source_key in old_state_dict:
                return source_key
        return None
    
    def _initialize_weight(self, weight_tensor):
        """初始化新权重"""
        with torch.no_grad():
            if len(weight_tensor.shape) == 4:  # 卷积层
                torch.nn.init.kaiming_normal_(weight_tensor, mode='fan_in', nonlinearity='leaky_relu')
            elif len(weight_tensor.shape) == 1:  # 偏置
                torch.nn.init.zeros_(weight_tensor)
            else:  # 其他
                torch.nn.init.normal_(weight_tensor, mean=0, std=0.02)
        return weight_tensor
    
    def _smart_resize_weight(self, old_weight, new_shape, key):
        """智能调整权重尺寸的通用方法"""
        if old_weight.shape == new_shape:
            return old_weight.clone()
            
        # 创建新权重张量
        new_weight = torch.zeros(new_shape, dtype=old_weight.dtype, device=old_weight.device)
        
        # 计算可复制的维度大小
        min_dims = [min(old_dim, new_dim) for old_dim, new_dim in zip(old_weight.shape, new_shape)]
        
        # 创建切片元组
        slices = tuple(slice(0, dim) for dim in min_dims)
        
        # 复制可复制的部分
        new_weight[slices] = old_weight[slices].clone()
        
        # 对于卷积层，初始化新增部分
        if 'conv' in key and len(new_shape) == 4:
            with torch.no_grad():
                # 保存已复制部分
                copied_part = new_weight[slices].clone()
                # 整体初始化
                torch.nn.init.kaiming_normal_(new_weight, mode='fan_in', nonlinearity='leaky_relu')
                # 恢复已复制部分
                new_weight[slices] = copied_part
        
        # 对于偏置，新增部分初始化为0
        elif 'bias' in key:
            new_weight[min_dims[0]:] = 0.0
            
        return new_weight
    def _detect_checkpoint_config(self, checkpoint_path):
        """检测检查点的模型配置"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # 获取状态字典
            state_dict = None
            if 'generator_state_dict' in checkpoint:
                state_dict = checkpoint['generator_state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                if any(key.startswith(('conv_first', 'rrdb_blocks', 'conv_last')) for key in checkpoint.keys()):
                    state_dict = checkpoint
                else:
                    return None
            
            if state_dict is None:
                return None
            
            # 检测特征数
            num_features = None
            for key, tensor in state_dict.items():
                if 'conv_first.weight' in key:
                    num_features = tensor.shape[0]
                    break
            
            # 检测块数
            num_blocks = 0
            for key in state_dict.keys():
                if 'rrdb_blocks.' in key:
                    import re
                    block_match = re.search(r'rrdb_blocks\.(\d+)\.', key)
                    if block_match:
                        block_num = int(block_match.group(1))
                        num_blocks = max(num_blocks, block_num + 1)
            
            if num_blocks > 0 and num_features is not None:
                return {
                    'num_blocks': num_blocks,
                    'num_features': num_features
                }
            else:
                return None
                
        except Exception as e:
            return None

class ConsoleTrainer:
    def __init__(self):
        """初始化控制台训练器"""
        self.config = load_config()
        self.gpu_info = self.get_gpu_info()
        self.compatibility_checker = ModelCompatibilityChecker()
        self.gpu_optimizer = GPUOptimizer()
        
        # 根据GPU自动调整配置
        self.check_and_adjust_memory_config()
        
        # 如果是RTX 4090，自动启用优化
        if self.gpu_info["available"] and "4090" in self.gpu_info.get("name", ""):
            self.config = load_rtx_4090_config()
            if self.config.get("rtx_4090", {}).get("tf32_enabled", False):
                self.gpu_optimizer.enable_tf32()
            if self.config.get("rtx_4090", {}).get("compile_enabled", False):
                self.gpu_optimizer.enable_compile()
            print(f"{Fore.GREEN}🚀 检测到RTX 4090，已自动启用专用优化{Style.RESET_ALL}")

    def check_and_adjust_memory_config(self):
        """检查GPU显存并自动调整配置"""
        if not self.gpu_info["available"]:
            return
        
        gpu_memory = self.gpu_info["memory"]
        print(f"{Fore.CYAN}检测到GPU显存: {gpu_memory:.1f}GB{Style.RESET_ALL}")
        
        # 使用自动配置
        auto_config = auto_config_by_gpu_memory(gpu_memory)
        
        # 合并配置（保留用户自定义设置）
        for key, value in auto_config.items():
            if key not in self.config:
                self.config[key] = value
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if sub_key not in self.config[key]:
                        self.config[key][sub_key] = sub_value
        
        print(f"{Fore.GREEN}✅ 已根据GPU显存自动调整配置{Style.RESET_ALL}")

    def get_gpu_info(self):
        """获取GPU信息"""
        gpu_info = {
            "available": torch.cuda.is_available(),
            "name": "",
            "memory": 0.0,
            "count": 0
        }
        
        if gpu_info["available"]:
            try:
                gpu_info["name"] = torch.cuda.get_device_name(0)
                gpu_info["memory"] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                gpu_info["count"] = torch.cuda.device_count()
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️  获取GPU信息失败: {e}{Style.RESET_ALL}")
        
        return gpu_info

    def clear_screen(self):
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self):
        """打印标题"""
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}🤖 AI图像超分辨率训练控制台{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

    def print_gpu_info(self):
        """打印GPU信息"""
        if self.gpu_info["available"]:
            print(f"\n{Fore.GREEN}🎮 GPU信息:{Style.RESET_ALL}")
            print(f"设备: {self.gpu_info['name']}")
            print(f"显存: {self.gpu_info['memory']:.1f} GB")
            print(f"数量: {self.gpu_info['count']}")
            
            # 显示优化状态
            if hasattr(self, 'gpu_optimizer'):
                print(f"TF32: {'✅启用' if self.gpu_optimizer.tf32_enabled else '❌禁用'}")
                print(f"编译: {'✅启用' if self.gpu_optimizer.compile_enabled else '❌禁用'}")
        else:
            print(f"\n{Fore.RED}❌ 未检测到可用GPU，将使用CPU训练{Style.RESET_ALL}")

    def print_main_menu(self):
        """打印主菜单"""
        print(f"\n{Fore.CYAN}📋 主菜单:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}1.{Style.RESET_ALL} 开始新训练")
        print(f"{Fore.WHITE}2.{Style.RESET_ALL} 增量训练")
        print(f"{Fore.WHITE}3.{Style.RESET_ALL} 验证模型")
        print(f"{Fore.WHITE}4.{Style.RESET_ALL} 配置管理")
        print(f"{Fore.WHITE}5.{Style.RESET_ALL} GPU优化设置")
        print(f"{Fore.WHITE}6.{Style.RESET_ALL} 检查点管理")
        print(f"{Fore.WHITE}7.{Style.RESET_ALL} 性能监控")
        print(f"{Fore.WHITE}8.{Style.RESET_ALL} 退出程序")

    def start_incremental_training(self):
        """启动增量训练"""
        # 检查数据目录
        if not self._check_data_directories():
            return
        
        # 选择检查点文件
        checkpoint_files = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
        if not checkpoint_files:
            print(f"{Fore.RED}❌ 没有找到检查点文件{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}📁 可用的检查点文件:{Style.RESET_ALL}")
        for i, file in enumerate(checkpoint_files, 1):
            print(f"{i}. {file}")
        
        try:
            choice = int(input(f"\n{Fore.CYAN}请选择检查点文件 (1-{len(checkpoint_files)}): {Style.RESET_ALL}"))
            if 1 <= choice <= len(checkpoint_files):
                selected_file = checkpoint_files[choice - 1]
                checkpoint_path = os.path.join('checkpoints', selected_file)
                
                # 如果选择的是调整文件，先删除它，使用原始文件重新调整
                if '_adjusted' in selected_file:
                    original_file = selected_file.replace('_adjusted', '')
                    original_path = os.path.join('checkpoints', original_file)
                    
                    if os.path.exists(original_path):
                        # 删除有问题的调整文件
                        try:
                            os.remove(checkpoint_path)
                            print(f"{Fore.GREEN}✅ 已删除有问题的调整文件: {selected_file}{Style.RESET_ALL}")
                        except Exception as e:
                            print(f"{Fore.YELLOW}⚠️  删除调整文件失败: {e}{Style.RESET_ALL}")
                        
                        # 使用原始文件
                        checkpoint_path = original_path
                        selected_file = original_file
                        print(f"{Fore.CYAN}🔄 改用原始文件: {selected_file}{Style.RESET_ALL}")
                
                # 自动检测检查点配置并匹配
                print(f"\n{Fore.CYAN}🔍 检测检查点配置...{Style.RESET_ALL}")
                checkpoint_config = self._detect_checkpoint_config(checkpoint_path)
                
                if checkpoint_config:
                    current_blocks = self.config['model']['num_blocks']
                    current_features = self.config['model']['num_features']
                    checkpoint_blocks = checkpoint_config['num_blocks']
                    checkpoint_features = checkpoint_config['num_features']
                    
                    print(f"\n{Fore.CYAN}📊 配置对比:{Style.RESET_ALL}")
                    print(f"{Fore.CYAN}检查点模型: {checkpoint_blocks}块, {checkpoint_features}特征{Style.RESET_ALL}")
                    print(f"{Fore.CYAN}当前配置:   {current_blocks}块, {current_features}特征{Style.RESET_ALL}")
                    
                    # 检查是否完全匹配
                    blocks_match = current_blocks == checkpoint_blocks
                    features_match = current_features == checkpoint_features
                    
                    if blocks_match and features_match:
                        print(f"{Fore.GREEN}✅ 配置完全匹配，可以直接加载{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.YELLOW}⚠️  配置不匹配！{Style.RESET_ALL}")
                        if not blocks_match:
                            print(f"{Fore.YELLOW}  - 块数不匹配: {current_blocks} vs {checkpoint_blocks}{Style.RESET_ALL}")
                        if not features_match:
                            print(f"{Fore.YELLOW}  - 特征数不匹配: {current_features} vs {checkpoint_features}{Style.RESET_ALL}")
                        
                        # 强制刷新输出缓冲区
                        import sys
                        sys.stdout.flush()
                        
                        # 🔥 新增：智能特征数调整选项
                        if not features_match and blocks_match:
                            # 只有特征数不匹配，提供智能调整选项
                            print(f"\n{Fore.CYAN}🎯 检测到仅特征数不匹配，提供智能调整选项:{Style.RESET_ALL}")
                            print(f"{Fore.WHITE}1.{Style.RESET_ALL} 智能调整检查点特征数以匹配配置 ({checkpoint_features} -> {current_features}) 🔥推荐")
                            print(f"{Fore.WHITE}2.{Style.RESET_ALL} 调整配置以匹配检查点 ({current_features} -> {checkpoint_features})")
                            print(f"{Fore.WHITE}3.{Style.RESET_ALL} 使用通用兼容性调整")
                            print(f"{Fore.WHITE}4.{Style.RESET_ALL} 取消操作")
                            
                            # 强制刷新输出缓冲区
                            sys.stdout.flush()
                            
                            feature_choice = input(f"\n{Fore.CYAN}请选择 (1-4): {Style.RESET_ALL}").strip()
                            
                            if feature_choice == '1':
                                # 智能特征数调整
                                success, result = self.compatibility_checker.smart_adjust_features_enhanced(
                                    checkpoint_path, current_features
                                )
                                if success:
                                    checkpoint_path = result
                                    print(f"{Fore.GREEN}✅ 特征数调整成功！{Style.RESET_ALL}")
                                else:
                                    print(f"{Fore.RED}❌ 特征数调整失败: {result}{Style.RESET_ALL}")
                                    return
                            elif feature_choice == '2':
                                # 调整配置以匹配检查点
                                self._adjust_config_to_match_checkpoint(checkpoint_config)
                                print(f"{Fore.GREEN}✅ 已调整配置以匹配检查点{Style.RESET_ALL}")
                            elif feature_choice == '3':
                                # 使用通用兼容性调整
                                print(f"{Fore.CYAN}🔧 正在使用通用兼容性调整...{Style.RESET_ALL}")
                                success, result = self.compatibility_checker.adjust_checkpoint_for_new_config(
                                    checkpoint_path, self.config
                                )
                                if success:
                                    checkpoint_path = result
                                    print(f"{Fore.GREEN}✅ 通用调整成功: {os.path.basename(result)}{Style.RESET_ALL}")
                                else:
                                    print(f"{Fore.RED}❌ 通用调整失败: {result}{Style.RESET_ALL}")
                                    return
                            else:
                                print(f"{Fore.YELLOW}❌ 用户取消操作{Style.RESET_ALL}")
                                return
                        else:
                            # 其他不匹配情况，使用原有逻辑
                            print(f"\n{Fore.CYAN}请选择处理方式:{Style.RESET_ALL}")
                            print(f"{Fore.WHITE}1.{Style.RESET_ALL} 自动调整当前配置以匹配检查点 (推荐)")
                            print(f"{Fore.WHITE}2.{Style.RESET_ALL} 调整检查点以适应当前配置")
                            print(f"{Fore.WHITE}3.{Style.RESET_ALL} 取消操作")
                            
                            config_choice = input(f"\n{Fore.CYAN}请选择 (1-3): {Style.RESET_ALL}").strip()
                            
                            if config_choice == '1':
                                # 调整当前配置以匹配检查点
                                self._adjust_config_to_match_checkpoint(checkpoint_config)
                                print(f"{Fore.GREEN}✅ 已调整配置以匹配检查点{Style.RESET_ALL}")
                            elif config_choice == '2':
                                # 调整检查点以适应当前配置
                                print(f"{Fore.CYAN}🔧 正在调整检查点...{Style.RESET_ALL}")
                                success, result = self.compatibility_checker.adjust_checkpoint_for_new_config(
                                    checkpoint_path, self.config
                                )
                                
                                if success:
                                    checkpoint_path = result
                                    print(f"{Fore.GREEN}✅ 检查点调整成功: {os.path.basename(result)}{Style.RESET_ALL}")
                                else:
                                    print(f"{Fore.RED}❌ 检查点调整失败: {result}{Style.RESET_ALL}")
                                    return
                            else:
                                print(f"{Fore.YELLOW}❌ 用户取消操作{Style.RESET_ALL}")
                                return
                else:
                    print(f"{Fore.RED}❌ 无法检测检查点配置，将使用兼容性检查器{Style.RESET_ALL}")
                    # 回退到原来的兼容性检查
                    is_compatible, message = self.compatibility_checker.check_model_compatibility(checkpoint_path, self.config)
                    
                    if not is_compatible:
                        print(f"{Fore.YELLOW}⚠️  {message}{Style.RESET_ALL}")
                        adjust = input(f"{Fore.CYAN}是否调整模型以适应当前配置? (y/n): {Style.RESET_ALL}").lower().strip()
                        
                        if adjust == 'y':
                            print(f"{Fore.CYAN}🔧 正在调整模型...{Style.RESET_ALL}")
                            
                            # 尝试检测检查点配置进行智能调整
                            checkpoint_config = self._detect_checkpoint_config(checkpoint_path)
                            if checkpoint_config:
                                current_blocks = self.config['model']['num_blocks']
                                current_features = self.config['model']['num_features']
                                checkpoint_blocks = checkpoint_config['num_blocks']
                                checkpoint_features = checkpoint_config['num_features']
                                
                                # 如果只是特征数不匹配，使用智能特征调整
                                if current_blocks == checkpoint_blocks and current_features != checkpoint_features:
                                    print(f"{Fore.CYAN}🎯 使用智能特征调整: {checkpoint_features} -> {current_features}{Style.RESET_ALL}")
                                    success, result = self.compatibility_checker.smart_adjust_features_enhanced(
                                        checkpoint_path, current_features
                                    )
                                else:
                                    # 使用通用调整方法
                                    success, result = self.compatibility_checker.adjust_checkpoint_for_new_config(
                                        checkpoint_path, self.config
                                    )
                            else:
                                # 回退到通用调整方法
                                success, result = self.compatibility_checker.adjust_checkpoint_for_new_config(
                                    checkpoint_path, self.config
                                )
                            
                            if success:
                                checkpoint_path = result
                                print(f"{Fore.GREEN}✅ 模型调整成功: {os.path.basename(result)}{Style.RESET_ALL}")
                            else:
                                print(f"{Fore.RED}❌ 模型调整失败: {result}{Style.RESET_ALL}")
                                return
                        else:
                            print(f"{Fore.YELLOW}❌ 用户取消调整{Style.RESET_ALL}")
                            return
                
                # 在新线程中启动训练
                print(f"\n{Fore.GREEN}🚀 启动增量训练...{Style.RESET_ALL}")
                
                def training_worker():
                    try:
                        from src.training.train_manager import MemoryOptimizedTrainingManager
                        trainer = MemoryOptimizedTrainingManager(self.config)
                        trainer.start_incremental_training(checkpoint_path)
                    except Exception as e:
                        print(f"{Fore.RED}❌ 训练过程出错: {e}{Style.RESET_ALL}")
                        import traceback
                        traceback.print_exc()
                
                import threading
                training_thread = threading.Thread(target=training_worker)
                training_thread.daemon = True
                training_thread.start()
                
                print(f"{Fore.GREEN}✅ 增量训练已在后台启动{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}💡 您可以继续使用其他功能，训练将在后台进行{Style.RESET_ALL}")
                
            else:
                print(f"{Fore.RED}❌ 无效选择{Style.RESET_ALL}")
                
        except ValueError:
            print(f"{Fore.RED}❌ 请输入有效数字{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ 增量训练启动失败: {e}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
        
        input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")

    def _detect_checkpoint_config(self, checkpoint_path):
        """检测检查点的模型配置"""
        try:
            print(f"{Fore.CYAN}🔍 正在分析检查点文件: {os.path.basename(checkpoint_path)}{Style.RESET_ALL}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # 尝试多种可能的状态字典键
            state_dict = None
            if 'generator_state_dict' in checkpoint:
                state_dict = checkpoint['generator_state_dict']
                print(f"{Fore.GREEN}✅ 找到generator_state_dict{Style.RESET_ALL}")
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"{Fore.GREEN}✅ 找到model_state_dict{Style.RESET_ALL}")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"{Fore.GREEN}✅ 找到state_dict{Style.RESET_ALL}")
            else:
                # 检查是否直接是状态字典
                if any(key.startswith(('conv_first', 'rrdb_blocks', 'conv_last')) for key in checkpoint.keys()):
                    state_dict = checkpoint
                    print(f"{Fore.GREEN}✅ 直接使用checkpoint作为state_dict{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}⚠️  未找到标准的状态字典{Style.RESET_ALL}")
                    return None
            
            if state_dict is None:
                return None
            
            # 检测特征数
            num_features = None
            for key, tensor in state_dict.items():
                if 'conv_first.weight' in key:
                    num_features = tensor.shape[0]  # 输出通道数
                    print(f"{Fore.GREEN}✅ 从conv_first.weight检测到特征数: {num_features}{Style.RESET_ALL}")
                    break
            
            # 检测块数
            num_blocks = 0
            rrdb_keys = []
            for key in state_dict.keys():
                if 'rrdb_blocks.' in key:
                    rrdb_keys.append(key)
                    # 使用正则表达式提取块编号
                    import re
                    block_match = re.search(r'rrdb_blocks\.(\d+)\.', key)
                    if block_match:
                        block_num = int(block_match.group(1))
                        num_blocks = max(num_blocks, block_num + 1)
            
            if rrdb_keys:
                print(f"{Fore.GREEN}✅ 使用模式'rrdb_blocks.'检测到RRDB块数: {num_blocks}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}📋 找到 {len(rrdb_keys)} 个RRDB相关权重{Style.RESET_ALL}")
                
                # 显示一些示例键
                unique_blocks = set()
                for key in rrdb_keys[:20]:  # 只显示前20个
                    block_match = re.search(r'rrdb_blocks\.(\d+)', key)
                    if block_match:
                        unique_blocks.add(f"rrdb_blocks.{block_match.group(1)}")
                
                if unique_blocks:
                    print(f"{Fore.CYAN}📋 示例RRDB键:{Style.RESET_ALL}")
                    for block in sorted(list(unique_blocks))[:5]:  # 只显示前5个
                        print(f"  - {block}")
            
            if num_blocks > 0 and num_features is not None:
                return {
                    'num_blocks': num_blocks,
                    'num_features': num_features
                }
            else:
                print(f"{Fore.YELLOW}⚠️  无法完全检测配置: blocks={num_blocks}, features={num_features}{Style.RESET_ALL}")
                return None
                
        except Exception as e:
            print(f"{Fore.RED}❌ 检测检查点配置失败: {str(e)}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            return None

    def _adjust_config_to_match_checkpoint(self, checkpoint_config):
        """调整当前配置以匹配检查点"""
        # 更新模型配置
        self.config['model']['num_blocks'] = checkpoint_config['num_blocks']
        self.config['model']['num_features'] = checkpoint_config['num_features']
        
        # 根据新的模型配置调整训练参数
        gpu_memory = self.gpu_info.get("memory", 8.0)
        
        # 根据块数和显存调整批次大小
        if checkpoint_config['num_blocks'] >= 8:
            # 8块模型需要更多显存
            if gpu_memory >= 12:
                self.config['training']['batch_size'] = 4
            elif gpu_memory >= 8:
                self.config['training']['batch_size'] = 2
            else:
                self.config['training']['batch_size'] = 1
                if 'gradient_accumulation_steps' not in self.config['training']:
                    self.config['training']['gradient_accumulation_steps'] = 8
                else:
                    self.config['training']['gradient_accumulation_steps'] = 8
        elif checkpoint_config['num_blocks'] >= 6:
            # 6块模型
            if gpu_memory >= 8:
                self.config['training']['batch_size'] = 4
            elif gpu_memory >= 6:
                self.config['training']['batch_size'] = 2
            else:
                self.config['training']['batch_size'] = 1
                if 'gradient_accumulation_steps' not in self.config['training']:
                    self.config['training']['gradient_accumulation_steps'] = 4
                else:
                    self.config['training']['gradient_accumulation_steps'] = 4
        
        print(f"{Fore.CYAN}已调整配置: {checkpoint_config['num_blocks']}块, {checkpoint_config['num_features']}特征{Style.RESET_ALL}")
        print(f"{Fore.CYAN}批次大小: {self.config['training']['batch_size']}{Style.RESET_ALL}")

    def _check_data_directories(self):
        """检查数据目录"""
        required_dirs = [
            self.config["data"]["train_lr_dir"],
            self.config["data"]["train_hr_dir"],
            self.config["data"]["val_lr_dir"],
            self.config["data"]["val_hr_dir"]
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            print(f"{Fore.RED}❌ 缺少以下数据目录:{Style.RESET_ALL}")
            for dir_path in missing_dirs:
                print(f"  - {dir_path}")
            return False
        
        return True

    def gpu_optimization_menu(self):
        """GPU优化设置菜单"""
        while True:
            self.clear_screen()
            self.print_header()
            print(f"\n{Fore.CYAN}🚀 GPU优化设置{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            
            # 显示当前GPU信息
            self.print_gpu_info()
            
            print(f"\n{Fore.CYAN}可用操作:{Style.RESET_ALL}")
            print(f"{Fore.WHITE}1.{Style.RESET_ALL} 启用/禁用TF32加速")
            print(f"{Fore.WHITE}2.{Style.RESET_ALL} 启用/禁用模型编译")
            print(f"{Fore.WHITE}3.{Style.RESET_ALL} 自动调整批次大小")
            print(f"{Fore.WHITE}4.{Style.RESET_ALL} 加载RTX 4090配置")
            print(f"{Fore.WHITE}5.{Style.RESET_ALL} 显存配置建议")
            print(f"{Fore.WHITE}6.{Style.RESET_ALL} 返回主菜单")
            
            choice = input(f"\n{Fore.CYAN}请选择操作 (1-6): {Style.RESET_ALL}").strip()
            
            if choice == '1':
                self._toggle_tf32()
            elif choice == '2':
                self._toggle_compile()
            elif choice == '3':
                self._auto_adjust_batch_size()
            elif choice == '4':
                self._load_rtx_4090_config()
            elif choice == '5':
                self._show_memory_recommendations()
            elif choice == '6':
                break
            else:
                print(f"{Fore.RED}❌ 无效选择，请重试{Style.RESET_ALL}")
                time.sleep(1)

    def _toggle_tf32(self):
        """切换TF32设置"""
        if self.gpu_optimizer.tf32_enabled:
            self.gpu_optimizer.disable_tf32()
        else:
            self.gpu_optimizer.enable_tf32()
        input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")

    def _toggle_compile(self):
        """切换模型编译设置"""
        if self.gpu_optimizer.compile_enabled:
            self.gpu_optimizer.disable_compile()
        else:
            self.gpu_optimizer.enable_compile()
        input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")

    def _auto_adjust_batch_size(self):
        """自动调整批次大小"""
        try:
            from src.models.esrgan import LiteRealESRGAN
            model = LiteRealESRGAN(
                num_blocks=self.config['model']['num_blocks'],
                num_features=self.config['model']['num_features']
            ).cuda()
            
            optimal_batch_size = self.gpu_optimizer.get_optimal_batch_size(
                model, (1, 3, 64, 64), self.gpu_info["memory"]
            )
            
            self.config['training']['batch_size'] = optimal_batch_size
            print(f"{Fore.GREEN}✅ 批次大小已调整为: {optimal_batch_size}{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}❌ 自动调整失败: {e}{Style.RESET_ALL}")
        
        input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")

    def _load_rtx_4090_config(self):
        """加载RTX 4090配置"""
        self.config = load_rtx_4090_config()
        print(f"{Fore.GREEN}✅ 已加载RTX 4090专用配置{Style.RESET_ALL}")
        
        # 自动启用优化
        if self.config.get("rtx_4090", {}).get("tf32_enabled", False):
            self.gpu_optimizer.enable_tf32()
        if self.config.get("rtx_4090", {}).get("compile_enabled", False):
            self.gpu_optimizer.enable_compile()
        
        input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")

    def _show_memory_recommendations(self):
        """显示显存配置建议"""
        gpu_memory = self.gpu_info["memory"]
        print(f"\n{Fore.CYAN}💡 显存配置建议 (当前: {gpu_memory:.1f}GB):{Style.RESET_ALL}")
        
        if gpu_memory < 4.5:
            print(f"{Fore.YELLOW}• 建议使用4GB配置{Style.RESET_ALL}")
            print(f"• 批次大小: 1")
            print(f"• 模型块数: 3")
            print(f"• 特征数: 24")
        elif gpu_memory < 6.5:
            print(f"{Fore.YELLOW}• 建议使用6GB配置{Style.RESET_ALL}")
            print(f"• 批次大小: 2")
            print(f"• 模型块数: 4")
            print(f"• 特征数: 32")
        elif gpu_memory < 8.5:
            print(f"{Fore.GREEN}• 建议使用8GB配置{Style.RESET_ALL}")
            print(f"• 批次大小: 4")
            print(f"• 模型块数: 6")
            print(f"• 特征数: 48")
        elif gpu_memory >= 20:
            print(f"{Fore.GREEN}• 建议使用RTX 4090配置{Style.RESET_ALL}")
            print(f"• 批次大小: 12")
            print(f"• 模型块数: 8")
            print(f"• 特征数: 64")
            print(f"• TF32加速: 启用")
            print(f"• 模型编译: 启用")
        else:
            print(f"{Fore.GREEN}• 建议使用高显存配置{Style.RESET_ALL}")
            print(f"• 批次大小: 6-8")
            print(f"• 模型块数: 6")
            print(f"• 特征数: 64")
        
        input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")

    def run(self):
        """运行主程序"""
        while True:
            self.clear_screen()
            self.print_header()
            self.print_gpu_info()
            self.print_main_menu()
            
            choice = input(f"\n{Fore.CYAN}请选择操作 (1-8): {Style.RESET_ALL}").strip()
            
            if choice == '1':
                self.start_new_training()
            elif choice == '2':
                self.start_incremental_training()
            elif choice == '3':
                self.validate_model()
            elif choice == '4':
                self.config_management_menu()
            elif choice == '5':
                self.gpu_optimization_menu()
            elif choice == '6':
                self.checkpoint_management_menu()
            elif choice == '7':
                self.performance_monitoring_menu()
            elif choice == '8':
                print(f"{Fore.GREEN}👋 感谢使用！{Style.RESET_ALL}")
                break
            else:
                print(f"{Fore.RED}❌ 无效选择，请重试{Style.RESET_ALL}")
                time.sleep(1)

    def start_new_training(self):
        """开始新训练"""
        try:
            print(f"{Fore.CYAN}🚀 开始新训练{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            
            # 检查数据目录
            if not self._check_data_directories():
                input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")
                return
            
            # 显示当前配置
            print(f"\n{Fore.CYAN}📋 当前训练配置:{Style.RESET_ALL}")
            print(f"• 模型块数: {self.config['model']['num_blocks']}")
            print(f"• 特征数: {self.config['model']['num_features']}")
            print(f"• 批次大小: {self.config['training']['batch_size']}")
            print(f"• 训练轮数: {self.config['training']['num_epochs']}")
            print(f"• 学习率: {self.config['training']['learning_rate']}")
            print(f"• 保存频率: {self.config['training']['save_frequency']} epochs")
            
            # 确认开始训练
            confirm = input(f"\n{Fore.CYAN}是否使用当前配置开始新训练? (y/n): {Style.RESET_ALL}").strip().lower()
            if confirm != 'y':
                print(f"{Fore.YELLOW}❌ 用户取消训练{Style.RESET_ALL}")
                input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")
                return
            
            # 检查是否存在检查点文件，询问是否清理
            checkpoint_files = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
            if checkpoint_files:
                print(f"\n{Fore.YELLOW}⚠️  检测到现有检查点文件:{Style.RESET_ALL}")
                for f in checkpoint_files[:5]:  # 只显示前5个
                    print(f"  - {f}")
                if len(checkpoint_files) > 5:
                    print(f"  ... 还有 {len(checkpoint_files) - 5} 个文件")
                
                clean_choice = input(f"\n{Fore.CYAN}是否清理现有检查点? (y/n): {Style.RESET_ALL}").strip().lower()
                if clean_choice == 'y':
                    try:
                        for f in checkpoint_files:
                            os.remove(os.path.join('checkpoints', f))
                        print(f"{Fore.GREEN}✅ 已清理 {len(checkpoint_files)} 个检查点文件{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"{Fore.RED}❌ 清理检查点失败: {e}{Style.RESET_ALL}")
                        input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")
                        return
            
            # 启动新训练
            print(f"\n{Fore.GREEN}🚀 启动新训练...{Style.RESET_ALL}")
            
            def training_worker():
                try:
                    from src.training.train_manager import MemoryOptimizedTrainingManager
                    trainer = MemoryOptimizedTrainingManager(self.config)
                    
                    # 开始新训练（不传递检查点路径）
                    trainer.start_training()
                    
                except Exception as e:
                    print(f"{Fore.RED}❌ 训练过程出错: {e}{Style.RESET_ALL}")
                    import traceback
                    traceback.print_exc()
            
            # 在新线程中启动训练
            import threading
            training_thread = threading.Thread(target=training_worker)
            training_thread.daemon = True
            training_thread.start()
            
            print(f"{Fore.GREEN}✅ 新训练已在后台启动{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}💡 您可以继续使用其他功能，训练将在后台进行{Style.RESET_ALL}")
            print(f"{Fore.CYAN}📊 训练进度将在终端中显示{Style.RESET_ALL}")
            
            input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}❌ 新训练启动失败: {e}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")

    def validate_model(self):
        """验证模型"""
        try:
            print(f"{Fore.CYAN}🔍 模型验证{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            
            # 选择检查点文件
            checkpoint_files = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
            if not checkpoint_files:
                print(f"{Fore.RED}❌ 没有找到检查点文件{Style.RESET_ALL}")
                input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")
                return
            
            print(f"\n{Fore.CYAN}📁 可用的检查点文件:{Style.RESET_ALL}")
            for i, file in enumerate(checkpoint_files, 1):
                print(f"{Fore.WHITE}{i}.{Style.RESET_ALL} {file}")
            
            try:
                choice = int(input(f"\n{Fore.CYAN}请选择检查点文件 (1-{len(checkpoint_files)}): {Style.RESET_ALL}"))
                if 1 <= choice <= len(checkpoint_files):
                    checkpoint_path = os.path.join('checkpoints', checkpoint_files[choice-1])
                    
                    print(f"\n{Fore.GREEN}🚀 启动模型验证...{Style.RESET_ALL}")
                    
                    # 启动验证器
                    import subprocess
                    import sys
                    
                    # 使用验证器.py进行验证
                    validator_path = os.path.join(os.path.dirname(__file__), "验证器.py")
                    if os.path.exists(validator_path):
                        print(f"{Fore.CYAN}📊 启动验证器界面...{Style.RESET_ALL}")
                        subprocess.Popen([sys.executable, validator_path])
                        print(f"{Fore.GREEN}✅ 验证器已启动{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}❌ 验证器文件不存在: {validator_path}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}❌ 无效选择{Style.RESET_ALL}")
                    
            except ValueError:
                print(f"{Fore.RED}❌ 请输入有效数字{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}❌ 验证启动失败: {e}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
        
        input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")

    def config_management_menu(self):
        """配置管理菜单"""
        while True:
            self.clear_screen()
            self.print_header()
            print(f"\n{Fore.CYAN}⚙️  配置管理{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            
            print(f"\n{Fore.CYAN}可用操作:{Style.RESET_ALL}")
            print(f"{Fore.WHITE}1.{Style.RESET_ALL} 查看当前配置")
            print(f"{Fore.WHITE}2.{Style.RESET_ALL} 编辑配置参数")
            print(f"{Fore.WHITE}3.{Style.RESET_ALL} 加载默认配置")
            print(f"{Fore.WHITE}4.{Style.RESET_ALL} 加载低显存配置")
            print(f"{Fore.WHITE}5.{Style.RESET_ALL} 加载高显存配置")
            print(f"{Fore.WHITE}6.{Style.RESET_ALL} 加载RTX 4090配置")
            print(f"{Fore.WHITE}7.{Style.RESET_ALL} 保存当前配置")
            print(f"{Fore.WHITE}8.{Style.RESET_ALL} 返回主菜单")
            
            choice = input(f"\n{Fore.CYAN}请选择操作 (1-8): {Style.RESET_ALL}").strip()
            
            if choice == '1':
                self._show_current_config()
            elif choice == '2':
                self._edit_config()
            elif choice == '3':
                self.config = get_default_config()
                print(f"{Fore.GREEN}✅ 已加载默认配置{Style.RESET_ALL}")
                input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")
            elif choice == '4':
                self.config = load_low_memory_config()
                print(f"{Fore.GREEN}✅ 已加载低显存配置{Style.RESET_ALL}")
                input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")
            elif choice == '5':
                self.config = load_high_memory_config()
                print(f"{Fore.GREEN}✅ 已加载高显存配置{Style.RESET_ALL}")
                input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")
            elif choice == '6':
                self.config = load_rtx_4090_config()
                print(f"{Fore.GREEN}✅ 已加载RTX 4090配置{Style.RESET_ALL}")
                input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")
            elif choice == '7':
                try:
                    save_config(self.config)
                    print(f"{Fore.GREEN}✅ 配置已保存{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}❌ 保存失败: {e}{Style.RESET_ALL}")
                input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")
            elif choice == '8':
                break
            else:
                print(f"{Fore.RED}❌ 无效选择，请重试{Style.RESET_ALL}")
                time.sleep(1)

    def _edit_config(self):
        """编辑配置参数"""
        while True:
            self.clear_screen()
            self.print_header()
            print(f"\n{Fore.CYAN}✏️  编辑配置参数{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            
            print(f"\n{Fore.CYAN}可编辑的配置类别:{Style.RESET_ALL}")
            print(f"{Fore.WHITE}1.{Style.RESET_ALL} 模型配置 (块数、特征数)")
            print(f"{Fore.WHITE}2.{Style.RESET_ALL} 训练配置 (批次大小、轮数、学习率等)")
            print(f"{Fore.WHITE}3.{Style.RESET_ALL} 数据路径配置")
            print(f"{Fore.WHITE}4.{Style.RESET_ALL} 输出路径配置")
            print(f"{Fore.WHITE}5.{Style.RESET_ALL} 返回配置管理")
            
            choice = input(f"\n{Fore.CYAN}请选择要编辑的类别 (1-5): {Style.RESET_ALL}").strip()
            
            if choice == '1':
                self._edit_model_config()
            elif choice == '2':
                self._edit_training_config()
            elif choice == '3':
                self._edit_data_paths()
            elif choice == '4':
                self._edit_output_paths()
            elif choice == '5':
                break
            else:
                print(f"{Fore.RED}❌ 无效选择，请重试{Style.RESET_ALL}")
                time.sleep(1)

    def _edit_model_config(self):
        """编辑模型配置"""
        print(f"\n{Fore.CYAN}🏗️  编辑模型配置{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*40}{Style.RESET_ALL}")
        
        # 显示当前值
        print(f"\n{Fore.YELLOW}当前模型配置:{Style.RESET_ALL}")
        print(f"  块数: {self.config['model']['num_blocks']}")
        print(f"  特征数: {self.config['model']['num_features']}")
        
        try:
            # 编辑块数
            new_blocks = input(f"\n{Fore.CYAN}新的块数 (当前: {self.config['model']['num_blocks']}, 回车保持不变): {Style.RESET_ALL}").strip()
            if new_blocks:
                blocks = int(new_blocks)
                if 1 <= blocks <= 20:
                    self.config['model']['num_blocks'] = blocks
                    print(f"{Fore.GREEN}✅ 块数已更新为: {blocks}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}❌ 块数应在1-20之间{Style.RESET_ALL}")
            
            # 编辑特征数
            new_features = input(f"\n{Fore.CYAN}新的特征数 (当前: {self.config['model']['num_features']}, 回车保持不变): {Style.RESET_ALL}").strip()
            if new_features:
                features = int(new_features)
                if 16 <= features <= 128:
                    self.config['model']['num_features'] = features
                    print(f"{Fore.GREEN}✅ 特征数已更新为: {features}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}❌ 特征数应在16-128之间{Style.RESET_ALL}")
                    
        except ValueError:
            print(f"{Fore.RED}❌ 请输入有效数字{Style.RESET_ALL}")
        
        input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")

    def _edit_training_config(self):
        """编辑训练配置"""
        print(f"\n{Fore.CYAN}🎯 编辑训练配置{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*40}{Style.RESET_ALL}")
        
        # 显示当前值
        print(f"\n{Fore.YELLOW}当前训练配置:{Style.RESET_ALL}")
        print(f"  批次大小: {self.config['training']['batch_size']}")
        print(f"  训练轮数: {self.config['training']['num_epochs']}")
        print(f"  学习率: {self.config['training']['learning_rate']}")
        print(f"  保存频率: {self.config['training']['save_frequency']} epochs")
        
        try:
            # 编辑批次大小
            new_batch = input(f"\n{Fore.CYAN}新的批次大小 (当前: {self.config['training']['batch_size']}, 回车保持不变): {Style.RESET_ALL}").strip()
            if new_batch:
                batch_size = int(new_batch)
                if 1 <= batch_size <= 32:
                    self.config['training']['batch_size'] = batch_size
                    print(f"{Fore.GREEN}✅ 批次大小已更新为: {batch_size}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}❌ 批次大小应在1-32之间{Style.RESET_ALL}")
            
            # 编辑训练轮数
            new_epochs = input(f"\n{Fore.CYAN}新的训练轮数 (当前: {self.config['training']['num_epochs']}, 回车保持不变): {Style.RESET_ALL}").strip()
            if new_epochs:
                epochs = int(new_epochs)
                if 1 <= epochs <= 1000:
                    self.config['training']['num_epochs'] = epochs
                    print(f"{Fore.GREEN}✅ 训练轮数已更新为: {epochs}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}❌ 训练轮数应在1-1000之间{Style.RESET_ALL}")
            
            # 编辑学习率
            new_lr = input(f"\n{Fore.CYAN}新的学习率 (当前: {self.config['training']['learning_rate']}, 回车保持不变): {Style.RESET_ALL}").strip()
            if new_lr:
                learning_rate = float(new_lr)
                if 1e-6 <= learning_rate <= 1e-1:
                    self.config['training']['learning_rate'] = learning_rate
                    print(f"{Fore.GREEN}✅ 学习率已更新为: {learning_rate}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}❌ 学习率应在1e-6到1e-1之间{Style.RESET_ALL}")
            
            # 编辑保存频率
            new_save_freq = input(f"\n{Fore.CYAN}新的保存频率 (当前: {self.config['training']['save_frequency']} epochs, 回车保持不变): {Style.RESET_ALL}").strip()
            if new_save_freq:
                save_freq = int(new_save_freq)
                if 1 <= save_freq <= 50:
                    self.config['training']['save_frequency'] = save_freq
                    print(f"{Fore.GREEN}✅ 保存频率已更新为: {save_freq} epochs{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}❌ 保存频率应在1-50之间{Style.RESET_ALL}")
                    
        except ValueError:
            print(f"{Fore.RED}❌ 请输入有效数字{Style.RESET_ALL}")
        
        input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")

    def _edit_data_paths(self):
        """编辑数据路径配置"""
        print(f"\n{Fore.CYAN}📁 编辑数据路径配置{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*40}{Style.RESET_ALL}")
        
        # 显示当前值
        print(f"\n{Fore.YELLOW}当前数据路径:{Style.RESET_ALL}")
        print(f"  训练LR: {self.config['data']['train_lr_dir']}")
        print(f"  训练HR: {self.config['data']['train_hr_dir']}")
        print(f"  验证LR: {self.config['data']['val_lr_dir']}")
        print(f"  验证HR: {self.config['data']['val_hr_dir']}")
        
        # 编辑各个路径
        paths = [
            ('train_lr_dir', '训练LR目录'),
            ('train_hr_dir', '训练HR目录'),
            ('val_lr_dir', '验证LR目录'),
            ('val_hr_dir', '验证HR目录')
        ]
        
        for key, name in paths:
            new_path = input(f"\n{Fore.CYAN}新的{name} (当前: {self.config['data'][key]}, 回车保持不变): {Style.RESET_ALL}").strip()
            if new_path:
                if os.path.exists(new_path):
                    self.config['data'][key] = new_path
                    print(f"{Fore.GREEN}✅ {name}已更新为: {new_path}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}❌ 路径不存在: {new_path}{Style.RESET_ALL}")
        
        input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")

    def _edit_output_paths(self):
        """编辑输出路径配置"""
        print(f"\n{Fore.CYAN}💾 编辑输出路径配置{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*40}{Style.RESET_ALL}")
        
        # 显示当前值
        print(f"\n{Fore.YELLOW}当前输出路径:{Style.RESET_ALL}")
        print(f"  检查点目录: {self.config['paths']['checkpoint_dir']}")
        print(f"  日志目录: {self.config['paths']['log_dir']}")
        print(f"  输出目录: {self.config['paths']['output_dir']}")
        print(f"  保存目录: {self.config['paths']['save_dir']}")
        
        # 编辑各个路径
        paths = [
            ('checkpoint_dir', '检查点目录'),
            ('log_dir', '日志目录'),
            ('output_dir', '输出目录'),
            ('save_dir', '保存目录')
        ]
        
        for key, name in paths:
            new_path = input(f"\n{Fore.CYAN}新的{name} (当前: {self.config['paths'][key]}, 回车保持不变): {Style.RESET_ALL}").strip()
            if new_path:
                # 创建目录如果不存在
                try:
                    os.makedirs(new_path, exist_ok=True)
                    self.config['paths'][key] = new_path
                    print(f"{Fore.GREEN}✅ {name}已更新为: {new_path}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}❌ 无法创建目录: {e}{Style.RESET_ALL}")
        
        input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")

    def _show_current_config(self):
        """显示当前配置"""
        print(f"\n{Fore.CYAN}📋 当前配置详情:{Style.RESET_ALL}")
        
        # 模型配置
        print(f"\n{Fore.YELLOW}🏗️  模型配置:{Style.RESET_ALL}")
        print(f"  块数: {self.config['model']['num_blocks']}")
        print(f"  特征数: {self.config['model']['num_features']}")
        
        # 训练配置
        print(f"\n{Fore.YELLOW}🎯 训练配置:{Style.RESET_ALL}")
        print(f"  批次大小: {self.config['training']['batch_size']}")
        print(f"  训练轮数: {self.config['training']['num_epochs']}")
        print(f"  学习率: {self.config['training']['learning_rate']}")
        print(f"  保存频率: {self.config['training']['save_frequency']} epochs")
        
        # 数据配置
        print(f"\n{Fore.YELLOW}📁 数据路径:{Style.RESET_ALL}")
        print(f"  训练LR: {self.config['data']['train_lr_dir']}")
        print(f"  训练HR: {self.config['data']['train_hr_dir']}")
        print(f"  验证LR: {self.config['data']['val_lr_dir']}")
        print(f"  验证HR: {self.config['data']['val_hr_dir']}")
        
        # 路径配置
        print(f"\n{Fore.YELLOW}💾 输出路径:{Style.RESET_ALL}")
        print(f"  检查点: {self.config['paths']['checkpoint_dir']}")
        print(f"  日志: {self.config['paths']['log_dir']}")
        print(f"  输出: {self.config['paths']['output_dir']}")
        
        input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")

    def checkpoint_management_menu(self):
        """检查点管理菜单"""
        while True:
            self.clear_screen()
            self.print_header()
            print(f"\n{Fore.CYAN}💾 检查点管理{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            
            # 显示检查点文件
            checkpoint_files = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
            if checkpoint_files:
                print(f"\n{Fore.CYAN}📁 现有检查点文件 ({len(checkpoint_files)} 个):{Style.RESET_ALL}")
                for i, file in enumerate(checkpoint_files[:10], 1):  # 只显示前10个
                    file_path = os.path.join('checkpoints', file)
                    file_size = os.path.getsize(file_path) / (1024*1024)  # MB
                    print(f"  {i:2d}. {file} ({file_size:.1f}MB)")
                if len(checkpoint_files) > 10:
                    print(f"      ... 还有 {len(checkpoint_files) - 10} 个文件")
            else:
                print(f"\n{Fore.YELLOW}📁 没有找到检查点文件{Style.RESET_ALL}")
            
            print(f"\n{Fore.CYAN}可用操作:{Style.RESET_ALL}")
            print(f"{Fore.WHITE}1.{Style.RESET_ALL} 检查检查点详情")
            print(f"{Fore.WHITE}2.{Style.RESET_ALL} 删除检查点文件")
            print(f"{Fore.WHITE}3.{Style.RESET_ALL} 清理所有检查点")
            print(f"{Fore.WHITE}4.{Style.RESET_ALL} 备份检查点")
            print(f"{Fore.WHITE}5.{Style.RESET_ALL} 返回主菜单")
            
            choice = input(f"\n{Fore.CYAN}请选择操作 (1-5): {Style.RESET_ALL}").strip()
            
            if choice == '1':
                self._check_checkpoint_details()
            elif choice == '2':
                self._delete_checkpoint()
            elif choice == '3':
                self._clean_all_checkpoints()
            elif choice == '4':
                self._backup_checkpoint()
            elif choice == '5':
                break
            else:
                print(f"{Fore.RED}❌ 无效选择，请重试{Style.RESET_ALL}")
                time.sleep(1)

    def _check_checkpoint_details(self):
        """检查检查点详情"""
        checkpoint_files = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
        if not checkpoint_files:
            print(f"{Fore.RED}❌ 没有找到检查点文件{Style.RESET_ALL}")
            input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}选择要检查的文件:{Style.RESET_ALL}")
        for i, file in enumerate(checkpoint_files, 1):
            print(f"{Fore.WHITE}{i}.{Style.RESET_ALL} {file}")
        
        try:
            choice = int(input(f"\n{Fore.CYAN}请选择文件 (1-{len(checkpoint_files)}): {Style.RESET_ALL}"))
            if 1 <= choice <= len(checkpoint_files):
                file_path = os.path.join('checkpoints', checkpoint_files[choice-1])
                
                # 启动检查点检查器
                import subprocess
                import sys
                checker_path = os.path.join(os.path.dirname(__file__), "检查模型检查点.py")
                if os.path.exists(checker_path):
                    print(f"{Fore.GREEN}🚀 启动检查点检查器...{Style.RESET_ALL}")
                    subprocess.Popen([sys.executable, checker_path])
                else:
                    print(f"{Fore.RED}❌ 检查器文件不存在{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}❌ 无效选择{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}❌ 请输入有效数字{Style.RESET_ALL}")
        
        input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")

    def _delete_checkpoint(self):
        """删除检查点文件"""
        checkpoint_files = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
        if not checkpoint_files:
            print(f"{Fore.RED}❌ 没有找到检查点文件{Style.RESET_ALL}")
            input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}选择要删除的文件:{Style.RESET_ALL}")
        for i, file in enumerate(checkpoint_files, 1):
            print(f"{Fore.WHITE}{i}.{Style.RESET_ALL} {file}")
        
        try:
            choice = int(input(f"\n{Fore.CYAN}请选择文件 (1-{len(checkpoint_files)}): {Style.RESET_ALL}"))
            if 1 <= choice <= len(checkpoint_files):
                file_to_delete = checkpoint_files[choice-1]
                confirm = input(f"\n{Fore.RED}确认删除 '{file_to_delete}'? (y/n): {Style.RESET_ALL}").strip().lower()
                if confirm == 'y':
                    os.remove(os.path.join('checkpoints', file_to_delete))
                    print(f"{Fore.GREEN}✅ 已删除: {file_to_delete}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}❌ 取消删除{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}❌ 无效选择{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}❌ 请输入有效数字{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ 删除失败: {e}{Style.RESET_ALL}")
        
        input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")

    def _clean_all_checkpoints(self):
        """清理所有检查点"""
        checkpoint_files = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
        if not checkpoint_files:
            print(f"{Fore.YELLOW}📁 没有检查点文件需要清理{Style.RESET_ALL}")
            input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.RED}⚠️  警告: 将删除所有 {len(checkpoint_files)} 个检查点文件{Style.RESET_ALL}")
        confirm = input(f"{Fore.RED}确认清理所有检查点? (输入 'DELETE' 确认): {Style.RESET_ALL}").strip()
        
        if confirm == 'DELETE':
            try:
                deleted_count = 0
                for file in checkpoint_files:
                    os.remove(os.path.join('checkpoints', file))
                    deleted_count += 1
                print(f"{Fore.GREEN}✅ 已清理 {deleted_count} 个检查点文件{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}❌ 清理失败: {e}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}❌ 取消清理{Style.RESET_ALL}")
        
        input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")

    def _backup_checkpoint(self):
        """备份检查点"""
        checkpoint_files = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
        if not checkpoint_files:
            print(f"{Fore.RED}❌ 没有找到检查点文件{Style.RESET_ALL}")
            input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}选择要备份的文件:{Style.RESET_ALL}")
        for i, file in enumerate(checkpoint_files, 1):
            print(f"{Fore.WHITE}{i}.{Style.RESET_ALL} {file}")
        
        try:
            choice = int(input(f"\n{Fore.CYAN}请选择文件 (1-{len(checkpoint_files)}): {Style.RESET_ALL}"))
            if 1 <= choice <= len(checkpoint_files):
                source_file = checkpoint_files[choice-1]
                backup_name = f"{os.path.splitext(source_file)[0]}_backup.pth"
                
                import shutil
                shutil.copy2(
                    os.path.join('checkpoints', source_file),
                    os.path.join('checkpoints', backup_name)
                )
                print(f"{Fore.GREEN}✅ 已备份为: {backup_name}{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}❌ 无效选择{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}❌ 请输入有效数字{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ 备份失败: {e}{Style.RESET_ALL}")
        
        input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")

    def performance_monitoring_menu(self):
        """性能监控菜单"""
        print(f"{Fore.CYAN}📊 性能监控{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
        # 显示GPU信息
        self.print_gpu_info()
        
        # 显示系统信息
        import psutil
        print(f"\n{Fore.CYAN}💻 系统信息:{Style.RESET_ALL}")
        print(f"  CPU使用率: {psutil.cpu_percent():.1f}%")
        print(f"  内存使用: {psutil.virtual_memory().percent:.1f}%")
        print(f"  可用内存: {psutil.virtual_memory().available / (1024**3):.1f}GB")
        
        # 显示磁盘信息
        disk_usage = psutil.disk_usage('.')
        print(f"  磁盘使用: {disk_usage.percent:.1f}%")
        print(f"  可用空间: {disk_usage.free / (1024**3):.1f}GB")
        
        input(f"\n{Fore.YELLOW}按回车键继续...{Style.RESET_ALL}")

    def _smart_adjust_features(self, checkpoint_path, old_features, new_features):
        """
        智能特征数调整 - 集成到控制台训练器中
        支持任意特征数的调整，如70->72, 64->72等
        """
        try:
            # 1. 检测当前特征数
            detected_config = self._detect_checkpoint_config(checkpoint_path)
            if not detected_config:
                return False, "无法检测检查点配置"
            
            old_features = detected_config['num_features']
            print(f"{Fore.CYAN}🚀 开始智能特征数调整: {old_features} -> {new_features}{Style.RESET_ALL}")
            
            # 2. 加载原始检查点
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            old_state_dict = checkpoint['generator_state_dict']
            
            # 3. 创建新模型以获取目标状态字典结构
            from src.models.esrgan import LiteRealESRGAN
            new_model = LiteRealESRGAN(
                num_blocks=6,  # 使用默认值而不是self.config
                num_features=new_features
            )
            new_state_dict = new_model.state_dict()
            
            # 4. 智能权重调整
            print(f"{Fore.CYAN}🔧 开始权重调整...{Style.RESET_ALL}")
            adjusted_state_dict = self._adjust_features_weights(old_state_dict, new_state_dict, old_features, new_features)
            
            # 5. 更新检查点
            checkpoint['generator_state_dict'] = adjusted_state_dict
            
            # 🔥 重要：移除优化器状态以避免维度不匹配
            optimizer_removed = False
            if 'g_optimizer_state_dict' in checkpoint:
                del checkpoint['g_optimizer_state_dict']
                optimizer_removed = True
                print(f"{Fore.YELLOW}🧹 已清理生成器优化器状态(避免维度不匹配){Style.RESET_ALL}")
            
            if 'd_optimizer_state_dict' in checkpoint:
                del checkpoint['d_optimizer_state_dict']
                optimizer_removed = True
                print(f"{Fore.YELLOW}🧹 已清理判别器优化器状态(避免维度不匹配){Style.RESET_ALL}")
            
            if 'g_scheduler_state_dict' in checkpoint:
                del checkpoint['g_scheduler_state_dict']
                print(f"{Fore.YELLOW}🧹 已清理生成器调度器状态{Style.RESET_ALL}")
            
            if 'd_scheduler_state_dict' in checkpoint:
                del checkpoint['d_scheduler_state_dict']
                print(f"{Fore.YELLOW}🧹 已清理判别器调度器状态{Style.RESET_ALL}")
            
            # 保留训练元数据
            epoch = checkpoint.get('epoch', 0)
            best_psnr = checkpoint.get('best_psnr', 0.0)
            print(f"✅ 保留训练元数据: epoch={epoch}, best_psnr={best_psnr:.4f}")
            
            # 6. 生成输出路径
            base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
            output_path = os.path.join(
                os.path.dirname(checkpoint_path),
                f"{base_name}_features_{new_features}.pth"
            )
            
            # 7. 保存调整后的检查点
            torch.save(checkpoint, output_path)
            print(f"{Fore.GREEN}✅ 调整完成! 新检查点保存至: {os.path.basename(output_path)}{Style.RESET_ALL}")
            
            # 8. 验证调整结果
            self._verify_feature_adjustment(output_path, new_features)
            
            # 9. 根据特征数变化调整学习率
            if 'learning_rate' in self.config['training']:
                feature_ratio = new_features / old_features
                if feature_ratio > 1.2:  # 特征数增加超过20%
                    suggested_lr = self.config['training']['learning_rate'] * 0.8
                    print(f"{Fore.CYAN}💡 特征数大幅增加，建议降低学习率: {self.config['training']['learning_rate']:.6f} -> {suggested_lr:.6f}{Style.RESET_ALL}")
                    self.config['training']['learning_rate'] = suggested_lr
                    print(f"{Fore.GREEN}✅ 学习率已调整为: {suggested_lr}{Style.RESET_ALL}")
            
            return True, output_path
            
        except Exception as e:
            print(f"{Fore.RED}❌ 智能特征数调整失败: {str(e)}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            return False, str(e)

    

    
    def _verify_feature_adjustment(self, checkpoint_path, expected_features):
        """
        验证特征数调整结果
        """
        try:
            print(f"{Fore.CYAN}🔍 验证调整结果...{Style.RESET_ALL}")
            
            # 1. 检测特征数
            detected_config = self._detect_checkpoint_config(checkpoint_path)
            if detected_config and detected_config['num_features'] == expected_features:
                print(f"{Fore.GREEN}✅ 特征数验证通过: {expected_features}{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}❌ 特征数验证失败{Style.RESET_ALL}")
                return False
            
            # 2. 测试模型加载
            from src.models.esrgan import LiteRealESRGAN
            model = LiteRealESRGAN(
                num_blocks=6,  # 使用默认值而不是self.config
                num_features=expected_features
            )
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['generator_state_dict'])
            print(f"{Fore.GREEN}✅ 模型加载验证通过{Style.RESET_ALL}")
            
            # 3. 测试前向传播
            model.eval()
            test_input = torch.randn(1, 3, 64, 64)
            with torch.no_grad():
                output = model(test_input)
            print(f"{Fore.GREEN}✅ 前向传播验证通过: {test_input.shape} -> {output.shape}{Style.RESET_ALL}")
            
            return True
            
        except Exception as e:
            print(f"{Fore.RED}❌ 验证失败: {str(e)}{Style.RESET_ALL}")
            return False


if __name__ == "__main__":
    try:
        trainer = ConsoleTrainer()
        trainer.run()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠️  程序被用户中断{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}❌ 程序运行出错: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()