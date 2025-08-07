import os
import sys
import json
import time
import torch
import threading
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
        else:
            print(f"{Fore.RED}❌ CUDA不可用，无需禁用TF32{Style.RESET_ALL}")
    
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
            
            # 3. 选择调整策略
            feature_diff = abs(new_features - old_features)
            
            if use_progressive and feature_diff > 20:
                # 大幅调整使用渐进式
                print(f"📈 特征数变化较大({feature_diff})，使用渐进式调整")
                success, result = self._progressive_feature_adjust(checkpoint_path, old_features, new_features, steps)
            else:
                # 小幅调整使用直接调整
                print(f"🔧 特征数变化较小({feature_diff})，使用直接调整")
                success, result = self._smart_adjust_features(checkpoint_path, old_features, new_features)
            
            if success:
                # 4. 最终验证
                if self._validate_rrdb_structure(torch.load(result, weights_only=False)['generator_state_dict'], new_features):
                    print(f"{Fore.GREEN}🎉 智能特征数调整成功完成!{Style.RESET_ALL}")
                    return True, result
                else:
                    print(f"{Fore.RED}❌ 最终验证失败{Style.RESET_ALL}")
                    return False, "最终验证失败"
            else:
                print(f"{Fore.RED}❌ 调整过程失败: {result}{Style.RESET_ALL}")
                return False, result
                
        except Exception as e:
            print(f"{Fore.RED}❌ 增强版调整失败: {str(e)}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            return False, str(e)

    def _detect_checkpoint_config(self, checkpoint_path):
        """检测检查点的配置信息"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'generator_state_dict' in checkpoint:
                return self._extract_model_config_from_state_dict(checkpoint['generator_state_dict'])
            return self._extract_model_config_from_state_dict(checkpoint)
        except Exception as e:
            print(f"{Fore.RED}❌ 检测检查点配置失败: {str(e)}{Style.RESET_ALL}")
            return None

    def _validate_rrdb_structure(self, state_dict, target_features):
        """验证RRDB结构是否符合目标特征数"""
        try:
            # 检查第一个卷积层
            if 'conv_first.weight' in state_dict:
                if state_dict['conv_first.weight'].shape[0] != target_features:
                    print(f"{Fore.RED}❌ conv_first.weight 特征数不匹配: {state_dict['conv_first.weight'].shape[0]} != {target_features}{Style.RESET_ALL}")
                    return False
            
            # 检查RRDB块的第一个卷积层
            for key in state_dict:
                if key.startswith('rrdb_blocks.') and '.dense1.conv1.weight' in key:
                    if state_dict[key].shape[1] != target_features:
                        print(f"{Fore.RED}❌ {key} 输入通道不匹配: {state_dict[key].shape[1]} != {target_features}{Style.RESET_ALL}")
                        return False
            return True
        except Exception as e:
            print(f"{Fore.RED}❌ 验证RRDB结构失败: {str(e)}{Style.RESET_ALL}")
            return False

    def _verify_feature_adjustment(self, checkpoint_path, target_features):
        """验证特征数调整结果"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint['generator_state_dict']
            return self._validate_rrdb_structure(state_dict, target_features)
        except Exception as e:
            print(f"{Fore.RED}❌ 验证调整结果失败: {str(e)}{Style.RESET_ALL}")
            return False

    def _smart_adjust_features(self, checkpoint_path, old_features, new_features):
        """智能调整特征数的核心方法"""
        try:
            print(f"{Fore.CYAN}🚀 开始智能特征数调整: {old_features} -> {new_features}{Style.RESET_ALL}")
            
            # 加载检查点
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            old_state_dict = checkpoint['generator_state_dict']
            
            # 创建目标模型
            from src.models.esrgan import LiteRealESRGAN
            target_model = LiteRealESRGAN(
                num_blocks=6,  # 假设块数不变
                num_features=new_features
            )
            new_state_dict = target_model.state_dict()
            
            # 调整权重
            adjusted_state_dict = self._adjust_features_weights(old_state_dict, new_state_dict, old_features, new_features)
            
            # 更新检查点
            checkpoint['generator_state_dict'] = adjusted_state_dict
            
            # 保存结果
            output_path = checkpoint_path.replace('.pth', f'_features_{new_features}.pth')
            torch.save(checkpoint, output_path)
            print(f"{Fore.GREEN}✅ 特征数调整完成，保存至: {os.path.basename(output_path)}{Style.RESET_ALL}")
            
            return True, output_path
            
        except Exception as e:
            print(f"{Fore.RED}❌ 智能特征数调整失败: {str(e)}{Style.RESET_ALL}")
            return False, str(e)
    
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
        block_idx = int(parts[1]) if len(parts) > 1 else 0
        dense_part = parts[2] if len(parts) > 2 else ""
        conv_part = parts[3] if len(parts) > 3 else ""
        
        # 提取dense和conv的索引
        try:
            dense_idx = int(dense_part.replace('dense', ''))
            conv_idx = int(conv_part.replace('conv', ''))
        except (ValueError, IndexError):
            print(f"{Fore.YELLOW}⚠️  无法解析层索引，使用默认计算: {key}{Style.RESET_ALL}")
            dense_idx = 1
            conv_idx = 1
        
        # 计算当前层在密集块中的序号
        layer_order = (dense_idx - 1) * 5 + conv_idx
        print(f"    🔗 RRDB Block{block_idx} Dense{dense_idx} Conv{conv_idx} (序号:{layer_order})")
        
        # 动态计算增长率（从实际通道数反推）
        if layer_order == 1:
            # 第一层的输入通道应该等于基础特征数
            growth_rate = 32  # 默认为32，第一层不依赖增长率
        else:
            # 从实际通道数反推增长率
            growth_rate = max(1, (old_in - old_features) // (layer_order - 1))
        print(f"    🔢 动态计算增长率: {growth_rate}")
        
        # 计算预期通道数
        expected_old_in = old_features + growth_rate * (layer_order - 1)
        expected_new_in = new_features + growth_rate * (layer_order - 1)
        print(f"    📊 通道计算: {old_features}+{growth_rate}×{layer_order-1} = {expected_old_in}")
        print(f"    🎯 目标通道: {new_features}+{growth_rate}×{layer_order-1} = {expected_new_in}")
        
        # 验证实际输入通道数
        if old_in != expected_old_in:
            print(f"    ⚠️  通道数异常: 实际({old_in}) != 预期({expected_old_in})")
            # 基于实际值重新计算增长率
            growth_rate = max(1, (old_in - old_features) // max(1, layer_order - 1))
            expected_new_in = new_features + growth_rate * (layer_order - 1)
            print(f"    🔄 修正增长率: {growth_rate}, 修正目标通道: {expected_new_in}")
        
        # 确保目标输入通道合理
        target_in = new_in
        if abs(expected_new_in - new_in) > 10:  # 允许小范围误差
            print(f"    ⚠️  目标通道不匹配: 计算({expected_new_in}) vs 实际({new_in})")
            target_in = expected_new_in
            # 创建调整后的目标形状
            new_weight_tensor = torch.zeros((new_out, target_in, new_h, new_w), 
                                          dtype=old_weight.dtype, 
                                          device=old_weight.device)
        else:
            new_weight_tensor = torch.zeros_like(new_weight)
        
        # 智能权重复制和扩展
        min_in = min(old_in, target_in)
        
        # 复制现有权重
        new_weight_tensor[:, :min_in] = old_weight[:, :min_in].clone()
        
        if target_in > old_in:
            # 扩展输入通道
            remaining = target_in - old_in
            print(f"    📈 扩展密集连接输入通道: +{remaining}")
            
            # 使用智能扩展策略
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
                    # 大量扩展使用Kaiming初始化
                    new_channels = new_weight_tensor[:, old_in:target_in]
                    torch.nn.init.kaiming_normal_(new_channels, mode='fan_in', nonlinearity='leaky_relu')
                    noise = torch.randn_like(new_channels) * 0.01
                    new_weight_tensor[:, old_in:target_in] = new_channels + noise
        
        elif target_in < old_in:
            # 收缩输入通道
            print(f"    📉 收缩密集连接输入通道: -{old_in - target_in}")
        
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
            new_weight_tensor[:min_out] = old_weight[:min_out].clone()
            
            if new_out > old_out:
                # 扩展输出通道
                remaining = new_out - old_out
                print(f"    📈 扩展输出通道: +{remaining}")
                
                # 使用Kaiming初始化新增部分
                with torch.no_grad():
                    torch.nn.init.kaiming_normal_(new_weight_tensor[old_out:], mode='fan_out', nonlinearity='leaky_relu')
        
        elif old_in == old_features and new_in == new_features:
            # 输入通道调整
            min_in = min(old_in, new_in)
            new_weight_tensor[:, :min_in] = old_weight[:, :min_in].clone()
            
            if new_in > old_in:
                # 扩展输入通道
                remaining = new_in - old_in
                print(f"    📈 扩展输入通道: +{remaining}")
                
                # 使用Kaiming初始化新增部分
                with torch.no_grad():
                    torch.nn.init.kaiming_normal_(new_weight_tensor[:, old_in:], mode='fan_in', nonlinearity='leaky_relu')
        
        return new_weight_tensor

    def _adjust_feature_bias(self, old_bias, new_bias, old_features, new_features):
        """
        调整特征相关的偏置
        """
        new_bias_tensor = torch.zeros_like(new_bias)
        min_features = min(old_features, new_features)
        new_bias_tensor[:min_features] = old_bias[:min_features].clone()
        
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
        
        new_weight[:min_out, :min_in, :min_h, :min_w] = old_weight[:min_out, :min_in, :min_h, :min_w].clone()
        
        # 对于新增的部分，使用Kaiming初始化
        if new_out > old_out or new_in > old_in:
            with torch.no_grad():
                # 保存已复制部分
                copied_part = new_weight[:min_out, :min_in, :min_h, :min_w].clone()
                # 整体初始化
                torch.nn.init.kaiming_normal_(new_weight, mode='fan_in', nonlinearity='leaky_relu')
                # 恢复已复制部分
                new_weight[:min_out, :min_in, :min_h, :min_w] = copied_part
        
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
        final_path = os.path.join(checkpoint_dir, f"{base_name}_features_{new_features}.pth")
        shutil.copy2(step_output_path, final_path)
        return True, final_path

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