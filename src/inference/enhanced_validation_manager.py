import os
import glob
import time
import json
import threading
from datetime import datetime
import numpy as np
from PIL import Image
import torch

from .enhanced_predictor import EnhancedSuperResolutionPredictor
from ..utils.metrics import calculate_psnr, calculate_ssim

class EnhancedValidationManager:
    """增强版验证管理器 - 支持大图片分块处理和模型属性自动读取"""
    
    def __init__(self, model_path=None, config=None):
        self.model_path = model_path
        self.config = config
        self.predictor = None
        self.is_validating = False
        self.validation_thread = None
        self.callbacks = {
            'on_validation_start': [],
            'on_validation_end': [],
            'on_progress_update': [],
            'on_image_processed': [],
            'on_error': [],
            'on_model_info_loaded': []  # 新增模型信息加载回调
        }
        
        # 验证结果
        self.validation_results = []
        self.model_info = {}
        
        # 如果提供了模型路径，立即加载
        if model_path:
            self.load_model(model_path)
        
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
    
    def load_model(self, model_path, device=None, tile_size=480):
        """加载模型"""
        try:
            if device is None:
                device = self.config.get('training', {}).get('device', 'cuda') if self.config else 'cuda'
            
            self.predictor = EnhancedSuperResolutionPredictor(model_path, device)
            self.predictor.set_tile_size(tile_size)
            self.model_path = model_path
            
            # 获取模型信息
            self.model_info = self.predictor.get_model_info()
            self._trigger_callback('on_model_info_loaded', self.model_info)
            
            return True
        except Exception as e:
            self._trigger_callback('on_error', f"模型加载失败: {str(e)}")
            return False
    
    def get_model_info(self):
        """获取模型信息"""
        return self.model_info.copy()
    
    def set_tile_size(self, tile_size):
        """设置分块大小"""
        if self.predictor:
            self.predictor.set_tile_size(tile_size)
    
    def estimate_processing_requirements(self, image_path_or_size):
        """估算处理需求"""
        if isinstance(image_path_or_size, str):
            # 从文件路径获取图像尺寸
            with Image.open(image_path_or_size) as img:
                image_size = img.size
        else:
            image_size = image_path_or_size
            
        if self.predictor:
            return self.predictor.estimate_memory_usage(image_size)
        return None
    
    def start_validation(self, lr_folder, hr_folder=None, auto_tile_size=True):
        """开始验证"""
        if self.is_validating:
            return False
            
        if not self.predictor:
            self._trigger_callback('on_error', "请先加载模型")
            return False
        
        self.is_validating = True
        self.validation_thread = threading.Thread(
            target=self._validation_loop, 
            args=(lr_folder, hr_folder, auto_tile_size)
        )
        self.validation_thread.daemon = True
        self.validation_thread.start()
        
        return True
    
    def stop_validation(self):
        """停止验证"""
        self.is_validating = False
        if self.validation_thread:
            self.validation_thread.join(timeout=5)
    
    def _validation_loop(self, lr_folder, hr_folder, auto_tile_size):
        """验证循环"""
        try:
            # 重置结果
            self.validation_results = []
            
            # 获取图像文件列表
            lr_files = self._get_image_files(lr_folder)
            
            if hr_folder:
                hr_files = self._get_image_files(hr_folder)
                matched_files = self._match_files(lr_files, hr_files)
            else:
                # 只有LR图像，用于推理
                matched_files = [(f, None) for f in lr_files]
            
            if not matched_files:
                self._trigger_callback('on_error', "未找到图像文件")
                return
            
            total_files = len(matched_files)
            self._trigger_callback('on_validation_start', total_files)
            
            # 如果启用自动分块大小，分析第一张图像来设置合适的分块大小
            if auto_tile_size and matched_files:
                first_lr_file = matched_files[0][0]
                self._auto_adjust_tile_size(first_lr_file)
            
            for i, (lr_file, hr_file) in enumerate(matched_files):
                if not self.is_validating:
                    break
                
                # 处理单张图像
                result = self._process_single_image(lr_file, hr_file)
                
                if result:
                    self.validation_results.append(result)
                    
                    # 触发单张图像处理完成回调
                    self._trigger_callback('on_image_processed', result)
                
                # 更新进度
                self._trigger_callback('on_progress_update', i + 1, total_files)
            
            # 验证完成
            self._trigger_callback('on_validation_end', self.validation_results)
            
        except Exception as e:
            self._trigger_callback('on_error', f"验证过程中发生错误: {str(e)}")
        finally:
            self.is_validating = False
    
    def _auto_adjust_tile_size(self, sample_image_path):
        """自动调整分块大小"""
        try:
            with Image.open(sample_image_path) as img:
                w, h = img.size
                
            # 估算内存需求
            memory_info = self.predictor.estimate_memory_usage((w, h))
            recommended_size = memory_info['recommended_tile_size']
            
            if recommended_size != self.predictor.max_tile_size:
                print(f"根据图像尺寸 {w}x{h} 自动调整分块大小: {self.predictor.max_tile_size} -> {recommended_size}")
                self.predictor.set_tile_size(recommended_size)
                
                # 触发回调通知UI更新
                self._trigger_callback('on_error', f"自动调整分块大小为: {recommended_size}x{recommended_size}")
                
        except Exception as e:
            print(f"自动调整分块大小失败: {e}")
    
    def _process_single_image(self, lr_file, hr_file):
        """处理单张图像"""
        try:
            # 加载LR图像
            lr_img = Image.open(lr_file).convert('RGB')
            
            print(f"处理图像: {os.path.basename(lr_file)} ({lr_img.size[0]}x{lr_img.size[1]})")
            
            # 超分辨率处理
            start_time = time.time()
            sr_img = self.predictor.process_image_with_tiling(lr_img)
            processing_time = (time.time() - start_time) * 1000  # 转换为毫秒
            
            result = {
                'name': os.path.basename(lr_file),
                'lr_image': lr_img,
                'sr_image': sr_img,
                'psnr': 0.0,
                'ssim': 0.0,
                'time': processing_time,
                'input_size': lr_img.size,
                'output_size': sr_img.size,
                'scale_factor': sr_img.size[0] / lr_img.size[0]
            }
            
            # 如果有HR图像，计算指标
            if hr_file and os.path.exists(hr_file):
                hr_img = Image.open(hr_file).convert('RGB')
                
                # 确保尺寸匹配
                hr_img = hr_img.resize(sr_img.size, Image.Resampling.LANCZOS)
                
                # 计算指标
                result['hr_image'] = hr_img
                result['psnr'] = self._calculate_psnr(sr_img, hr_img)
                result['ssim'] = self._calculate_ssim(sr_img, hr_img)
            
            print(f"处理完成: {processing_time:.1f}ms, PSNR: {result['psnr']:.2f}dB, SSIM: {result['ssim']:.4f}")
            
            return result
            
        except Exception as e:
            print(f"处理图像 {lr_file} 时出错: {e}")
            return None
    
    def _get_image_files(self, folder):
        """获取图像文件列表"""
        if not os.path.exists(folder):
            return []
            
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
            files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
        return sorted(files)
    
    def _match_files(self, lr_files, hr_files):
        """匹配LR和HR文件"""
        matched = []
        lr_names = {os.path.splitext(os.path.basename(f))[0]: f for f in lr_files}
        hr_names = {os.path.splitext(os.path.basename(f))[0]: f for f in hr_files}
        
        for name in lr_names:
            if name in hr_names:
                matched.append((lr_names[name], hr_names[name]))
        
        return matched
    
    def _calculate_psnr(self, img1, img2):
        """计算PSNR"""
        try:
            # 转换为numpy数组
            arr1 = np.array(img1, dtype=np.float32)
            arr2 = np.array(img2, dtype=np.float32)
            
            # 计算MSE
            mse = np.mean((arr1 - arr2) ** 2)
            if mse == 0:
                return float('inf')
            
            # 计算PSNR
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            return float(psnr)
            
        except Exception as e:
            print(f"PSNR计算错误: {e}")
            return 0.0
    
    def _calculate_ssim(self, img1, img2):
        """计算SSIM"""
        try:
            return calculate_ssim(img1, img2)
        except Exception as e:
            print(f"SSIM计算错误: {e}")
            return 0.0
    
    def get_validation_status(self):
        """获取验证状态"""
        return {
            'is_validating': self.is_validating,
            'total_processed': len(self.validation_results),
            'results': self.validation_results,
            'model_info': self.model_info
        }
    
    def save_validation_report(self, filepath):
        """保存验证报告"""
        if not self.validation_results:
            return False
            
        try:
            # 计算统计信息
            psnr_values = [r['psnr'] for r in self.validation_results if r['psnr'] > 0]
            ssim_values = [r['ssim'] for r in self.validation_results if r['ssim'] > 0]
            time_values = [r['time'] for r in self.validation_results]
            
            report = {
                'model_info': self.model_info,
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_images': len(self.validation_results),
                    'avg_psnr': np.mean(psnr_values) if psnr_values else 0,
                    'avg_ssim': np.mean(ssim_values) if ssim_values else 0,
                    'max_psnr': max(psnr_values) if psnr_values else 0,
                    'max_ssim': max(ssim_values) if ssim_values else 0,
                    'avg_processing_time': np.mean(time_values),
                    'total_processing_time': sum(time_values)
                },
                'results': [
                    {
                        'name': r['name'],
                        'psnr': r['psnr'],
                        'ssim': r['ssim'],
                        'time': r['time'],
                        'input_size': r['input_size'],
                        'output_size': r['output_size'],
                        'scale_factor': r['scale_factor']
                    }
                    for r in self.validation_results
                ]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"保存验证报告失败: {e}")
            return False
    
    def export_images(self, output_folder):
        """导出验证图像"""
        if not self.validation_results:
            return False
            
        try:
            os.makedirs(output_folder, exist_ok=True)
            
            for result in self.validation_results:
                name = os.path.splitext(result['name'])[0]
                
                # 保存LR图像
                lr_path = os.path.join(output_folder, f"{name}_lr.png")
                result['lr_image'].save(lr_path)
                
                # 保存SR图像
                sr_path = os.path.join(output_folder, f"{name}_sr.png")
                result['sr_image'].save(sr_path)
                
                # 保存HR图像（如果存在）
                if 'hr_image' in result:
                    hr_path = os.path.join(output_folder, f"{name}_hr.png")
                    result['hr_image'].save(hr_path)
            
            return True
            
        except Exception as e:
            print(f"导出图像失败: {e}")
            return False