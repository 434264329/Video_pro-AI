import os
import glob
import time
import json
import threading
from datetime import datetime
import numpy as np
from PIL import Image
import torch

from .predictor import ImageSuperResolution
from ..utils.metrics import calculate_psnr, calculate_ssim

class ValidationManager:
    """验证管理器 - 负责模型验证和结果分析"""
    
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
            'on_error': []
        }
        
        # 验证结果
        self.validation_results = []
        
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
    
    def load_model(self, model_path, device=None):
        """加载模型"""
        try:
            if device is None:
                device = self.config.get('training', {}).get('device', 'cuda') if self.config else 'cuda'
            
            self.predictor = ImageSuperResolution(model_path, device)
            self.model_path = model_path
            return True
        except Exception as e:
            self._trigger_callback('on_error', f"模型加载失败: {str(e)}")
            return False
    
    def start_validation(self, lr_folder, hr_folder=None):
        """开始验证"""
        if self.is_validating:
            return False
            
        if not self.predictor:
            self._trigger_callback('on_error', "请先加载模型")
            return False
        
        self.is_validating = True
        self.validation_thread = threading.Thread(
            target=self._validation_loop, 
            args=(lr_folder, hr_folder)
        )
        self.validation_thread.daemon = True
        self.validation_thread.start()
        
        return True
    
    def stop_validation(self):
        """停止验证"""
        self.is_validating = False
        if self.validation_thread:
            self.validation_thread.join(timeout=5)
    
    def _validation_loop(self, lr_folder, hr_folder):
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
    
    def _process_single_image(self, lr_file, hr_file):
        """处理单张图像"""
        try:
            # 加载LR图像
            lr_img = Image.open(lr_file).convert('RGB')
            
            # 超分辨率处理
            start_time = time.time()
            sr_img = self.predictor.process_image(lr_img)  # 修改方法名
            processing_time = (time.time() - start_time) * 1000  # 转换为毫秒
            
            result = {
                'name': os.path.basename(lr_file),
                'lr_image': lr_img,
                'sr_image': sr_img,
                'psnr': 0.0,
                'ssim': 0.0,
                'time': processing_time
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
            return psnr
            
        except Exception as e:
            print(f"PSNR计算错误: {e}")
            return 0.0
    
    def _calculate_ssim(self, img1, img2):
        """计算SSIM"""
        try:
            # 转换为numpy数组
            arr1 = np.array(img1, dtype=np.float32) / 255.0
            arr2 = np.array(img2, dtype=np.float32) / 255.0
            
            # 简化的SSIM计算
            mu1 = np.mean(arr1)
            mu2 = np.mean(arr2)
            sigma1 = np.var(arr1)
            sigma2 = np.var(arr2)
            sigma12 = np.mean((arr1 - mu1) * (arr2 - mu2))
            
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
            
            return ssim
            
        except Exception as e:
            print(f"SSIM计算错误: {e}")
            return 0.0
    
    def get_validation_results(self):
        """获取验证结果"""
        return self.validation_results
    
    def save_results(self, output_path):
        """保存验证结果"""
        try:
            results_data = []
            for result in self.validation_results:
                results_data.append({
                    'name': result['name'],
                    'psnr': result['psnr'],
                    'ssim': result['ssim'],
                    'time': result['time']
                })
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'model_path': self.model_path,
                    'total_images': len(results_data),
                    'average_psnr': np.mean([r['psnr'] for r in results_data]) if results_data else 0,
                    'average_ssim': np.mean([r['ssim'] for r in results_data]) if results_data else 0,
                    'average_time': np.mean([r['time'] for r in results_data]) if results_data else 0,
                    'results': results_data
                }, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"保存结果失败: {e}")
            return False
    
    def export_images(self, output_folder):
        """导出处理后的图像"""
        try:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            for result in self.validation_results:
                # 保存SR图像
                sr_path = os.path.join(output_folder, f"sr_{result['name']}")
                result['sr_image'].save(sr_path)
                
                # 如果有HR图像，也保存
                if 'hr_image' in result:
                    hr_path = os.path.join(output_folder, f"hr_{result['name']}")
                    result['hr_image'].save(hr_path)
            
            return True
            
        except Exception as e:
            print(f"导出图像失败: {e}")
            return False