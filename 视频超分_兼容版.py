#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频超分辨率处理工具 - 兼容版
支持多种视频格式，包括AV1、MPEG、DAT等
具备空间节省模式、GPU加速、可视化界面等功能
增强模型兼容性，自动检测模型架构参数
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import time
import json
import shutil
import psutil
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib

# 配置中文字体支持
try:
    # 尝试设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False
    # 测试中文字体是否可用
    test_fig = plt.figure()
    test_ax = test_fig.add_subplot(111)
    test_ax.text(0.5, 0.5, '测试中文', fontsize=12)
    plt.close(test_fig)
    print("中文字体配置成功")
except Exception as e:
    print(f"中文字体配置失败，使用默认字体: {e}")
    # 如果中文字体不可用，使用英文标签
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.esrgan import LiteRealESRGAN

class SmartModelLoader:
    """智能模型加载器 - 自动检测模型架构参数"""
    
    def __init__(self):
        # 定义常见的参数组合
        self.param_combinations = [
            # 您提到的参数组合
            {'num_blocks': 8, 'num_features': 70},
            
            # 常见的参数组合
            {'num_blocks': 6, 'num_features': 64},
            {'num_blocks': 8, 'num_features': 64},
            {'num_blocks': 6, 'num_features': 32},
            {'num_blocks': 4, 'num_features': 64},
            {'num_blocks': 12, 'num_features': 64},
            {'num_blocks': 16, 'num_features': 64},
            
            # 轻量化版本
            {'num_blocks': 4, 'num_features': 32},
            {'num_blocks': 6, 'num_features': 48},
            {'num_blocks': 8, 'num_features': 48},
            
            # 高性能版本
            {'num_blocks': 8, 'num_features': 96},
            {'num_blocks': 12, 'num_features': 96},
            {'num_blocks': 16, 'num_features': 96},
            {'num_blocks': 20, 'num_features': 64},
            {'num_blocks': 23, 'num_features': 64},
            
            # 特殊尺寸
            {'num_blocks': 8, 'num_features': 70},  
            {'num_blocks': 10, 'num_features': 128},
            {'num_blocks': 12, 'num_features': 72},
            {'num_blocks': 6, 'num_features': 56},
            {'num_blocks': 8, 'num_features': 56},
            {'num_blocks': 16, 'num_features': 48},
            {'num_blocks': 24, 'num_features': 32},
            
            # 更多可能的组合
            {'num_blocks': 5, 'num_features': 40},
            {'num_blocks': 7, 'num_features': 60},
            {'num_blocks': 9, 'num_features': 72},
            {'num_blocks': 11, 'num_features': 88},
            {'num_blocks': 15, 'num_features': 80},
            {'num_blocks': 18, 'num_features': 64},
            {'num_blocks': 20, 'num_features': 96},
            {'num_blocks': 32, 'num_features': 32},
        ]
    
    def analyze_checkpoint(self, checkpoint_path):
        """分析检查点文件，尝试推断模型参数"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 获取状态字典
            if 'generator_state_dict' in checkpoint:
                state_dict = checkpoint['generator_state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # 分析权重形状来推断参数
            analysis_info = {
                'total_params': len(state_dict),
                'layer_info': {},
                'possible_params': []
            }
            
            # 分析关键层的形状
            for key, tensor in state_dict.items():
                if 'conv_first' in key and 'weight' in key:
                    analysis_info['layer_info']['conv_first_out'] = tensor.shape[0]
                elif 'rrdb_blocks.0.dense1.conv1.weight' in key:
                    analysis_info['layer_info']['rrdb_features'] = tensor.shape[1]
                elif 'rrdb_blocks' in key and 'dense1.conv1.weight' in key:
                    # 计算RRDB块数
                    block_num = int(key.split('.')[1])
                    if 'max_blocks' not in analysis_info['layer_info']:
                        analysis_info['layer_info']['max_blocks'] = block_num + 1
                    else:
                        analysis_info['layer_info']['max_blocks'] = max(
                            analysis_info['layer_info']['max_blocks'], block_num + 1
                        )
            
            return analysis_info
            
        except Exception as e:
            print(f"分析检查点失败: {e}")
            return None
    
    def load_model_smart(self, checkpoint_path, device='cuda'):
        """智能加载模型，自动尝试不同参数组合"""
        print(f"开始智能加载模型: {checkpoint_path}")
        
        # 首先分析检查点
        analysis = self.analyze_checkpoint(checkpoint_path)
        if analysis:
            print(f"检查点分析结果: {analysis}")
        
        # 加载检查点
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'generator_state_dict' in checkpoint:
                target_state_dict = checkpoint['generator_state_dict']
                epoch = checkpoint.get('epoch', 'unknown')
                best_psnr = checkpoint.get('best_psnr', 'unknown')
            elif 'model_state_dict' in checkpoint:
                target_state_dict = checkpoint['model_state_dict']
                epoch = checkpoint.get('epoch', 'unknown')
                best_psnr = checkpoint.get('best_psnr', 'unknown')
            else:
                target_state_dict = checkpoint
                epoch = 'unknown'
                best_psnr = 'unknown'
        except Exception as e:
            raise Exception(f"无法加载检查点文件: {e}")
        
        # 尝试不同的参数组合
        successful_params = None
        last_error = None
        
        for i, params in enumerate(self.param_combinations):
            try:
                print(f"尝试参数组合 {i+1}/{len(self.param_combinations)}: "
                      f"blocks={params['num_blocks']}, features={params['num_features']}")
                
                # 创建模型
                model = LiteRealESRGAN(
                    num_blocks=params['num_blocks'],
                    num_features=params['num_features']
                )
                
                # 尝试加载权重
                model.load_state_dict(target_state_dict, strict=True)
                
                # 移动到设备
                model = model.to(device)
                model.eval()
                
                print(f"✓ 成功加载模型! 参数: blocks={params['num_blocks']}, features={params['num_features']}")
                successful_params = params
                
                return model, successful_params, epoch, best_psnr
                
            except Exception as e:
                last_error = str(e)
                print(f"✗ 参数组合失败: {e}")
                continue
        
        # 如果所有组合都失败，尝试宽松加载
        print("所有严格匹配都失败，尝试宽松加载...")
        for i, params in enumerate(self.param_combinations[:10]):  # 只尝试前10个常见组合
            try:
                print(f"宽松加载尝试 {i+1}: blocks={params['num_blocks']}, features={params['num_features']}")
                
                model = LiteRealESRGAN(
                    num_blocks=params['num_blocks'],
                    num_features=params['num_features']
                )
                
                # 宽松加载（忽略不匹配的层）
                model.load_state_dict(target_state_dict, strict=False)
                
                model = model.to(device)
                model.eval()
                
                print(f"⚠ 宽松加载成功! 参数: blocks={params['num_blocks']}, features={params['num_features']}")
                print("注意: 某些层可能未正确加载，请检查模型性能")
                successful_params = params
                
                return model, successful_params, epoch, best_psnr
                
            except Exception as e:
                print(f"✗ 宽松加载也失败: {e}")
                continue
        
        # 所有尝试都失败
        raise Exception(f"无法找到匹配的模型架构。最后错误: {last_error}")

class VideoSuperResolutionGUI:
    """视频超分辨率处理图形界面 - 兼容版"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("视频超分辨率处理工具 - 兼容版")
        self.root.geometry("1400x900")
        
        # 变量
        self.video_path_var = tk.StringVar()
        self.model_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.final_video_path_var = tk.StringVar()
        self.space_save_mode_var = tk.BooleanVar(value=False)
        self.batch_size_var = tk.IntVar(value=500)
        self.gpu_enabled_var = tk.BooleanVar(value=True)
        
        # 高级设置变量
        self.chunk_threshold_var = tk.IntVar(value=1024)
        self.chunk_overlap_var = tk.IntVar(value=32)
        self.codec_var = tk.StringVar(value="mp4v")
        self.quality_var = tk.IntVar(value=90)
        
        # 处理状态
        self.is_processing = False
        self.should_stop = False
        self.model = None
        self.device = None
        self.video_info = {}
        self.model_params = None
        self.smart_loader = SmartModelLoader()
        self.processing_stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'start_time': None,
            'processing_times': [],
            'memory_usage': []
        }
        
        # 可视化数据
        self.progress_data = []
        self.memory_data = []
        self.time_data = []
        
        self.setup_ui()
        self.check_gpu_availability()
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建标签页
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # 主控制标签页
        self.setup_main_tab(notebook)
        
        # 监控标签页
        self.setup_monitor_tab(notebook)
        
        # 日志标签页
        self.setup_log_tab(notebook)
        
        # 设置标签页
        self.setup_settings_tab(notebook)
        
        # 状态栏
        self.setup_status_bar(main_frame)
        
    def setup_main_tab(self, notebook):
        """设置主控制标签页"""
        main_tab = ttk.Frame(notebook)
        notebook.add(main_tab, text="主控制")
        
        # 文件选择区域
        file_frame = ttk.LabelFrame(main_tab, text="文件选择")
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 视频文件选择
        video_frame = ttk.Frame(file_frame)
        video_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(video_frame, text="输入视频:").pack(side=tk.LEFT)
        video_entry = ttk.Entry(video_frame, textvariable=self.video_path_var)
        video_entry.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        ttk.Button(video_frame, text="浏览", command=self.browse_video).pack(side=tk.LEFT)
        
        # 模型文件选择
        model_frame = ttk.Frame(file_frame)
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(model_frame, text="模型文件:").pack(side=tk.LEFT)
        model_entry = ttk.Entry(model_frame, textvariable=self.model_path_var)
        model_entry.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        ttk.Button(model_frame, text="浏览", command=self.browse_model).pack(side=tk.LEFT)
        ttk.Button(model_frame, text="智能加载", command=self.load_model_smart).pack(side=tk.LEFT, padx=(5, 0))
        
        # 输出路径选择
        output_frame = ttk.Frame(file_frame)
        output_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(output_frame, text="输出目录:").pack(side=tk.LEFT)
        output_entry = ttk.Entry(output_frame, textvariable=self.output_dir_var)
        output_entry.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        ttk.Button(output_frame, text="浏览", command=self.browse_output_dir).pack(side=tk.LEFT)
        
        # 最终视频路径
        final_frame = ttk.Frame(file_frame)
        final_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(final_frame, text="最终视频:").pack(side=tk.LEFT)
        final_entry = ttk.Entry(final_frame, textvariable=self.final_video_path_var)
        final_entry.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        ttk.Button(final_frame, text="浏览", command=self.browse_final_video).pack(side=tk.LEFT)
        
        # 模型信息区域
        model_info_frame = ttk.LabelFrame(main_tab, text="模型信息")
        model_info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.model_info_text = tk.Text(model_info_frame, height=6, state=tk.DISABLED)
        self.model_info_text.pack(fill=tk.X, padx=10, pady=5)
        
        # 视频信息区域
        video_info_frame = ttk.LabelFrame(main_tab, text="视频信息")
        video_info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.video_info_text = tk.Text(video_info_frame, height=4, state=tk.DISABLED)
        self.video_info_text.pack(fill=tk.X, padx=10, pady=5)
        
        # 处理选项区域
        options_frame = ttk.LabelFrame(main_tab, text="处理选项")
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 空间节省模式
        space_frame = ttk.Frame(options_frame)
        space_frame.pack(fill=tk.X, padx=10, pady=5)
        
        space_check = ttk.Checkbutton(space_frame, text="启用空间节省模式", 
                                     variable=self.space_save_mode_var,
                                     command=self.on_space_mode_changed)
        space_check.pack(side=tk.LEFT)
        
        ttk.Label(space_frame, text="批次大小:").pack(side=tk.LEFT, padx=(20, 5))
        batch_spin = ttk.Spinbox(space_frame, from_=100, to=2000, increment=100,
                                textvariable=self.batch_size_var, width=10)
        batch_spin.pack(side=tk.LEFT)
        
        self.space_warning_label = ttk.Label(space_frame, text="", foreground="orange")
        self.space_warning_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # GPU选项
        gpu_frame = ttk.Frame(options_frame)
        gpu_frame.pack(fill=tk.X, padx=10, pady=5)
        
        gpu_check = ttk.Checkbutton(gpu_frame, text="启用GPU加速", 
                                   variable=self.gpu_enabled_var)
        gpu_check.pack(side=tk.LEFT)
        
        self.gpu_info_label = ttk.Label(gpu_frame, text="")
        self.gpu_info_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # 控制按钮区域
        control_frame = ttk.LabelFrame(main_tab, text="处理控制")
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="开始处理", 
                                      command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="停止处理", 
                                     command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.check_space_button = ttk.Button(button_frame, text="检查磁盘空间", 
                                           command=self.check_disk_space)
        self.check_space_button.pack(side=tk.LEFT)
        
        # 进度区域
        progress_frame = ttk.Frame(control_frame)
        progress_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                          maximum=100, length=400)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.progress_label = ttk.Label(progress_frame, text="准备就绪")
        self.progress_label.pack(side=tk.LEFT, padx=(10, 0))
    
    def load_model_smart(self):
        """智能加载模型"""
        model_path = self.model_path_var.get().strip()
        if not model_path:
            messagebox.showwarning("警告", "请先选择模型文件")
            return
            
        if not os.path.exists(model_path):
            messagebox.showerror("错误", "模型文件不存在")
            return
        
        # 创建加载进度窗口
        progress_window = tk.Toplevel(self.root)
        progress_window.title("智能加载模型")
        progress_window.geometry("500x300")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        # 进度显示
        progress_text = scrolledtext.ScrolledText(progress_window, height=15, width=60)
        progress_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        def update_progress(message):
            progress_text.insert(tk.END, message + "\n")
            progress_text.see(tk.END)
            progress_window.update()
        
        def load_in_thread():
            try:
                update_progress("开始智能模型加载...")
                
                # 检查CUDA可用性
                if not torch.cuda.is_available():
                    update_progress("警告: 未检测到CUDA设备")
                    self.device = torch.device('cpu')
                else:
                    self.device = torch.device('cuda')
                    update_progress(f"使用设备: {self.device}")
                
                # 使用智能加载器
                model, params, epoch, best_psnr = self.smart_loader.load_model_smart(
                    model_path, self.device
                )
                
                self.model = model
                self.model_params = params
                
                # 更新模型信息显示
                info_text = f"模型: LiteRealESRGAN (智能加载)\n"
                info_text += f"架构参数: blocks={params['num_blocks']}, features={params['num_features']}\n"
                info_text += f"设备: {self.device}\n"
                info_text += f"训练轮次: {epoch}\n"
                info_text += f"最佳PSNR: {best_psnr}\n"
                info_text += f"模型文件: {os.path.basename(model_path)}"
                
                self.root.after(0, lambda: self.update_model_info(info_text))
                
                update_progress("✓ 模型加载成功!")
                update_progress(f"✓ 检测到的架构: blocks={params['num_blocks']}, features={params['num_features']}")
                
                # 延迟关闭窗口
                progress_window.after(2000, progress_window.destroy)
                self.root.after(0, lambda: messagebox.showinfo("成功", 
                    f"模型智能加载成功!\n架构: blocks={params['num_blocks']}, features={params['num_features']}"))
                
            except Exception as e:
                error_msg = f"模型加载失败: {str(e)}"
                update_progress(f"✗ {error_msg}")
                self.root.after(0, lambda: messagebox.showerror("错误", error_msg))
                progress_window.after(3000, progress_window.destroy)
        
        # 在后台线程中加载
        load_thread = threading.Thread(target=load_in_thread)
        load_thread.daemon = True
        load_thread.start()
    
    def start_processing(self):
        """开始处理"""
        # 验证输入
        if not self.video_path_var.get():
            messagebox.showwarning("警告", "请选择输入视频")
            return
            
        if not self.model:
            messagebox.showwarning("警告", "请先加载模型")
            return
            
        if not self.output_dir_var.get():
            messagebox.showwarning("警告", "请选择输出目录")
            return
            
        if not self.final_video_path_var.get():
            messagebox.showwarning("警告", "请选择最终视频保存路径")
            return
            
        # 创建输出目录
        os.makedirs(self.output_dir_var.get(), exist_ok=True)
        
        # 重置状态
        self.is_processing = True
        self.should_stop = False
        self.processing_stats = {
            'total_frames': self.video_info.get('frame_count', 0),
            'processed_frames': 0,
            'start_time': time.time(),
            'processing_times': [],
            'memory_usage': []
        }
        
        # 重置进度条
        self.progress_var.set(0)
        self.progress_data = []
        self.memory_data = []
        self.time_data = []
        
        # 更新UI状态
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # 在后台线程中开始处理
        processing_thread = threading.Thread(target=self._process_video)
        processing_thread.daemon = True
        processing_thread.start()
        
        # 启动监控更新
        self.update_monitor()
        
    def stop_processing(self):
        """停止处理"""
        self.should_stop = True
        self.update_status("正在停止处理...")
        
    def _process_video(self):
        """视频处理主函数"""
        try:
            self.log_message("开始视频处理")
            if self.space_save_mode_var.get():
                self.log_message("使用空间节省模式")
                self._process_video_batch_mode()
            else:
                self.log_message("使用普通模式")
                self._process_video_normal_mode()
                
        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            self.log_message(error_msg, "ERROR")
            # 修复lambda函数的变量作用域问题
            def show_error():
                messagebox.showerror("错误", error_msg)
            self.root.after(0, show_error)
        finally:
            self.root.after(0, self._processing_finished)
            
    def _process_video_normal_mode(self):
        """普通模式处理视频"""
        video_path = self.video_path_var.get()
        output_dir = self.output_dir_var.get()
        final_video_path = self.final_video_path_var.get()
        
        self.log_message(f"输入视频: {video_path}")
        self.log_message(f"输出目录: {output_dir}")
        self.log_message(f"最终视频: {final_video_path}")
        
        # 打开视频
        cap = self._open_video_with_fallback(video_path)
        if cap is None:
            raise Exception("无法打开视频文件")
            
        try:
            # 创建临时目录
            temp_frames_dir = os.path.join(output_dir, "temp_frames")
            temp_sr_dir = os.path.join(output_dir, "temp_sr")
            os.makedirs(temp_frames_dir, exist_ok=True)
            os.makedirs(temp_sr_dir, exist_ok=True)
            
            self.log_message(f"创建临时目录: {temp_frames_dir}")
            self.log_message(f"创建超分目录: {temp_sr_dir}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            self.processing_stats['total_frames'] = total_frames
            self.log_message(f"总帧数: {total_frames}, FPS: {fps}")
            
            def update_status_msg():
                self.update_status(f"开始提取 {total_frames} 帧...")
            self.root.after(0, update_status_msg)
            
            # 提取所有帧
            frame_idx = 0
            while True:
                if self.should_stop:
                    self.log_message("用户停止处理", "WARNING")
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 保存原始帧
                frame_path = os.path.join(temp_frames_dir, f"frame_{frame_idx:06d}.png")
                cv2.imwrite(frame_path, frame)
                
                frame_idx += 1
                
                # 更新进度 - 修复变量作用域问题
                progress = (frame_idx / total_frames) * 30  # 提取帧占30%进度
                def update_progress(p=progress, idx=frame_idx, total=total_frames):
                    self.progress_var.set(p)
                    self.update_status(f"提取帧: {idx}/{total}")
                self.root.after(0, update_progress)
                
                # 每100帧记录一次日志
                if frame_idx % 100 == 0:
                    self.log_message(f"已提取 {frame_idx}/{total_frames} 帧")
                
            cap.release()
            self.log_message(f"帧提取完成，共提取 {frame_idx} 帧")
            
            if self.should_stop:
                return
                
            # 处理帧
            def update_status_sr():
                self.update_status("开始超分辨率处理...")
            self.root.after(0, update_status_sr)
            self.log_message("开始超分辨率处理")
            
            frame_files = sorted([f for f in os.listdir(temp_frames_dir) if f.endswith('.png')])
            self.log_message(f"找到 {len(frame_files)} 个帧文件")
            
            for i, frame_file in enumerate(frame_files):
                if self.should_stop:
                    self.log_message("用户停止处理", "WARNING")
                    break
                    
                frame_start_time = time.time()
                
                # 加载和处理帧
                frame_path = os.path.join(temp_frames_dir, frame_file)
                sr_frame_path = os.path.join(temp_sr_dir, frame_file)
                
                self._process_single_frame(frame_path, sr_frame_path)
                
                # 删除原始帧以节省空间
                os.remove(frame_path)
                
                # 更新统计
                frame_time = time.time() - frame_start_time
                self.processing_stats['processing_times'].append(frame_time)
                self.processing_stats['processed_frames'] = i + 1
                
                # 更新进度 - 修复变量作用域问题
                progress = 30 + (i + 1) / len(frame_files) * 60  # 处理占60%进度
                def update_progress_sr(p=progress, idx=i+1, total=len(frame_files)):
                    self.progress_var.set(p)
                    self.update_status(f"处理帧: {idx}/{total}")
                self.root.after(0, update_progress_sr)
                
                # 每50帧记录一次日志
                if (i + 1) % 50 == 0:
                    avg_time = sum(self.processing_stats['processing_times'][-50:]) / min(50, len(self.processing_stats['processing_times']))
                    self.log_message(f"已处理 {i+1}/{len(frame_files)} 帧，平均耗时: {avg_time:.3f}秒/帧")
                
            if self.should_stop:
                return
                
            # 合成最终视频
            def update_status_merge():
                self.update_status("合成最终视频...")
            self.root.after(0, update_status_merge)
            self.log_message("开始合成最终视频")
            
            self._create_final_video(temp_sr_dir, final_video_path, fps)
            
            # 清理临时文件
            self.log_message("清理临时文件")
            shutil.rmtree(temp_frames_dir, ignore_errors=True)
            shutil.rmtree(temp_sr_dir, ignore_errors=True)
            
            def update_final_progress():
                self.progress_var.set(100)
                self.update_status("处理完成！")
            self.root.after(0, update_final_progress)
            self.log_message("视频处理完成！")
            
        finally:
            cap.release()
    
    def _process_single_frame(self, input_path, output_path):
        """处理单帧图像"""
        # 加载图像
        image = Image.open(input_path).convert('RGB')
        
        # 检查是否需要分块处理
        width, height = image.size
        max_size = self.chunk_threshold_var.get()
        
        if width > max_size or height > max_size:
            # 分块处理
            sr_image = self._process_large_image(image)
        else:
            # 直接处理
            sr_image = self._process_image_direct(image)
            
        # 保存结果
        sr_image.save(output_path, 'PNG')
        
    def _process_image_direct(self, image):
        """直接处理图像"""
        # 预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            sr_tensor = self.model(image_tensor)
            
        # 后处理
        sr_tensor = (sr_tensor + 1) / 2  # 反归一化到[0, 1]
        sr_tensor = torch.clamp(sr_tensor, 0, 1)
        sr_tensor = sr_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # BCHW -> HWC
        sr_image = Image.fromarray((sr_tensor * 255).astype(np.uint8))
        
        return sr_image
    
    def _process_large_image(self, image):
        """分块处理大图像"""
        width, height = image.size
        chunk_size = self.chunk_threshold_var.get()
        overlap = self.chunk_overlap_var.get()
        
        # 计算输出尺寸
        output_width = width * 2
        output_height = height * 2
        
        # 创建输出图像
        sr_image = Image.new('RGB', (output_width, output_height))
        
        # 分块处理
        for y in range(0, height, chunk_size - overlap):
            for x in range(0, width, chunk_size - overlap):
                # 计算块边界
                x1 = x
                y1 = y
                x2 = min(x + chunk_size, width)
                y2 = min(y + chunk_size, height)
                
                # 提取块
                chunk = image.crop((x1, y1, x2, y2))
                
                # 处理块
                sr_chunk = self._process_image_direct(chunk)
                
                # 计算输出位置
                out_x1 = x1 * 2
                out_y1 = y1 * 2
                out_x2 = x2 * 2
                out_y2 = y2 * 2
                
                # 粘贴到输出图像
                sr_image.paste(sr_chunk, (out_x1, out_y1, out_x2, out_y2))
        
        return sr_image
    
    def _create_final_video(self, frames_dir, output_path, fps):
        """创建最终视频"""
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        
        if not frame_files:
            raise Exception("没有找到处理后的帧文件")
        
        # 读取第一帧获取尺寸
        first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
        height, width = first_frame.shape[:2]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*self.codec_var.get())
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            for frame_file in frame_files:
                if self.should_stop:
                    break
                    
                frame_path = os.path.join(frames_dir, frame_file)
                frame = cv2.imread(frame_path)
                out.write(frame)
                
        finally:
            out.release()
    
    def _open_video_with_fallback(self, video_path):
        """使用多种后端尝试打开视频"""
        backends = [cv2.CAP_FFMPEG, cv2.CAP_DSHOW, cv2.CAP_ANY]
        
        for backend in backends:
            try:
                cap = cv2.VideoCapture(video_path, backend)
                if cap.isOpened():
                    return cap
                cap.release()
            except Exception as e:
                continue
                
        return None
    
    def _processing_finished(self):
        """处理完成后的清理工作"""
        self.is_processing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        if not self.should_stop:
            messagebox.showinfo("完成", "视频处理完成！")
    
    def update_status(self, message):
        """更新状态显示"""
        self.progress_label.config(text=message)
        
    def update_monitor(self):
        """更新监控显示"""
        if self.is_processing:
            try:
                # 计算当前统计信息
                current_time = time.time()
                elapsed_time = current_time - self.processing_stats['start_time']
                processed_frames = self.processing_stats['processed_frames']
                total_frames = self.processing_stats['total_frames']
                
                # 计算处理速度
                if elapsed_time > 0:
                    speed = processed_frames / elapsed_time
                else:
                    speed = 0
                
                # 计算剩余时间
                if speed > 0 and total_frames > processed_frames:
                    remaining_frames = total_frames - processed_frames
                    remaining_seconds = remaining_frames / speed
                    remaining_time = self._format_time(remaining_seconds)
                else:
                    remaining_time = "--:--:--"
                
                # 获取内存使用情况
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                # 获取GPU使用率（如果可用）
                gpu_usage = 0
                if torch.cuda.is_available():
                    try:
                        gpu_usage = torch.cuda.utilization()
                    except:
                        gpu_usage = 0
                
                # 更新标签
                self.processed_frames_label.config(text=f"{processed_frames}/{total_frames}")
                self.processing_speed_label.config(text=f"{speed:.2f} 帧/秒")
                self.remaining_time_label.config(text=remaining_time)
                self.memory_usage_label.config(text=f"{memory_mb:.1f} MB")
                
                # 更新图表数据
                self.monitor_data['times'].append(elapsed_time)
                self.monitor_data['speeds'].append(speed)
                self.monitor_data['memory'].append(memory_mb)
                self.monitor_data['progress'].append((processed_frames / total_frames) * 100 if total_frames > 0 else 0)
                self.monitor_data['gpu_usage'].append(gpu_usage)
                
                # 限制数据点数量（保留最近100个点）
                max_points = 100
                for key in self.monitor_data:
                    if len(self.monitor_data[key]) > max_points:
                        self.monitor_data[key] = self.monitor_data[key][-max_points:]
                
                # 更新图表
                self._update_monitor_charts()
                
            except Exception as e:
                print(f"监控更新错误: {e}")
            
            # 继续更新
            self.root.after(1000, self.update_monitor)
    
    def _update_monitor_charts(self):
        """更新监控图表"""
        try:
            # 清除旧数据
            self.speed_ax.clear()
            self.memory_ax.clear()
            self.progress_ax.clear()
            self.gpu_ax.clear()
            
            # 绘制新数据
            if len(self.monitor_data['times']) > 1:
                self.speed_ax.plot(self.monitor_data['times'], self.monitor_data['speeds'], 'b-')
                self.memory_ax.plot(self.monitor_data['times'], self.monitor_data['memory'], 'g-')
                self.progress_ax.plot(self.monitor_data['times'], self.monitor_data['progress'], 'r-')
                self.gpu_ax.plot(self.monitor_data['times'], self.monitor_data['gpu_usage'], 'm-')
            
            # 重新设置标题和标签
            self.speed_ax.set_title('处理速度 (帧/秒)')
            self.speed_ax.set_xlabel('时间 (秒)')
            self.speed_ax.set_ylabel('帧/秒')
            self.speed_ax.grid(True, alpha=0.3)
            
            self.memory_ax.set_title('内存使用 (MB)')
            self.memory_ax.set_xlabel('时间 (秒)')
            self.memory_ax.set_ylabel('MB')
            self.memory_ax.grid(True, alpha=0.3)
            
            self.progress_ax.set_title('处理进度 (%)')
            self.progress_ax.set_xlabel('时间 (秒)')
            self.progress_ax.set_ylabel('%')
            self.progress_ax.set_ylim(0, 100)
            self.progress_ax.grid(True, alpha=0.3)
            
            self.gpu_ax.set_title('GPU使用率 (%)')
            self.gpu_ax.set_xlabel('时间 (秒)')
            self.gpu_ax.set_ylabel('%')
            self.gpu_ax.set_ylim(0, 100)
            self.gpu_ax.grid(True, alpha=0.3)
            
            # 调整布局并刷新
            self.monitor_fig.tight_layout()
            self.monitor_canvas.draw()
            
        except Exception as e:
            print(f"图表更新错误: {e}")
    
    def _format_time(self, seconds):
        """格式化时间显示"""
        if seconds < 0:
            return "--:--:--"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def log_message(self, message, level="INFO"):
        """添加日志消息"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def browse_video(self):
        """浏览视频文件"""
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v *.dat"),
                ("MP4文件", "*.mp4"),
                ("AVI文件", "*.avi"),
                ("所有文件", "*.*")
            ]
        )
        if file_path:
            self.video_path_var.set(file_path)
            self.analyze_video()
    
    def browse_model(self):
        """浏览模型文件"""
        file_path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[
                ("PyTorch模型", "*.pth *.pt"),
                ("所有文件", "*.*")
            ]
        )
        if file_path:
            self.model_path_var.set(file_path)
    
    def browse_output_dir(self):
        """浏览输出目录"""
        dir_path = filedialog.askdirectory(title="选择输出目录")
        if dir_path:
            self.output_dir_var.set(dir_path)
    
    def browse_final_video(self):
        """浏览最终视频保存路径"""
        file_path = filedialog.asksaveasfilename(
            title="选择最终视频保存路径",
            defaultextension=".mp4",
            filetypes=[
                ("MP4文件", "*.mp4"),
                ("AVI文件", "*.avi"),
                ("所有文件", "*.*")
            ]
        )
        if file_path:
            self.final_video_path_var.set(file_path)
    
    def analyze_video(self):
        """分析视频信息"""
        video_path = self.video_path_var.get()
        if not video_path or not os.path.exists(video_path):
            return
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return
            
            # 获取视频信息
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            self.video_info = {
                'frame_count': frame_count,
                'fps': fps,
                'width': width,
                'height': height,
                'duration': duration
            }
            
            # 更新显示
            info_text = f"分辨率: {width}x{height}\n"
            info_text += f"帧数: {frame_count}\n"
            info_text += f"帧率: {fps:.2f} FPS\n"
            info_text += f"时长: {duration:.2f} 秒"
            
            self.update_video_info(info_text)
            cap.release()
            
        except Exception as e:
            print(f"分析视频失败: {e}")
    
    def update_model_info(self, text):
        """更新模型信息显示"""
        self.model_info_text.config(state=tk.NORMAL)
        self.model_info_text.delete(1.0, tk.END)
        self.model_info_text.insert(1.0, text)
        self.model_info_text.config(state=tk.DISABLED)
    
    def update_video_info(self, text):
        """更新视频信息显示"""
        self.video_info_text.config(state=tk.NORMAL)
        self.video_info_text.delete(1.0, tk.END)
        self.video_info_text.insert(1.0, text)
        self.video_info_text.config(state=tk.DISABLED)
    
    def check_gpu_availability(self):
        """检查GPU可用性"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.gpu_info_label.config(text=f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            self.gpu_info_label.config(text="GPU: 不可用", foreground="red")
            self.gpu_enabled_var.set(False)
    
    def on_space_mode_changed(self):
        """空间节省模式变化回调"""
        if self.space_save_mode_var.get():
            self.space_warning_label.config(text="注意: 启用此模式会降低处理速度")
        else:
            self.space_warning_label.config(text="")
    
    def setup_monitor_tab(self, notebook):
        """设置监控标签页"""
        monitor_tab = ttk.Frame(notebook)
        notebook.add(monitor_tab, text="处理监控")
        
        # 创建监控界面
        # 实时统计信息
        stats_frame = ttk.LabelFrame(monitor_tab, text="实时统计")
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X, padx=10, pady=10)
        
        # 统计标签
        ttk.Label(stats_grid, text="已处理帧数:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.processed_frames_label = ttk.Label(stats_grid, text="0/0")
        self.processed_frames_label.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(stats_grid, text="处理速度:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        self.processing_speed_label = ttk.Label(stats_grid, text="0.00 帧/秒")
        self.processing_speed_label.grid(row=0, column=3, sticky=tk.W)
        
        ttk.Label(stats_grid, text="剩余时间:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        self.remaining_time_label = ttk.Label(stats_grid, text="--:--:--")
        self.remaining_time_label.grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(stats_grid, text="内存使用:").grid(row=1, column=2, sticky=tk.W, padx=(20, 10))
        self.memory_usage_label = ttk.Label(stats_grid, text="0 MB")
        self.memory_usage_label.grid(row=1, column=3, sticky=tk.W)
        
        # 性能图表
        chart_frame = ttk.LabelFrame(monitor_tab, text="性能监控")
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 创建matplotlib图表
        self.monitor_fig = Figure(figsize=(12, 6), dpi=80)
        self.monitor_fig.patch.set_facecolor('white')
        
        # 创建子图
        self.speed_ax = self.monitor_fig.add_subplot(221)
        self.memory_ax = self.monitor_fig.add_subplot(222)
        self.progress_ax = self.monitor_fig.add_subplot(223)
        self.gpu_ax = self.monitor_fig.add_subplot(224)
        
        # 设置图表标题和标签
        self.speed_ax.set_title('处理速度 (帧/秒)')
        self.speed_ax.set_xlabel('时间')
        self.speed_ax.set_ylabel('帧/秒')
        
        self.memory_ax.set_title('内存使用 (MB)')
        self.memory_ax.set_xlabel('时间')
        self.memory_ax.set_ylabel('MB')
        
        self.progress_ax.set_title('处理进度 (%)')
        self.progress_ax.set_xlabel('时间')
        self.progress_ax.set_ylabel('%')
        
        self.gpu_ax.set_title('GPU使用率 (%)')
        self.gpu_ax.set_xlabel('时间')
        self.gpu_ax.set_ylabel('%')
        
        # 调整布局
        self.monitor_fig.tight_layout()
        
        # 创建画布
        self.monitor_canvas = FigureCanvasTkAgg(self.monitor_fig, chart_frame)
        self.monitor_canvas.draw()
        self.monitor_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 初始化监控数据
        self.monitor_data = {
            'times': [],
            'speeds': [],
            'memory': [],
            'progress': [],
            'gpu_usage': []
        }
    
    def setup_log_tab(self, notebook):
        """设置日志标签页"""
        log_tab = ttk.Frame(notebook)
        notebook.add(log_tab, text="详细日志")
        
        # 日志显示区域
        self.log_text = scrolledtext.ScrolledText(log_tab, height=25, state=tk.DISABLED,
                                                 font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def setup_settings_tab(self, notebook):
        """设置设置标签页"""
        settings_tab = ttk.Frame(notebook)
        notebook.add(settings_tab, text="高级设置")
        
        # 处理设置
        processing_frame = ttk.LabelFrame(settings_tab, text="处理设置")
        processing_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 分块处理设置
        chunk_frame = ttk.Frame(processing_frame)
        chunk_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(chunk_frame, text="大图分块阈值:").pack(side=tk.LEFT)
        chunk_spin = ttk.Spinbox(chunk_frame, from_=512, to=2048, increment=128,
                                textvariable=self.chunk_threshold_var, width=10)
        chunk_spin.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Label(chunk_frame, text="分块重叠:").pack(side=tk.LEFT)
        overlap_spin = ttk.Spinbox(chunk_frame, from_=16, to=64, increment=8,
                                  textvariable=self.chunk_overlap_var, width=10)
        overlap_spin.pack(side=tk.LEFT, padx=(5, 0))
        
        # 视频编码设置
        encoding_frame = ttk.LabelFrame(settings_tab, text="视频编码设置")
        encoding_frame.pack(fill=tk.X, padx=10, pady=5)
        
        codec_frame = ttk.Frame(encoding_frame)
        codec_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(codec_frame, text="输出编码:").pack(side=tk.LEFT)
        codec_combo = ttk.Combobox(codec_frame, textvariable=self.codec_var,
                                  values=["mp4v", "XVID", "H264", "MJPG"], state="readonly")
        codec_combo.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Label(codec_frame, text="质量:").pack(side=tk.LEFT)
        quality_spin = ttk.Spinbox(codec_frame, from_=50, to=100, increment=5,
                                  textvariable=self.quality_var, width=10)
        quality_spin.pack(side=tk.LEFT, padx=(5, 0))
    
    def setup_status_bar(self, parent):
        """设置状态栏"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame, text="准备就绪")
        self.status_label.pack(side=tk.LEFT)
    
    def check_disk_space(self):
        """检查磁盘空间"""
        output_dir = self.output_dir_var.get()
        if not output_dir:
            messagebox.showwarning("警告", "请先选择输出目录")
            return
            
        try:
            # 获取磁盘空间信息
            disk_usage = shutil.disk_usage(output_dir)
            free_space_gb = disk_usage.free / (1024**3)
            
            # 估算所需空间
            if self.video_info:
                frame_count = self.video_info['frame_count']
                width = self.video_info['width']
                height = self.video_info['height']
                
                # 估算每帧大小（假设PNG格式，压缩比约3:1）
                estimated_frame_size = (width * 2) * (height * 2) * 3 / 3  # 2倍分辨率，RGB，压缩
                estimated_total_size_gb = (estimated_frame_size * frame_count) / (1024**3)
                
                if self.space_save_mode_var.get():
                    batch_size = self.batch_size_var.get()
                    estimated_temp_size_gb = (estimated_frame_size * batch_size) / (1024**3)
                    
                    message = f"可用空间: {free_space_gb:.2f} GB\n"
                    message += f"预计临时空间需求: {estimated_temp_size_gb:.2f} GB\n"
                    message += f"预计最终视频大小: {estimated_total_size_gb * 0.1:.2f} GB"  # 视频压缩比约10:1
                else:
                    message = f"可用空间: {free_space_gb:.2f} GB\n"
                    message += f"预计空间需求: {estimated_total_size_gb:.2f} GB"
                    
                if free_space_gb < estimated_total_size_gb:
                    message += "\n\n警告: 磁盘空间可能不足！"
                    messagebox.showwarning("磁盘空间检查", message)
                else:
                    message += "\n\n磁盘空间充足"
                    messagebox.showinfo("磁盘空间检查", message)
            else:
                messagebox.showinfo("磁盘空间", f"可用空间: {free_space_gb:.2f} GB")
                
        except Exception as e:
            messagebox.showerror("错误", f"检查磁盘空间失败:\n{str(e)}")
    
    def _process_video_batch_mode(self):
        """批次模式处理视频 - 简化版本"""
        self.log_message("批次模式暂未完全实现，使用普通模式")
        self._process_video_normal_mode()

def main():
    """主函数"""
    app = VideoSuperResolutionGUI()
    app.root.mainloop()

if __name__ == "__main__":
    main()
