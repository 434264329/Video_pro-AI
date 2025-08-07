#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频超分辨率处理工具
支持多种视频格式，包括AV1、MPEG、DAT等
具备空间节省模式、GPU加速、可视化界面等功能
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
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.esrgan import LiteRealESRGAN

class VideoSuperResolutionGUI:
    """视频超分辨率处理图形界面"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("视频超分辨率处理工具")
        self.root.geometry("1400x900")
        
        # 变量
        self.video_path_var = tk.StringVar()
        self.model_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.final_video_path_var = tk.StringVar()
        self.space_save_mode_var = tk.BooleanVar(value=False)
        self.batch_size_var = tk.IntVar(value=500)
        self.gpu_enabled_var = tk.BooleanVar(value=True)
        
        # 处理状态
        self.is_processing = False
        self.should_stop = False
        self.model = None
        self.device = None
        self.video_info = {}
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
        ttk.Button(model_frame, text="加载模型", command=self.load_model).pack(side=tk.LEFT, padx=(5, 0))
        
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
        
        self.model_info_text = tk.Text(model_info_frame, height=4, state=tk.DISABLED)
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
        
    def setup_monitor_tab(self, notebook):
        """设置监控标签页"""
        monitor_tab = ttk.Frame(notebook)
        notebook.add(monitor_tab, text="处理监控")
        
        # 创建matplotlib图表
        self.monitor_fig = Figure(figsize=(12, 8), dpi=100)
        self.monitor_canvas = FigureCanvasTkAgg(self.monitor_fig, monitor_tab)
        self.monitor_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 创建子图
        self.progress_ax = self.monitor_fig.add_subplot(2, 2, 1)
        self.memory_ax = self.monitor_fig.add_subplot(2, 2, 2)
        self.time_ax = self.monitor_fig.add_subplot(2, 2, 3)
        self.fps_ax = self.monitor_fig.add_subplot(2, 2, 4)
        
        self.monitor_fig.tight_layout()
        
        # 初始化图表
        self.init_monitor_charts()
        
    def setup_log_tab(self, notebook):
        """设置日志标签页"""
        log_tab = ttk.Frame(notebook)
        notebook.add(log_tab, text="详细日志")
        
        # 日志控制区域
        log_control_frame = ttk.Frame(log_tab)
        log_control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(log_control_frame, text="清空日志", command=self.clear_log).pack(side=tk.LEFT)
        ttk.Button(log_control_frame, text="保存日志", command=self.save_log).pack(side=tk.LEFT, padx=(10, 0))
        
        # 日志显示区域
        log_frame = ttk.LabelFrame(log_tab, text="处理日志")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=25, state=tk.DISABLED,
                                                 font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def log_message(self, message, level="INFO"):
        """添加日志消息"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        
    def clear_log(self):
        """清空日志"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        
    def save_log(self):
        """保存日志到文件"""
        file_path = filedialog.asksaveasfilename(
            title="保存日志文件",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                messagebox.showinfo("成功", "日志已保存")
            except Exception as e:
                messagebox.showerror("错误", f"保存日志失败: {str(e)}")
        
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
        self.chunk_threshold_var = tk.IntVar(value=1024)
        chunk_spin = ttk.Spinbox(chunk_frame, from_=512, to=2048, increment=128,
                                textvariable=self.chunk_threshold_var, width=10)
        chunk_spin.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Label(chunk_frame, text="分块重叠:").pack(side=tk.LEFT)
        self.chunk_overlap_var = tk.IntVar(value=32)
        overlap_spin = ttk.Spinbox(chunk_frame, from_=16, to=64, increment=8,
                                  textvariable=self.chunk_overlap_var, width=10)
        overlap_spin.pack(side=tk.LEFT, padx=(5, 0))
        
        # 视频编码设置
        encoding_frame = ttk.LabelFrame(settings_tab, text="视频编码设置")
        encoding_frame.pack(fill=tk.X, padx=10, pady=5)
        
        codec_frame = ttk.Frame(encoding_frame)
        codec_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(codec_frame, text="输出编码:").pack(side=tk.LEFT)
        self.codec_var = tk.StringVar(value="mp4v")
        codec_combo = ttk.Combobox(codec_frame, textvariable=self.codec_var,
                                  values=["mp4v", "XVID", "H264", "MJPG"], state="readonly")
        codec_combo.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Label(codec_frame, text="质量:").pack(side=tk.LEFT)
        self.quality_var = tk.IntVar(value=90)
        quality_spin = ttk.Spinbox(codec_frame, from_=50, to=100, increment=5,
                                  textvariable=self.quality_var, width=10)
        quality_spin.pack(side=tk.LEFT, padx=(5, 0))
        
    def setup_status_bar(self, parent):
        """设置状态栏"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame, text="准备就绪")
        self.status_label.pack(side=tk.LEFT)
        
        self.time_label = ttk.Label(status_frame, text="")
        self.time_label.pack(side=tk.RIGHT)
        
    def init_monitor_charts(self):
        """初始化监控图表"""
        # 进度图表
        self.progress_ax.set_title("处理进度")
        self.progress_ax.set_xlabel("时间")
        self.progress_ax.set_ylabel("完成百分比 (%)")
        self.progress_ax.grid(True, alpha=0.3)
        
        # 内存使用图表
        self.memory_ax.set_title("内存使用")
        self.memory_ax.set_xlabel("时间")
        self.memory_ax.set_ylabel("内存 (MB)")
        self.memory_ax.grid(True, alpha=0.3)
        
        # 处理时间图表
        self.time_ax.set_title("帧处理时间")
        self.time_ax.set_xlabel("帧数")
        self.time_ax.set_ylabel("时间 (秒)")
        self.time_ax.grid(True, alpha=0.3)
        
        # FPS图表
        self.fps_ax.set_title("处理速度")
        self.fps_ax.set_xlabel("时间")
        self.fps_ax.set_ylabel("FPS")
        self.fps_ax.grid(True, alpha=0.3)
        
        self.monitor_canvas.draw()
        
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
            
    def browse_video(self):
        """浏览视频文件"""
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v *.dat"),
                ("MP4文件", "*.mp4"),
                ("AVI文件", "*.avi"),
                ("MOV文件", "*.mov"),
                ("MKV文件", "*.mkv"),
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
            filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")]
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
            title="保存最终视频",
            defaultextension=".mp4",
            filetypes=[("MP4文件", "*.mp4"), ("AVI文件", "*.avi"), ("所有文件", "*.*")]
        )
        if file_path:
            self.final_video_path_var.set(file_path)
            
    def analyze_video(self):
        """分析视频信息"""
        video_path = self.video_path_var.get()
        if not video_path or not os.path.exists(video_path):
            return
            
        try:
            cap = self._open_video_with_fallback(video_path)
            if cap is None:
                self.update_video_info("无法打开视频文件")
                return
                
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # 获取编码信息
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            self.video_info = {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration,
                'codec': codec
            }
            
            cap.release()
            
            # 更新显示
            info_text = f"分辨率: {width}x{height}\n"
            info_text += f"帧率: {fps:.2f} FPS\n"
            info_text += f"总帧数: {frame_count}\n"
            info_text += f"时长: {duration:.2f} 秒\n"
            info_text += f"编码: {codec}"
            
            self.update_video_info(info_text)
            
        except Exception as e:
            self.update_video_info(f"分析视频失败: {str(e)}")
            
    def _open_video_with_fallback(self, video_path):
        """使用多种后端尝试打开视频"""
        backends = [
            cv2.CAP_FFMPEG,
            cv2.CAP_ANY,
            cv2.CAP_DSHOW,  # Windows DirectShow
            cv2.CAP_MSMF,   # Microsoft Media Foundation
        ]
        
        for backend in backends:
            try:
                cap = cv2.VideoCapture(video_path, backend)
                if cap.isOpened():
                    # 测试读取一帧
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到开始
                        return cap
                cap.release()
            except Exception as e:
                continue
                
        return None
        
    def update_video_info(self, text):
        """更新视频信息显示"""
        self.video_info_text.config(state=tk.NORMAL)
        self.video_info_text.delete(1.0, tk.END)
        self.video_info_text.insert(1.0, text)
        self.video_info_text.config(state=tk.DISABLED)
        
    def load_model(self):
        """加载模型"""
        model_path = self.model_path_var.get().strip()
        if not model_path:
            messagebox.showwarning("警告", "请先选择模型文件")
            return
            
        if not os.path.exists(model_path):
            messagebox.showerror("错误", "模型文件不存在")
            return
            
        try:
            # 检查CUDA可用性
            if not torch.cuda.is_available():
                messagebox.showerror("错误", "未检测到CUDA设备，此工具仅支持GPU处理")
                return
                
            # 设置设备
            self.device = torch.device('cuda')
            
            # 创建模型
            self.model = LiteRealESRGAN(num_blocks=6)
            
            # 加载权重
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'generator_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['generator_state_dict'])
                epoch = checkpoint.get('epoch', 'unknown')
                best_psnr = checkpoint.get('best_psnr', 'unknown')
            else:
                self.model.load_state_dict(checkpoint)
                epoch = 'unknown'
                best_psnr = 'unknown'
                
            # 移动到设备
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # 更新模型信息显示
            info_text = f"模型: LiteRealESRGAN\n"
            info_text += f"设备: {self.device}\n"
            info_text += f"训练轮次: {epoch}\n"
            info_text += f"最佳PSNR: {best_psnr}"
            
            self.update_model_info(info_text)
            
            messagebox.showinfo("成功", "模型加载成功！")
            
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败:\n{str(e)}")
            
    def update_model_info(self, text):
        """更新模型信息显示"""
        self.model_info_text.config(state=tk.NORMAL)
        self.model_info_text.delete(1.0, tk.END)
        self.model_info_text.insert(1.0, text)
        self.model_info_text.config(state=tk.DISABLED)
        
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
            
    def _process_video_batch_mode(self):
        """批次模式处理视频"""
        video_path = self.video_path_var.get()
        output_dir = self.output_dir_var.get()
        final_video_path = self.final_video_path_var.get()
        batch_size = self.batch_size_var.get()
        
        # 打开视频
        cap = self._open_video_with_fallback(video_path)
        if cap is None:
            raise Exception("无法打开视频文件")
            
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 计算批次数
            num_batches = (total_frames + batch_size - 1) // batch_size
            
            temp_videos = []
            
            for batch_idx in range(num_batches):
                if self.should_stop:
                    break
                    
                start_frame = batch_idx * batch_size
                end_frame = min(start_frame + batch_size, total_frames)
                
                def update_batch_status(batch=batch_idx+1, total=num_batches):
                    self.update_status(f"处理批次 {batch}/{total}...")
                self.root.after(0, update_batch_status)
                
                # 处理当前批次
                batch_video_path = self._process_batch(cap, start_frame, end_frame, 
                                                     output_dir, fps, batch_idx)
                if batch_video_path:
                    temp_videos.append(batch_video_path)
                    
                # 更新进度
                progress = (batch_idx + 1) / num_batches * 90  # 批次处理占90%进度
                def update_batch_progress(p=progress):
                    self.progress_var.set(p)
                self.root.after(0, update_batch_progress)
                
            cap.release()
            
            if self.should_stop or not temp_videos:
                return
                
            # 合并所有批次视频
            def update_merge_status():
                self.update_status("合并视频片段...")
            self.root.after(0, update_merge_status)
            self._merge_videos(temp_videos, final_video_path)
            
            # 清理临时视频
            for temp_video in temp_videos:
                try:
                    os.remove(temp_video)
                except:
                    pass
                    
            def update_final_status():
                self.progress_var.set(100)
                self.update_status("处理完成！")
            self.root.after(0, update_final_status)
            
        finally:
            cap.release()
            
    def _process_batch(self, cap, start_frame, end_frame, output_dir, fps, batch_idx):
        """处理单个批次"""
        # 创建临时目录
        temp_frames_dir = os.path.join(output_dir, f"temp_batch_{batch_idx}")
        temp_sr_dir = os.path.join(output_dir, f"temp_sr_{batch_idx}")
        os.makedirs(temp_frames_dir, exist_ok=True)
        os.makedirs(temp_sr_dir, exist_ok=True)
        
        try:
            # 设置视频位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # 提取批次帧
            for frame_idx in range(start_frame, end_frame):
                if self.should_stop:
                    return None
                    
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_path = os.path.join(temp_frames_dir, f"frame_{frame_idx:06d}.png")
                cv2.imwrite(frame_path, frame)
                
            # 处理帧
            frame_files = sorted([f for f in os.listdir(temp_frames_dir) if f.endswith('.png')])
            
            for frame_file in frame_files:
                if self.should_stop:
                    return None
                    
                frame_path = os.path.join(temp_frames_dir, frame_file)
                sr_frame_path = os.path.join(temp_sr_dir, frame_file)
                
                self._process_single_frame(frame_path, sr_frame_path)
                
                # 删除原始帧
                os.remove(frame_path)
                
            # 创建批次视频
            batch_video_path = os.path.join(output_dir, f"batch_{batch_idx}.mp4")
            self._create_final_video(temp_sr_dir, batch_video_path, fps)
            
            return batch_video_path
            
        finally:
            # 清理临时目录
            shutil.rmtree(temp_frames_dir, ignore_errors=True)
            shutil.rmtree(temp_sr_dir, ignore_errors=True)
            
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
                
                # 处理重叠区域
                if x > 0 or y > 0:
                    # 需要混合重叠区域
                    existing_chunk = sr_image.crop((out_x1, out_y1, out_x2, out_y2))
                    if existing_chunk.size == sr_chunk.size:
                        # 简单平均混合
                        blended_chunk = Image.blend(existing_chunk, sr_chunk, 0.5)
                        sr_image.paste(blended_chunk, (out_x1, out_y1))
                    else:
                        sr_image.paste(sr_chunk, (out_x1, out_y1))
                else:
                    sr_image.paste(sr_chunk, (out_x1, out_y1))
                    
        return sr_image
        
    def _create_final_video(self, frames_dir, output_path, fps):
        """创建最终视频"""
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        
        if not frame_files:
            raise Exception("没有找到处理后的帧")
            
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
                
                if frame is not None:
                    out.write(frame)
                    
        finally:
            out.release()
            
    def _merge_videos(self, video_paths, output_path):
        """合并多个视频"""
        if not video_paths:
            return
            
        if len(video_paths) == 1:
            shutil.copy2(video_paths[0], output_path)
            return
            
        # 使用OpenCV合并视频
        # 读取第一个视频获取参数
        cap = cv2.VideoCapture(video_paths[0])
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # 创建输出视频
        fourcc = cv2.VideoWriter_fourcc(*self.codec_var.get())
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            for video_path in video_paths:
                if self.should_stop:
                    break
                    
                cap = cv2.VideoCapture(video_path)
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                    
                cap.release()
                
        finally:
            out.release()
            
    def _processing_finished(self):
        """处理完成回调"""
        self.is_processing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        if self.should_stop:
            self.update_status("处理已停止")
        else:
            self.update_status("处理完成！")
            messagebox.showinfo("完成", "视频超分辨率处理完成！")
            
    def update_status(self, text):
        """更新状态"""
        self.status_label.config(text=text)
        current_time = time.strftime("%H:%M:%S")
        self.time_label.config(text=current_time)
        
    def update_monitor(self):
        """更新监控图表"""
        if not self.is_processing:
            return
            
        current_time = time.time()
        
        # 更新数据
        if self.processing_stats['start_time']:
            elapsed_time = current_time - self.processing_stats['start_time']
            progress_percent = (self.processing_stats['processed_frames'] / 
                              max(self.processing_stats['total_frames'], 1)) * 100
            
            self.progress_data.append((elapsed_time, progress_percent))
            
            # 内存使用
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            self.memory_data.append((elapsed_time, memory_mb))
            
            # 处理速度
            if len(self.processing_stats['processing_times']) > 0:
                recent_times = self.processing_stats['processing_times'][-10:]
                avg_time = sum(recent_times) / len(recent_times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                self.time_data.append((elapsed_time, fps))
                
        # 更新图表
        self._update_charts()
        
        # 继续更新
        if self.is_processing:
            self.root.after(1000, self.update_monitor)
            
    def _update_charts(self):
        """更新图表显示"""
        # 清除旧图表
        self.progress_ax.clear()
        self.memory_ax.clear()
        self.time_ax.clear()
        self.fps_ax.clear()
        
        # 重新设置标题和标签
        self.progress_ax.set_title("处理进度")
        self.progress_ax.set_xlabel("时间 (秒)")
        self.progress_ax.set_ylabel("完成百分比 (%)")
        self.progress_ax.grid(True, alpha=0.3)
        
        self.memory_ax.set_title("内存使用")
        self.memory_ax.set_xlabel("时间 (秒)")
        self.memory_ax.set_ylabel("内存 (MB)")
        self.memory_ax.grid(True, alpha=0.3)
        
        self.time_ax.set_title("处理时间")
        self.time_ax.set_xlabel("时间 (秒)")
        self.time_ax.set_ylabel("每帧时间 (秒)")
        self.time_ax.grid(True, alpha=0.3)
        
        self.fps_ax.set_title("处理速度")
        self.fps_ax.set_xlabel("时间 (秒)")
        self.fps_ax.set_ylabel("FPS")
        self.fps_ax.grid(True, alpha=0.3)
        
        # 绘制数据
        if self.progress_data:
            times, progress = zip(*self.progress_data)
            self.progress_ax.plot(times, progress, 'b-', linewidth=2)
            
        if self.memory_data:
            times, memory = zip(*self.memory_data)
            self.memory_ax.plot(times, memory, 'r-', linewidth=2)
            
        if len(self.processing_stats['processing_times']) > 0:
            frame_indices = list(range(len(self.processing_stats['processing_times'])))
            self.time_ax.plot(frame_indices, self.processing_stats['processing_times'], 'g-', linewidth=1)
            
        if self.time_data:
            times, fps = zip(*self.time_data)
            self.fps_ax.plot(times, fps, 'm-', linewidth=2)
            
        self.monitor_fig.tight_layout()
        self.monitor_canvas.draw()
        
    def run(self):
        """运行GUI"""
        self.root.mainloop()

def main():
    """主函数"""
    app = VideoSuperResolutionGUI()
    app.run()

if __name__ == "__main__":
    main()