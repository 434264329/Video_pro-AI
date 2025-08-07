#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI超分辨率 - 高速视频分帧工具 (优化版)
保持原始分辨率，不进行任何裁剪或压缩，优化处理速度
"""

import os
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import traceback
from datetime import datetime
import queue
import time
import numpy as np

class FastVideoFrameExtractor:
    """高速视频分帧提取器 - 保持原始分辨率"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI超分辨率 - 高速视频分帧工具")
        self.root.geometry("900x800")
        self.root.resizable(True, True)
        
        # 设置窗口居中
        self.center_window()
        
        # 配置变量
        self.hr_video_path = tk.StringVar()
        self.lr_video_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="data")
        self.frame_interval = tk.IntVar(value=30)  # 每30帧提取一帧
        self.max_frames = tk.IntVar(value=1000)    # 最大提取帧数
        self.image_format = tk.StringVar(value="png")  # 图像格式
        self.jpeg_quality = tk.IntVar(value=95)    # JPEG质量(当选择JPEG时)
        self.preserve_original_size = tk.BooleanVar(value=True)  # 保持原始分辨率
        
        # 状态变量
        self.is_processing = False
        self.processing_thread = None
        self.stop_requested = False
        self.progress_queue = queue.Queue()
        
        # 创建界面
        self.create_widgets()
        
        # 设置默认输出目录
        self.set_default_output_dir()
        
        # 定期检查进度更新
        self.check_progress_updates()
    
    def center_window(self):
        """窗口居中显示"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="高速视频分帧工具 (优化版)", font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 25))
        
        # 视频选择区域
        self.create_video_selection_area(main_frame, 1)
        
        # 输出设置区域
        self.create_output_settings_area(main_frame, 2)
        
        # 分帧参数区域
        self.create_frame_settings_area(main_frame, 3)
        
        # 预览区域
        self.create_preview_area(main_frame, 4)
        
        # 控制按钮区域
        self.create_control_area(main_frame, 5)
        
        # 进度区域
        self.create_progress_area(main_frame, 6)
        
        # 状态栏
        self.create_status_bar(main_frame, 7)
    
    def create_video_selection_area(self, parent, row):
        """创建视频选择区域"""
        video_frame = ttk.LabelFrame(parent, text="视频文件选择", padding="15")
        video_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        video_frame.columnconfigure(1, weight=1)
        
        # 高分辨率视频
        ttk.Label(video_frame, text="高分辨率视频:", font=("Arial", 10)).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        hr_entry = ttk.Entry(video_frame, textvariable=self.hr_video_path, state="readonly", font=("Arial", 9))
        hr_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Button(video_frame, text="选择", command=self.select_hr_video).grid(row=0, column=2)
        
        # 低分辨率视频
        ttk.Label(video_frame, text="低分辨率视频:", font=("Arial", 10)).grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(15, 0))
        lr_entry = ttk.Entry(video_frame, textvariable=self.lr_video_path, state="readonly", font=("Arial", 9))
        lr_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(15, 0))
        ttk.Button(video_frame, text="选择", command=self.select_lr_video).grid(row=1, column=2, pady=(15, 0))
    
    def create_output_settings_area(self, parent, row):
        """创建输出设置区域"""
        output_frame = ttk.LabelFrame(parent, text="输出设置", padding="15")
        output_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        output_frame.columnconfigure(1, weight=1)
        
        # 输出目录
        ttk.Label(output_frame, text="输出目录:", font=("Arial", 10)).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        output_entry = ttk.Entry(output_frame, textvariable=self.output_dir, font=("Arial", 9))
        output_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Button(output_frame, text="选择", command=self.select_output_dir).grid(row=0, column=2)
        
        # 图像格式选择
        ttk.Label(output_frame, text="图像格式:", font=("Arial", 10)).grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        format_combo = ttk.Combobox(output_frame, textvariable=self.image_format, width=8)
        format_combo['values'] = ('png', 'jpg', 'bmp', 'tiff')
        format_combo.grid(row=1, column=1, sticky=tk.W, pady=(10, 0))
        
        # JPEG质量(仅当选择JPEG时显示)
        self.jpeg_quality_label = ttk.Label(output_frame, text="JPEG质量:", font=("Arial", 10))
        self.jpeg_quality_spin = ttk.Spinbox(output_frame, from_=1, to=100, textvariable=self.jpeg_quality, width=5)
        
        # 保持原始分辨率选项
        preserve_check = ttk.Checkbutton(output_frame, text="保持原始分辨率 (推荐)", 
                                       variable=self.preserve_original_size, 
                                       style="TCheckbutton")
        preserve_check.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        
        # 绑定格式变化事件
        self.image_format.trace_add('write', self.update_format_options)
        self.update_format_options()
        
        # 说明文本
        info_text = "将在输出目录下创建 train/hr, train/lr, val/hr, val/lr 文件夹"
        ttk.Label(output_frame, text=info_text, font=("Arial", 9), foreground="gray").grid(
            row=3, column=0, columnspan=3, sticky=tk.W, pady=(10, 0)
        )
    
    def update_format_options(self, *args):
        """更新格式选项显示"""
        if self.image_format.get() == 'jpg':
            self.jpeg_quality_label.grid(row=1, column=2, sticky=tk.W, padx=(20, 5), pady=(10, 0))
            self.jpeg_quality_spin.grid(row=1, column=3, sticky=tk.W, pady=(10, 0))
        else:
            self.jpeg_quality_label.grid_remove()
            self.jpeg_quality_spin.grid_remove()
    
    def create_frame_settings_area(self, parent, row):
        """创建分帧参数区域"""
        frame_settings = ttk.LabelFrame(parent, text="分帧参数", padding="15")
        frame_settings.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # 第一行：帧间隔和最大帧数
        ttk.Label(frame_settings, text="帧间隔:", font=("Arial", 10)).grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        interval_spin = ttk.Spinbox(frame_settings, from_=1, to=120, textvariable=self.frame_interval, width=12)
        interval_spin.grid(row=0, column=1, sticky=tk.W, padx=(0, 30))
        
        ttk.Label(frame_settings, text="最大帧数:", font=("Arial", 10)).grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        max_frames_spin = ttk.Spinbox(frame_settings, from_=100, to=10000, textvariable=self.max_frames, width=12)
        max_frames_spin.grid(row=0, column=3, sticky=tk.W)
        
        # 说明文本
        info_text = "帧间隔：每N帧提取一帧 | 最大帧数：限制总提取数量"
        ttk.Label(frame_settings, text=info_text, font=("Arial", 9), foreground="gray").grid(
            row=1, column=0, columnspan=4, sticky=tk.W, pady=(10, 0)
        )
    
    def create_preview_area(self, parent, row):
        """创建预览区域"""
        preview_frame = ttk.LabelFrame(parent, text="视频预览", padding="15")
        preview_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        
        # HR预览
        self.hr_preview_frame = ttk.Frame(preview_frame)
        self.hr_preview_frame.grid(row=0, column=0, padx=(0, 10))
        ttk.Label(self.hr_preview_frame, text="高分辨率预览", font=("Arial", 10, "bold")).pack()
        self.hr_preview_label = ttk.Label(self.hr_preview_frame, text="未选择视频", 
                                         background="lightgray", width=35, anchor="center",
                                         font=("Arial", 9))
        self.hr_preview_label.pack(pady=10)
        
        # LR预览
        self.lr_preview_frame = ttk.Frame(preview_frame)
        self.lr_preview_frame.grid(row=0, column=1, padx=(10, 0))
        ttk.Label(self.lr_preview_frame, text="低分辨率预览", font=("Arial", 10, "bold")).pack()
        self.lr_preview_label = ttk.Label(self.lr_preview_frame, text="未选择视频", 
                                         background="lightgray", width=35, anchor="center",
                                         font=("Arial", 9))
        self.lr_preview_label.pack(pady=10)
    
    def create_control_area(self, parent, row):
        """创建控制按钮区域"""
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=row, column=0, columnspan=3, pady=(0, 15))
        
        # 分析视频按钮
        self.analyze_btn = ttk.Button(control_frame, text="分析视频", command=self.analyze_videos,
                                     style="Accent.TButton")
        self.analyze_btn.pack(side=tk.LEFT, padx=(0, 15))
        
        # 开始分帧按钮
        self.start_btn = ttk.Button(control_frame, text="开始分帧", command=self.start_extraction,
                                   style="Accent.TButton")
        self.start_btn.pack(side=tk.LEFT, padx=(0, 15))
        
        # 停止按钮
        self.stop_btn = ttk.Button(control_frame, text="停止", command=self.stop_extraction, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 15))
        
        # 清空按钮
        self.clear_btn = ttk.Button(control_frame, text="清空", command=self.clear_all)
        self.clear_btn.pack(side=tk.LEFT)
    
    def create_progress_area(self, parent, row):
        """创建进度区域"""
        progress_frame = ttk.LabelFrame(parent, text="处理进度", padding="15")
        progress_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        progress_frame.columnconfigure(0, weight=1)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100,
                                           style="TProgressbar")
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 进度文本
        self.progress_text = tk.StringVar(value="准备就绪")
        ttk.Label(progress_frame, textvariable=self.progress_text, font=("Arial", 10)).grid(row=1, column=0, sticky=tk.W)
        
        # 速度和时间信息
        self.speed_text = tk.StringVar(value="")
        ttk.Label(progress_frame, textvariable=self.speed_text, font=("Arial", 9), foreground="gray").grid(row=2, column=0, sticky=tk.W)
    
    def create_status_bar(self, parent, row):
        """创建状态栏"""
        self.status_var = tk.StringVar(value="请选择视频文件")
        status_bar = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W,
                              font=("Arial", 9), padding="5")
        status_bar.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(15, 0))
    
    def check_progress_updates(self):
        """检查进度更新"""
        try:
            while True:
                progress_data = self.progress_queue.get_nowait()
                self.update_progress_display(progress_data)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.check_progress_updates)
    
    def update_progress_display(self, data):
        """更新进度显示"""
        if 'progress' in data:
            self.progress_var.set(data['progress'])
        if 'text' in data:
            self.progress_text.set(data['text'])
        if 'speed' in data:
            self.speed_text.set(data['speed'])
        if 'status' in data:
            self.status_var.set(data['status'])
    
    def update_status(self, message):
        """更新状态栏"""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def set_default_output_dir(self):
        """设置默认输出目录"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_output = os.path.join(current_dir, "data")
        self.output_dir.set(default_output)
    
    def select_hr_video(self):
        """选择高分辨率视频"""
        file_path = filedialog.askopenfilename(
            title="选择高分辨率视频",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("所有文件", "*.*")
            ]
        )
        if file_path:
            self.hr_video_path.set(file_path)
            self.update_preview()
            self.update_status(f"已选择HR视频: {os.path.basename(file_path)}")
    
    def select_lr_video(self):
        """选择低分辨率视频"""
        file_path = filedialog.askopenfilename(
            title="选择低分辨率视频",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("所有文件", "*.*")
            ]
        )
        if file_path:
            self.lr_video_path.set(file_path)
            self.update_preview()
            self.update_status(f"已选择LR视频: {os.path.basename(file_path)}")
    
    def select_output_dir(self):
        """选择输出目录"""
        dir_path = filedialog.askdirectory(title="选择输出目录")
        if dir_path:
            self.output_dir.set(dir_path)
    
    def update_preview(self):
        """更新视频预览"""
        try:
            # 更新HR预览
            if self.hr_video_path.get():
                hr_cap = cv2.VideoCapture(self.hr_video_path.get())
                if hr_cap.isOpened():
                    ret, frame = hr_cap.read()
                    if ret:
                        height, width = frame.shape[:2]
                        fps = hr_cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(hr_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = frame_count / fps if fps > 0 else 0
                        
                        preview_text = f"分辨率: {width}x{height}\n帧率: {fps:.1f} FPS\n时长: {duration:.1f}s\n总帧数: {frame_count}"
                        self.hr_preview_label.config(text=preview_text)
                hr_cap.release()
            
            # 更新LR预览
            if self.lr_video_path.get():
                lr_cap = cv2.VideoCapture(self.lr_video_path.get())
                if lr_cap.isOpened():
                    ret, frame = lr_cap.read()
                    if ret:
                        height, width = frame.shape[:2]
                        fps = lr_cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(lr_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = frame_count / fps if fps > 0 else 0
                        
                        preview_text = f"分辨率: {width}x{height}\n帧率: {fps:.1f} FPS\n时长: {duration:.1f}s\n总帧数: {frame_count}"
                        self.lr_preview_label.config(text=preview_text)
                lr_cap.release()
                
        except Exception as e:
            print(f"预览更新失败: {e}")
    
    def analyze_videos(self):
        """分析视频信息"""
        if not self.hr_video_path.get() or not self.lr_video_path.get():
            messagebox.showwarning("警告", "请先选择高分辨率和低分辨率视频文件")
            return
        
        try:
            # 分析HR视频
            hr_cap = cv2.VideoCapture(self.hr_video_path.get())
            hr_frame_count = int(hr_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            hr_fps = hr_cap.get(cv2.CAP_PROP_FPS)
            ret, hr_frame = hr_cap.read()
            hr_resolution = f"{hr_frame.shape[1]}x{hr_frame.shape[0]}" if ret else "未知"
            hr_cap.release()
            
            # 分析LR视频
            lr_cap = cv2.VideoCapture(self.lr_video_path.get())
            lr_frame_count = int(lr_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            lr_fps = lr_cap.get(cv2.CAP_PROP_FPS)
            ret, lr_frame = lr_cap.read()
            lr_resolution = f"{lr_frame.shape[1]}x{lr_frame.shape[0]}" if ret else "未知"
            lr_cap.release()
            
            # 计算预估提取帧数
            min_frame_count = min(hr_frame_count, lr_frame_count)
            interval = self.frame_interval.get()
            max_frames = self.max_frames.get()
            estimated_frames = min(min_frame_count // interval, max_frames)
            
            # 显示分析结果
            analysis_text = f"""视频分析结果:

HR视频: {hr_frame_count} 帧, {hr_fps:.1f} FPS, {hr_resolution}
LR视频: {lr_frame_count} 帧, {lr_fps:.1f} FPS, {lr_resolution}

提取设置:
帧间隔: {interval}
最大帧数: {max_frames}
保持原始分辨率: {'是' if self.preserve_original_size.get() else '否'}

预估提取:
总帧数: {estimated_frames} 帧
训练集: {int(estimated_frames * 0.8)} 帧
验证集: {int(estimated_frames * 0.2)} 帧

注意: HR图像将保持原始{hr_resolution}分辨率，不进行任何裁剪或压缩"""
            
            messagebox.showinfo("分析结果", analysis_text)
            self.update_status(f"分析完成 - 预估提取 {estimated_frames} 帧")
            
        except Exception as e:
            messagebox.showerror("错误", f"视频分析失败:\n{str(e)}")
    
    def start_extraction(self):
        """开始分帧提取"""
        if not self.hr_video_path.get() or not self.lr_video_path.get():
            messagebox.showwarning("警告", "请先选择高分辨率和低分辨率视频文件")
            return
        
        if not self.output_dir.get():
            messagebox.showwarning("警告", "请选择输出目录")
            return
        
        # 检查输出目录
        if not os.path.exists(self.output_dir.get()):
            try:
                os.makedirs(self.output_dir.get())
            except Exception as e:
                messagebox.showerror("错误", f"无法创建输出目录:\n{str(e)}")
                return
        
        # 开始处理
        self.is_processing = True
        self.stop_requested = False
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress_var.set(0)
        self.progress_text.set("正在初始化...")
        
        self.processing_thread = threading.Thread(target=self._extraction_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _create_output_directories(self):
        """创建输出目录结构"""
        base_dir = self.output_dir.get()
        
        # 创建目录结构
        dirs = [
            os.path.join(base_dir, "train", "hr"),
            os.path.join(base_dir, "train", "lr"),
            os.path.join(base_dir, "val", "hr"),
            os.path.join(base_dir, "val", "lr")
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def _extraction_worker(self):
        """分帧提取工作线程 - 高速优化版本"""
        start_time = time.time()
        
        try:
            # 创建输出目录
            self._create_output_directories()
            
            # 打开视频文件
            hr_cap = cv2.VideoCapture(self.hr_video_path.get())
            lr_cap = cv2.VideoCapture(self.lr_video_path.get())
            
            if not hr_cap.isOpened() or not lr_cap.isOpened():
                raise Exception("无法打开视频文件")
            
            # 🚀 优化1: 设置视频解码器缓冲区
            hr_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            lr_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # 获取视频信息
            hr_frame_count = int(hr_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            lr_frame_count = int(lr_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            min_frame_count = min(hr_frame_count, lr_frame_count)
            
            # 计算提取参数
            interval = self.frame_interval.get()
            max_frames = self.max_frames.get()
            total_extract = min(min_frame_count // interval, max_frames)
            
            # 获取现有文件数量
            existing_train, existing_val = self._get_existing_file_counts()
            
            # 处理增量生成逻辑
            if existing_train > 0 or existing_val > 0:
                response = messagebox.askyesnocancel(
                    "发现现有数据",
                    f"发现现有训练集 {existing_train} 帧，验证集 {existing_val} 帧\n\n"
                    "选择操作：\n"
                    "是 - 增量添加新帧\n"
                    "否 - 清空现有文件重新开始\n"
                    "取消 - 停止操作"
                )
                
                if response is None:  # 取消
                    self._extraction_stopped()
                    return
                elif response is False:  # 否 - 清空现有文件
                    self._clear_existing_files()
                    existing_train, existing_val = 0, 0
            
            # 获取视频基础名称
            hr_base_name = self._get_video_base_name(self.hr_video_path.get())
            
            # 获取图像格式和质量设置
            image_format = self.image_format.get()
            jpeg_quality = self.jpeg_quality.get()
            
            # 开始提取
            extracted_count = 0
            train_count = 0
            val_count = 0
            
            # 🚀 优化2: 减少UI更新频率
            ui_update_interval = max(1, total_extract // 100)  # 最多更新100次
            
            # 🚀 优化3: 预分配帧列表，批量处理
            frame_indices = list(range(0, min_frame_count, interval))[:total_extract]
            
            for idx, frame_idx in enumerate(frame_indices):
                if not self.is_processing or self.stop_requested:
                    break
                
                # 设置帧位置
                hr_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                lr_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                # 读取帧
                hr_ret, hr_frame = hr_cap.read()
                lr_ret, lr_frame = lr_cap.read()
                
                if not hr_ret or not lr_ret:
                    break
                
                # 🚀 关键优化: 保持原始分辨率，不进行任何裁剪或压缩
                if self.preserve_original_size.get():
                    # 完全保持原始分辨率，不做任何处理
                    pass
                else:
                    # 如果用户选择不保持原始分辨率，可以在这里添加自定义处理
                    # 但默认推荐保持原始分辨率
                    pass
                
                # 确定保存到训练集还是验证集（80%训练，20%验证）
                is_train = (extracted_count % 5) != 0  # 每5个中4个用于训练
                
                if is_train:
                    hr_dir = os.path.join(self.output_dir.get(), "train", "hr")
                    lr_dir = os.path.join(self.output_dir.get(), "train", "lr")
                    current_count = existing_train + train_count
                else:
                    hr_dir = os.path.join(self.output_dir.get(), "val", "hr")
                    lr_dir = os.path.join(self.output_dir.get(), "val", "lr")
                    current_count = existing_val + val_count
                
                # 生成文件名
                file_ext = f".{image_format}"
                filename = f"{hr_base_name}_frame_{current_count:06d}{file_ext}"
                hr_path = os.path.join(hr_dir, filename)
                lr_path = os.path.join(lr_dir, filename)
                
                # 🚀 优化4: 使用高速保存函数
                hr_success = self._fast_imwrite(hr_path, hr_frame, image_format, jpeg_quality)
                lr_success = self._fast_imwrite(lr_path, lr_frame, image_format, jpeg_quality)
                
                if hr_success and lr_success:
                    extracted_count += 1
                    if is_train:
                        train_count += 1
                    else:
                        val_count += 1
                    
                    # 🚀 优化5: 减少UI更新频率，计算处理速度
                    if extracted_count % ui_update_interval == 0 or extracted_count == total_extract:
                        elapsed_time = time.time() - start_time
                        speed = extracted_count / elapsed_time if elapsed_time > 0 else 0
                        eta = (total_extract - extracted_count) / speed if speed > 0 else 0
                        
                        progress_data = {
                            'progress': (extracted_count / total_extract) * 100,
                            'text': f"已提取 {extracted_count}/{total_extract} 帧",
                            'speed': f"速度: {speed:.1f} 帧/秒 | 预计剩余: {eta:.0f}秒",
                            'status': f"处理中... {extracted_count}/{total_extract}"
                        }
                        self.progress_queue.put(progress_data)
                else:
                    print(f"保存失败: {filename}")
            
            # 释放资源
            hr_cap.release()
            lr_cap.release()
            
            if self.is_processing and not self.stop_requested:
                # 保存提取信息
                self._save_extraction_info(extracted_count, train_count, val_count)
                
                # 计算总数
                total_train = existing_train + train_count
                total_val = existing_val + val_count
                
                # 完成
                self._extraction_completed(extracted_count, total_train, total_val, time.time() - start_time)
            else:
                self._extraction_stopped()
                
        except Exception as e:
            error_msg = f"{str(e)}\n\n详细错误:\n{traceback.format_exc()}"
            self._extraction_error(error_msg)
        finally:
            self._reset_ui_state()
    
    def _get_existing_file_counts(self):
        """统计现有训练和验证集中的文件数量"""
        base_dir = self.output_dir.get()
        
        train_hr_dir = os.path.join(base_dir, "train", "hr")
        train_lr_dir = os.path.join(base_dir, "train", "lr")
        val_hr_dir = os.path.join(base_dir, "val", "hr")
        val_lr_dir = os.path.join(base_dir, "val", "lr")
        
        train_count = 0
        val_count = 0
        
        # 统计训练集
        if os.path.exists(train_hr_dir) and os.path.exists(train_lr_dir):
            hr_files = [f for f in os.listdir(train_hr_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            lr_files = [f for f in os.listdir(train_lr_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            train_count = min(len(hr_files), len(lr_files))
        
        # 统计验证集
        if os.path.exists(val_hr_dir) and os.path.exists(val_lr_dir):
            hr_files = [f for f in os.listdir(val_hr_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            lr_files = [f for f in os.listdir(val_lr_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            val_count = min(len(hr_files), len(lr_files))
        
        return train_count, val_count
    
    def _clear_existing_files(self):
        """清空现有文件"""
        base_dir = self.output_dir.get()
        dirs = [
            os.path.join(base_dir, "train", "hr"),
            os.path.join(base_dir, "train", "lr"),
            os.path.join(base_dir, "val", "hr"),
            os.path.join(base_dir, "val", "lr")
        ]
        
        for dir_path in dirs:
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        try:
                            os.remove(os.path.join(dir_path, file))
                        except:
                            pass
    
    def _fast_imwrite(self, filepath, image, image_format, jpeg_quality):
        """高速保存图像 - 支持多种格式"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            if image_format.lower() == 'jpg' or image_format.lower() == 'jpeg':
                # JPEG格式 - 速度最快
                success, encoded_img = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                if success:
                    with open(filepath, 'wb') as f:
                        f.write(encoded_img.tobytes())
                    return True
            elif image_format.lower() == 'png':
                # PNG格式 - 无损压缩
                success, encoded_img = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 1])  # 最快压缩
                if success:
                    with open(filepath, 'wb') as f:
                        f.write(encoded_img.tobytes())
                    return True
            else:
                # 其他格式使用OpenCV直接保存
                return cv2.imwrite(filepath, image)
                
        except Exception as e:
            print(f"保存图像时出错 {filepath}: {e}")
            return False
        
        return False
    
    def _get_video_base_name(self, video_path):
        """从视频路径中提取基础文件名"""
        if not video_path:
            return "video"
        
        # 获取文件名（不含路径）
        filename = os.path.basename(video_path)
        # 去除扩展名
        base_name = os.path.splitext(filename)[0]
        
        # 清理文件名，移除特殊字符，保留中文、英文、数字、下划线、连字符
        import re
        # 保留中文字符、英文字母、数字、下划线、连字符
        base_name = re.sub(r'[^\w\u4e00-\u9fff\-]', '_', base_name)
        
        # 如果文件名为空或只包含特殊字符，使用默认名称
        if not base_name or base_name.replace('_', '').replace('-', '') == '':
            base_name = "video"
        
        return base_name
    
    def _save_extraction_info(self, new_extracted, new_train, new_val):
        """保存提取信息到文件"""
        try:
            info_file = os.path.join(self.output_dir.get(), "extraction_info.txt")
            
            # 获取视频基础名称
            hr_base_name = self._get_video_base_name(self.hr_video_path.get())
            
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(f"视频分帧提取信息 (高速优化版)\n")
                f.write(f"提取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"HR视频: {self.hr_video_path.get()}\n")
                f.write(f"LR视频: {self.lr_video_path.get()}\n")
                f.write(f"视频基础名称: {hr_base_name}\n")
                f.write(f"本次新提取帧数: {new_extracted}\n")
                f.write(f"新增训练集: {new_train}\n")
                f.write(f"新增验证集: {new_val}\n")
                f.write(f"帧间隔: {self.frame_interval.get()}\n")
                f.write(f"图像格式: {self.image_format.get()}\n")
                f.write(f"保持原始分辨率: {self.preserve_original_size.get()}\n")
                if self.image_format.get().lower() in ['jpg', 'jpeg']:
                    f.write(f"JPEG质量: {self.jpeg_quality.get()}\n")
                f.write(f"命名格式: {hr_base_name}_frame_{{序号:06d}}.{self.image_format.get()}\n")
        except Exception as e:
            print(f"保存提取信息失败: {e}")
    
    def _extraction_completed(self, new_extracted, total_train, total_val, elapsed_time):
        """提取完成"""
        speed = new_extracted / elapsed_time if elapsed_time > 0 else 0
        
        message = f"""分帧提取完成！

本次新提取: {new_extracted} 帧
处理时间: {elapsed_time:.1f} 秒
平均速度: {speed:.1f} 帧/秒

当前总计: {total_train + total_val} 帧
├─ 训练集: {total_train} 帧
└─ 验证集: {total_val} 帧

特点:
✓ 保持原始分辨率，无裁剪压缩
✓ 高速处理，优化性能
✓ 支持多种图像格式

文件保存在: {self.output_dir.get()}

可以开始训练模型了！"""
        
        messagebox.showinfo("完成", message)
        
        # 更新UI
        progress_data = {
            'progress': 100,
            'text': f"完成 - 总计 {total_train + total_val} 帧 (新增 {new_extracted})",
            'speed': f"平均速度: {speed:.1f} 帧/秒",
            'status': "分帧提取完成"
        }
        self.progress_queue.put(progress_data)
    
    def _extraction_stopped(self):
        """提取停止"""
        progress_data = {
            'text': "提取已停止",
            'status': "提取已停止"
        }
        self.progress_queue.put(progress_data)
        messagebox.showinfo("停止", "分帧提取已停止")
    
    def _extraction_error(self, error_msg):
        """提取错误"""
        progress_data = {
            'text': "提取出错",
            'status': "提取出错"
        }
        self.progress_queue.put(progress_data)
        messagebox.showerror("错误", f"分帧提取出错:\n{error_msg}")
    
    def _reset_ui_state(self):
        """重置UI状态"""
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.is_processing = False
        self.stop_requested = False
    
    def stop_extraction(self):
        """停止分帧提取"""
        self.stop_requested = True
        self.is_processing = False
        self.update_status("正在停止...")
    
    def clear_all(self):
        """清空所有设置"""
        self.hr_video_path.set("")
        self.lr_video_path.set("")
        self.hr_preview_label.config(text="未选择视频")
        self.lr_preview_label.config(text="未选择视频")
        self.progress_var.set(0)
        self.progress_text.set("准备就绪")
        self.speed_text.set("")
        self.update_status("已清空所有设置")

def main():
    """程序入口点"""
    try:
        app = FastVideoFrameExtractor()
        app.root.mainloop()
    except Exception as e:
        import traceback
        print(f"程序启动失败: {e}")
        print(f"详细错误: {traceback.format_exc()}")
        input("按回车键退出...")

if __name__ == "__main__":
    main()