#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 超分辨率模型训练图形界面
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import datetime
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.training.train_manager import MemoryOptimizedTrainingManager as TrainingManager
from config.config import load_config, save_config, load_low_memory_config, print_gpu_info, get_optimal_config_for_gpu

class TrainingGUI:
    """训练图形界面"""
    
    def __init__(self):
        # 设置CUDA内存分配策略
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32,expandable_segments:True'
        torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优
        print("已设置 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,expandable_segments:True")
        
        # 更彻底的GPU内存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print("GPU内存已清理")
        
        self.root = tk.Tk()
        self.root.title("AI 超分辨率模型训练工具 - GPU专用版")
        self.root.geometry("1400x900")
        
        # 检查CUDA可用性
        if not torch.cuda.is_available():
            messagebox.showerror("错误", "未检测到CUDA设备，此训练器仅支持GPU训练")
            self.root.destroy()
            return
        
        # 获取GPU信息
        self.gpu_info = self.get_gpu_info()
        
        # 智能配置加载
        self.config = self.load_smart_config()
        
        # 训练状态
        self.is_training = False
        self.is_paused = False
        self.train_manager = None
        self.pretrained_model_path = None
        
        # 训练历史数据
        self.training_history = {
            'epochs': [],
            'generator_loss': [],
            'discriminator_loss': [],
            'psnr': [],
            'ssim': []
        }
        
        # 设置UI
        self.setup_ui()
        self.setup_callbacks()
        
        # 更新UI显示
        self.update_ui_from_config()

    def load_smart_config(self):
        """智能配置加载，根据GPU显存自动选择合适的配置"""
        gpu_memory = self.gpu_info["memory"]
        
        try:
            if gpu_memory < 6.0:
                print(f"检测到GPU显存: {gpu_memory:.1f}GB，自动使用低显存配置")
                if gpu_memory >= 4.0:
                    # 使用低显存配置
                    config = load_low_memory_config()
                    print("✅ 已加载低显存配置")
                else:
                    # 显存不足4GB，使用极限配置
                    config = self.create_ultra_low_memory_config()
                    print("✅ 已创建极限低显存配置")
            else:
                # 显存充足，使用标准配置
                config = load_config()
                print("✅ 已加载标准配置")
            
            return config
            
        except Exception as e:
            print(f"配置加载失败: {e}")
            print("使用手动创建的4GB显存配置")
            return self.create_ultra_low_memory_config()

    def create_ultra_low_memory_config(self):
        """创建极限低显存配置（适用于4GB及以下显存）"""
        return {
            "model": {
                "num_blocks": 3,
                "num_features": 24
            },
            "training": {
                "batch_size": 1,
                "learning_rate": 1e-4,
                "num_epochs": 100,
                "device": "cuda",
                "save_frequency": 10,
                "early_stopping_patience": 20,
                "gradient_accumulation_steps": 32,
                "mixed_precision": True,
                "memory_efficient": True,
                "max_cache_size": 25,
                "preload_batch_size": 1,
                "memory_cleanup_frequency": 3,
                "checkpoint_memory_cleanup": True,
                "image_max_size": 192,
                "crop_size": (96, 96),
                "enable_memory_monitoring": True,
                "memory_warning_threshold": 3.5
            },
            "data": {
                "train_lr_dir": "data/train/lr",
                "train_hr_dir": "data/train/hr",
                "val_lr_dir": "data/val/lr",
                "val_hr_dir": "data/val/hr",
                "num_workers": 1,
                "pin_memory": False
            },
            "paths": {
                "save_dir": "checkpoints",
                "log_dir": "logs",
                "output_dir": "outputs"
            },
            "loss": {
                "l1_weight": 1.0,
                "perceptual_weight": 0.02,
                "gan_weight": 0.002
            },
            "memory": {
                "gradient_accumulation_steps": 32,
                "max_cache_size": 25,
                "enable_checkpointing": True,
                "clear_cache_frequency": 3
            }
        }

    def get_gpu_info(self):
        """获取GPU信息"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return {
                "available": True,
                "name": gpu_name,
                "memory": gpu_memory
            }
        return {"available": False, "name": "无GPU", "memory": 0}
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="AI 超分辨率模型训练工具 - GPU专用版", 
                              font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # 左侧控制面板
        self.setup_control_panel(main_frame)
        
        # 右侧监控面板
        self.setup_monitor_panel(main_frame)
        
        # 底部状态栏
        self.setup_status_bar(main_frame)
        
    def setup_control_panel(self, parent):
        """设置控制面板"""
        control_frame = ttk.LabelFrame(parent, text="训练控制", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # GPU信息显示
        self.setup_gpu_info_panel(control_frame)
        
        # 内存优化设置
        self.setup_memory_optimization_panel(control_frame)
        
        # 配置设置
        config_frame = ttk.LabelFrame(control_frame, text="配置设置", padding="5")
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(config_frame, text="加载配置", 
                  command=self.load_config_file).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(config_frame, text="保存配置", 
                  command=self.save_config_file).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(config_frame, text="重置配置", 
                  command=self.reset_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(config_frame, text="低显存模式", 
                  command=self.load_low_memory_mode).pack(side=tk.LEFT)
        
        # 数据路径设置
        data_frame = ttk.LabelFrame(control_frame, text="数据路径", padding="5")
        data_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 训练数据
        train_frame = ttk.Frame(data_frame)
        train_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(train_frame, text="训练LR:", width=10).pack(side=tk.LEFT)
        self.train_lr_var = tk.StringVar(value=self.config['data']['train_lr_dir'])
        ttk.Entry(train_frame, textvariable=self.train_lr_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(train_frame, text="浏览", 
                  command=lambda: self.browse_folder(self.train_lr_var)).pack(side=tk.RIGHT)
        
        train_hr_frame = ttk.Frame(data_frame)
        train_hr_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(train_hr_frame, text="训练HR:", width=10).pack(side=tk.LEFT)
        self.train_hr_var = tk.StringVar(value=self.config['data']['train_hr_dir'])
        ttk.Entry(train_hr_frame, textvariable=self.train_hr_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(train_hr_frame, text="浏览", 
                  command=lambda: self.browse_folder(self.train_hr_var)).pack(side=tk.RIGHT)
        
        # 验证数据
        val_frame = ttk.Frame(data_frame)
        val_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(val_frame, text="验证LR:", width=10).pack(side=tk.LEFT)
        self.val_lr_var = tk.StringVar(value=self.config['data']['val_lr_dir'])
        ttk.Entry(val_frame, textvariable=self.val_lr_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(val_frame, text="浏览", 
                  command=lambda: self.browse_folder(self.val_lr_var)).pack(side=tk.RIGHT)
        
        val_hr_frame = ttk.Frame(data_frame)
        val_hr_frame.pack(fill=tk.X)
        
        ttk.Label(val_hr_frame, text="验证HR:", width=10).pack(side=tk.LEFT)
        self.val_hr_var = tk.StringVar(value=self.config['data']['val_hr_dir'])
        ttk.Entry(val_hr_frame, textvariable=self.val_hr_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(val_hr_frame, text="浏览", 
                  command=lambda: self.browse_folder(self.val_hr_var)).pack(side=tk.RIGHT)
        
        # 训练参数
        params_frame = ttk.LabelFrame(control_frame, text="训练参数", padding="5")
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 第一行参数
        params_row1 = ttk.Frame(params_frame)
        params_row1.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(params_row1, text="Epochs:").pack(side=tk.LEFT)
        self.epochs_var = tk.StringVar(value=str(self.config['training']['num_epochs']))
        ttk.Entry(params_row1, textvariable=self.epochs_var, width=8).pack(side=tk.LEFT, padx=(5, 15))
        
        ttk.Label(params_row1, text="Batch Size:").pack(side=tk.LEFT)
        self.batch_size_var = tk.StringVar(value=str(self.config['training']['batch_size']))
        ttk.Entry(params_row1, textvariable=self.batch_size_var, width=8).pack(side=tk.LEFT, padx=(5, 0))
        
        # 第二行参数
        params_row2 = ttk.Frame(params_frame)
        params_row2.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(params_row2, text="Learning Rate:").pack(side=tk.LEFT)
        self.lr_var = tk.StringVar(value=str(self.config['training']['learning_rate']))
        ttk.Entry(params_row2, textvariable=self.lr_var, width=12).pack(side=tk.LEFT, padx=(5, 15))
        
        ttk.Label(params_row2, text="Device:").pack(side=tk.LEFT)
        self.device_var = tk.StringVar(value="cuda")
        device_combo = ttk.Combobox(params_row2, textvariable=self.device_var, 
                                   values=['cuda'], width=8, state="readonly")
        device_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # 第三行参数（梯度累积）
        params_row3 = ttk.Frame(params_frame)
        params_row3.pack(fill=tk.X)
        
        ttk.Label(params_row3, text="梯度累积步数:").pack(side=tk.LEFT)
        self.grad_accum_var = tk.StringVar(value=str(self.config['training'].get('gradient_accumulation_steps', 1)))
        ttk.Entry(params_row3, textvariable=self.grad_accum_var, width=8).pack(side=tk.LEFT, padx=(5, 15))
        
        # 混合精度训练
        self.mixed_precision_var = tk.BooleanVar(value=self.config['training'].get('mixed_precision', False))
        ttk.Checkbutton(params_row3, text="混合精度训练", 
                       variable=self.mixed_precision_var).pack(side=tk.LEFT, padx=(5, 0))
        
        # 训练控制按钮
        control_buttons_frame = ttk.LabelFrame(control_frame, text="训练控制", padding="5")
        control_buttons_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 第一行按钮
        buttons_row1 = ttk.Frame(control_buttons_frame)
        buttons_row1.pack(fill=tk.X, pady=(0, 5))
        
        self.start_button = ttk.Button(buttons_row1, text="开始训练", 
                                     command=self.start_training)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_button = ttk.Button(buttons_row1, text="停止训练", 
                                    command=self.stop_training, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.pause_button = ttk.Button(buttons_row1, text="暂停训练", 
                                     command=self.pause_training, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT)
        
        # 第二行按钮
        buttons_row2 = ttk.Frame(control_buttons_frame)
        buttons_row2.pack(fill=tk.X)
        
        self.incremental_button = ttk.Button(buttons_row2, text="增量训练", 
                                           command=self.start_incremental_training, 
                                           state=tk.DISABLED)
        self.incremental_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # 模型加载按钮
        self.load_model_button = ttk.Button(buttons_row2, text="加载预训练模型", 
                                          command=self.load_pretrained_model)
        self.load_model_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # 进度显示
        progress_frame = ttk.LabelFrame(control_frame, text="训练进度", padding="5")
        progress_frame.pack(fill=tk.X)
        
        self.progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress.pack(fill=tk.X, pady=(0, 5))
        
        self.progress_label = ttk.Label(progress_frame, text="等待开始训练...")
        self.progress_label.pack()
        
    def setup_gpu_info_panel(self, parent):
        """设置GPU信息面板"""
        gpu_frame = ttk.LabelFrame(parent, text="GPU信息", padding="5")
        gpu_frame.pack(fill=tk.X, pady=(0, 10))
        
        # GPU状态显示
        self.gpu_name_label = ttk.Label(gpu_frame, text=f"GPU: {self.gpu_info['name']}")
        self.gpu_name_label.pack(anchor=tk.W)
        
        self.gpu_memory_label = ttk.Label(gpu_frame, 
                                        text=f"显存: {self.gpu_info['memory']:.1f} GB")
        self.gpu_memory_label.pack(anchor=tk.W)
        
        # 推荐设置按钮
        ttk.Button(gpu_frame, text="应用推荐设置", 
                  command=self.apply_recommended_settings).pack(anchor=tk.W, pady=(5, 0))
            
    def setup_memory_optimization_panel(self, parent):
        """设置内存优化面板"""
        memory_frame = ttk.LabelFrame(parent, text="内存优化", padding="5")
        memory_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 内存优化选项
        self.enable_memory_optimization_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(memory_frame, text="启用内存优化", 
                       variable=self.enable_memory_optimization_var).pack(anchor=tk.W)
        
        self.enable_gradient_checkpointing_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(memory_frame, text="启用梯度检查点", 
                       variable=self.enable_gradient_checkpointing_var).pack(anchor=tk.W)
        
        # 缓存设置
        cache_frame = ttk.Frame(memory_frame)
        cache_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(cache_frame, text="缓存大小:").pack(side=tk.LEFT)
        self.cache_size_var = tk.StringVar(value=str(self.config['data'].get('max_cache_size', 100)))
        ttk.Entry(cache_frame, textvariable=self.cache_size_var, width=8).pack(side=tk.LEFT, padx=(5, 0))

    def setup_monitor_panel(self, parent):
        """设置监控面板"""
        monitor_frame = ttk.LabelFrame(parent, text="训练监控", padding="10")
        monitor_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 创建Notebook用于多标签页
        self.notebook = ttk.Notebook(monitor_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 实时监控标签页
        self.setup_realtime_tab()
        
        # 训练曲线标签页
        self.setup_curves_tab()
        
        # 日志标签页
        self.setup_log_tab()
        
    def setup_realtime_tab(self):
        """设置实时监控标签页"""
        realtime_frame = ttk.Frame(self.notebook)
        self.notebook.add(realtime_frame, text="实时监控")
        
        # 当前状态显示
        status_frame = ttk.LabelFrame(realtime_frame, text="当前状态", padding="10")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 创建状态标签
        self.current_epoch_label = ttk.Label(status_frame, text="当前Epoch: 0/0", 
                                           font=("Arial", 12))
        self.current_epoch_label.pack(anchor=tk.W)
        
        self.current_loss_label = ttk.Label(status_frame, text="生成器损失: 0.0000", 
                                          font=("Arial", 12))
        self.current_loss_label.pack(anchor=tk.W)
        
        self.current_psnr_label = ttk.Label(status_frame, text="验证PSNR: 0.00 dB", 
                                          font=("Arial", 12))
        self.current_psnr_label.pack(anchor=tk.W)
        
        self.best_psnr_label = ttk.Label(status_frame, text="最佳PSNR: 0.00 dB", 
                                       font=("Arial", 12, "bold"))
        self.best_psnr_label.pack(anchor=tk.W)
        
        # 内存使用显示
        self.memory_label = ttk.Label(status_frame, text="GPU内存: 0.0/0.0 GB", 
                                    font=("Arial", 12))
        self.memory_label.pack(anchor=tk.W)
        
        # 实时图表
        self.realtime_fig = Figure(figsize=(8, 6), dpi=100)
        self.realtime_canvas = FigureCanvasTkAgg(self.realtime_fig, realtime_frame)
        self.realtime_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_curves_tab(self):
        """设置训练曲线标签页"""
        curves_frame = ttk.Frame(self.notebook)
        self.notebook.add(curves_frame, text="训练曲线")
        
        # 训练曲线图表
        self.curves_fig = Figure(figsize=(12, 8), dpi=100)
        self.curves_canvas = FigureCanvasTkAgg(self.curves_fig, curves_frame)
        self.curves_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_log_tab(self):
        """设置日志标签页"""
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="训练日志")
        
        # 日志文本框
        log_text_frame = ttk.Frame(log_frame)
        log_text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_text_frame, wrap=tk.WORD, font=("Consolas", 10))
        log_scrollbar = ttk.Scrollbar(log_text_frame, orient=tk.VERTICAL, 
                                     command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 日志控制
        log_control_frame = ttk.Frame(log_frame)
        log_control_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(log_control_frame, text="清空日志", 
                  command=self.clear_log).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(log_control_frame, text="保存日志", 
                  command=self.save_log).pack(side=tk.LEFT)
        
    def setup_status_bar(self, parent):
        """设置状态栏"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 状态标签
        self.status_label = ttk.Label(status_frame, text="就绪", 
                                    relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 时间显示
        self.time_label = ttk.Label(status_frame, text="训练时间: 00:00:00", 
                                  relief=tk.SUNKEN)
        self.time_label.pack(side=tk.RIGHT, padx=(5, 0))
        
    def setup_callbacks(self):
        """设置回调函数"""
        pass
        
    def browse_folder(self, var):
        """浏览文件夹"""
        folder = filedialog.askdirectory()
        if folder:
            var.set(folder)
            
    def load_config_file(self):
        """加载配置文件"""
        file_path = filedialog.askopenfilename(
            title="选择配置文件",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                self.update_ui_from_config()
                self.log_message(f"配置已从 {file_path} 加载")
            except Exception as e:
                messagebox.showerror("错误", f"加载配置失败: {str(e)}")
                
    def save_config_file(self):
        """保存配置文件"""
        file_path = filedialog.asksaveasfilename(
            title="保存配置文件",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.update_config_from_ui()
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
                self.log_message(f"配置已保存到 {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存配置失败: {str(e)}")
                
    def reset_config(self):
        """重置配置"""
        self.config = load_config()
        self.update_ui_from_config()
        self.log_message("配置已重置为默认值")
        
    def apply_recommended_settings(self):
        """应用推荐设置"""
        if self.gpu_info['available']:
            recommended_config = get_optimal_config_for_gpu()
            
            # 更新配置
            self.config['training'].update(recommended_config)
            
            # 更新UI
            self.update_ui_from_config()
            self.log_message(f"已应用 {self.gpu_info['memory']:.1f}GB GPU 的推荐设置")
        else:
            messagebox.showwarning("警告", "未检测到GPU，无法应用推荐设置")
            
    def load_low_memory_mode(self):
        """加载低显存模式"""
        try:
            gpu_memory = self.gpu_info["memory"]
            
            if gpu_memory >= 4.0:
                self.config = load_low_memory_config()
                messagebox.showinfo("成功", f"已加载低显存GPU训练模式配置\n当前显存: {gpu_memory:.1f}GB")
            else:
                self.config = self.create_ultra_low_memory_config()
                messagebox.showinfo("成功", f"已加载极限低显存配置\n当前显存: {gpu_memory:.1f}GB\n注意: 显存极低，训练速度会较慢")
            
            self.update_ui_from_config()
            
        except Exception as e:
            messagebox.showerror("错误", f"低显存配置加载失败: {e}")
            
    def update_ui_from_config(self):
        """从配置更新UI"""
        # 更新数据路径
        self.train_lr_var.set(self.config['data']['train_lr_dir'])
        self.train_hr_var.set(self.config['data']['train_hr_dir'])
        self.val_lr_var.set(self.config['data']['val_lr_dir'])
        self.val_hr_var.set(self.config['data']['val_hr_dir'])
        
        # 更新训练参数
        self.epochs_var.set(str(self.config['training']['num_epochs']))
        self.batch_size_var.set(str(self.config['training']['batch_size']))
        self.lr_var.set(str(self.config['training']['learning_rate']))
        self.device_var.set(self.config['training']['device'])
        self.grad_accum_var.set(str(self.config['training'].get('gradient_accumulation_steps', 1)))
        self.mixed_precision_var.set(self.config['training'].get('mixed_precision', False))
        
        # 更新缓存设置
        self.cache_size_var.set(str(self.config['data'].get('max_cache_size', 100)))
        
    def update_config_from_ui(self):
        """从UI更新配置"""
        # 更新数据路径
        self.config['data']['train_lr_dir'] = self.train_lr_var.get()
        self.config['data']['train_hr_dir'] = self.train_hr_var.get()
        self.config['data']['val_lr_dir'] = self.val_lr_var.get()
        self.config['data']['val_hr_dir'] = self.val_hr_var.get()
        
        # 更新训练参数
        self.config['training']['num_epochs'] = int(self.epochs_var.get())
        self.config['training']['batch_size'] = int(self.batch_size_var.get())
        self.config['training']['learning_rate'] = float(self.lr_var.get())
        self.config['training']['device'] = self.device_var.get()
        self.config['training']['gradient_accumulation_steps'] = int(self.grad_accum_var.get())
        self.config['training']['mixed_precision'] = self.mixed_precision_var.get()
        
        # 更新缓存设置
        self.config['training']['max_cache_size'] = int(self.cache_size_var.get())
        
    def start_training(self):
        """开始训练"""
        try:
            # 更保守的PyTorch内存配置
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:16,expandable_segments:True'
            
            # 先从UI更新配置，保存用户设置
            self.update_config_from_ui()
            
            # 保存用户设置的epoch数
            user_epochs = self.config['training']['num_epochs']
            
            # 强制清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # 检查可用显存
                available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"可用显存: {available_memory:.1f} GB")
                
                # 自动检测并应用低显存模式，但保留用户的epoch设置
                if available_memory <= 6.0:  # 6GB以下自动启用低显存模式
                    print("检测到低显存GPU，自动应用低显存配置...")
                    low_memory_config = load_low_memory_config()
                    # 只更新内存相关设置，保留用户的epoch设置
                    self.config['training'].update({
                        'batch_size': low_memory_config['training']['batch_size'],
                        'gradient_accumulation_steps': low_memory_config['training']['gradient_accumulation_steps'],
                        'mixed_precision': low_memory_config['training']['mixed_precision'],
                        'memory_efficient': low_memory_config['training']['memory_efficient'],
                        'max_cache_size': low_memory_config['training']['max_cache_size'],
                        'preload_batch_size': low_memory_config['training']['preload_batch_size'],
                        'memory_cleanup_frequency': low_memory_config['training']['memory_cleanup_frequency'],
                        'checkpoint_memory_cleanup': low_memory_config['training']['checkpoint_memory_cleanup']
                    })
                    self.config['model'].update(low_memory_config['model'])
                    self.config['memory'].update(low_memory_config['memory'])
                    # 恢复用户设置的epoch数
                    self.config['training']['num_epochs'] = user_epochs
                    self.update_ui_from_config()
                    self.log_message("自动应用低显存模式配置（保留用户epoch设置）")
                elif available_memory <= 8.0:  # 8GB以下应用推荐设置
                    print("应用推荐设置...")
                    recommended_config = get_optimal_config_for_gpu()
                    # 只更新内存相关设置，保留用户的epoch设置
                    self.config['training'].update(recommended_config)
                    self.config['training']['num_epochs'] = user_epochs  # 恢复用户设置的epoch数
                    self.update_ui_from_config()
                    self.log_message("应用推荐设置（保留用户epoch设置）")
            
            # 更新按钮状态
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.NORMAL)
            
            # 创建训练管理器 - 修复：只传递config参数
            self.train_manager = TrainingManager(config=self.config)
            
            # 添加回调函数 - 修复：使用add_callback方法
            self.train_manager.add_callback('on_training_start', self.on_training_start)
            self.train_manager.add_callback('on_training_end', self.on_training_end)
            self.train_manager.add_callback('on_epoch_start', self.on_epoch_start)
            self.train_manager.add_callback('on_epoch_end', self.on_epoch_end)
            self.train_manager.add_callback('on_progress_update', self.on_progress_update)
            self.train_manager.add_callback('on_error', self.on_error)
            self.train_manager.add_callback('on_memory_status', self.on_memory_update)
            
            # 开始训练
            self.train_manager.start_training()
            self.log_message(f"训练已开始 - 将训练 {user_epochs} 个epoch")
            
        except Exception as e:
            self.log_message(f"启动训练失败: {str(e)}")
            messagebox.showerror("错误", f"启动训练失败: {str(e)}")
            
            # 恢复按钮状态
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.DISABLED)
            
    def stop_training(self):
        """停止训练"""
        if self.train_manager:
            self.train_manager.stop_training()
            self.log_message("训练已停止")
        
        # 恢复按钮状态
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.DISABLED)
        
        # 重置进度
        self.progress['value'] = 0
        self.progress_label.config(text="训练已停止")
        
    def pause_training(self):
        """暂停训练"""
        if self.train_manager:
            # 注意：训练管理器可能没有pause_training方法，这里只是停止
            self.train_manager.stop_training()
            self.log_message("训练已暂停")
            
    def log_message(self, message):
        """记录日志消息"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        if hasattr(self, 'log_text'):
            self.log_text.insert(tk.END, log_entry)
            self.log_text.see(tk.END)
        
        print(log_entry.strip())
        
    def clear_log(self):
        """清空日志"""
        if hasattr(self, 'log_text'):
            self.log_text.delete(1.0, tk.END)
            
    def save_log(self):
        """保存日志"""
        if hasattr(self, 'log_text'):
            file_path = filedialog.asksaveasfilename(
                title="保存日志文件",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if file_path:
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(self.log_text.get(1.0, tk.END))
                    self.log_message(f"日志已保存到 {file_path}")
                except Exception as e:
                    messagebox.showerror("错误", f"保存日志失败: {str(e)}")
                    
    # 训练回调函数 - 修复：调整参数以匹配训练管理器的调用方式
    def on_training_start(self):
        """训练开始回调"""
        self.status_label.config(text="训练中...")
        
    def on_training_end(self, training_history):
        """训练结束回调"""
        self.status_label.config(text="训练完成")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.DISABLED)
        self.log_message("训练已完成")
        
    def on_epoch_start(self, epoch):
        """Epoch开始回调"""
        total_epochs = self.config['training']['num_epochs']
        self.current_epoch_label.config(text=f"当前Epoch: {epoch + 1}/{total_epochs}")
        
    def on_epoch_end(self, epoch_info):
        """Epoch结束回调"""
        epoch = epoch_info['epoch']
        total_epochs = epoch_info['total_epochs']
        g_loss = epoch_info['g_loss']
        val_psnr = epoch_info['val_psnr']
        is_best = epoch_info['is_best']
        
        # 更新进度
        progress = (epoch / total_epochs) * 100
        self.progress['value'] = progress
        self.progress_label.config(text=f"Epoch {epoch}/{total_epochs} 完成")
        
        # 更新损失和PSNR显示
        self.current_loss_label.config(text=f"生成器损失: {g_loss:.4f}")
        self.current_psnr_label.config(text=f"验证PSNR: {val_psnr:.2f} dB")
        
        # 更新最佳PSNR
        if is_best:
            self.best_psnr_label.config(text=f"最佳PSNR: {val_psnr:.2f} dB")
            
        # 记录日志
        self.log_message(f"Epoch {epoch}/{total_epochs} - 生成器损失: {g_loss:.4f}, 验证PSNR: {val_psnr:.2f} dB")
        if is_best:
            self.log_message(f"新的最佳PSNR: {val_psnr:.2f} dB")
            
    def on_progress_update(self, progress_percentage):
        """进度更新回调"""
        self.progress['value'] = progress_percentage
        
    def on_error(self, error_message):
        """错误回调"""
        self.log_message(f"错误: {error_message}")
        messagebox.showerror("训练错误", error_message)
        
        # 恢复按钮状态
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.DISABLED)
        
    def on_memory_update(self, memory_info):
        """内存更新回调"""
        if 'gpu_allocated' in memory_info:
            gpu_used = memory_info['gpu_allocated']
            gpu_total = self.gpu_info['memory'] if self.gpu_info['available'] else 0
            self.memory_label.config(text=f"GPU内存: {gpu_used:.1f}/{gpu_total:.1f} GB")
    
    def load_pretrained_model(self):
        """加载预训练模型"""
        file_path = filedialog.askopenfilename(
            title="选择预训练模型文件",
            filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                # 验证模型文件
                checkpoint = torch.load(file_path, map_location='cuda')
                if 'generator_state_dict' not in checkpoint:
                    messagebox.showerror("错误", "无效的模型文件：缺少生成器状态字典")
                    return
                
                self.pretrained_model_path = file_path
                self.log_message(f"已加载预训练模型: {os.path.basename(file_path)}")
                messagebox.showinfo("成功", f"预训练模型加载成功:\n{os.path.basename(file_path)}")
                
                # 启用增量训练按钮
                self.incremental_button.config(state=tk.NORMAL)
                
            except Exception as e:
                self.log_message(f"加载预训练模型失败: {str(e)}")
                messagebox.showerror("错误", f"加载预训练模型失败:\n{str(e)}")
    
    def start_incremental_training(self):
        """开始增量训练"""
        try:
            # 检查是否已加载预训练模型
            if not hasattr(self, 'pretrained_model_path') or not self.pretrained_model_path:
                messagebox.showwarning("警告", "请先加载预训练模型")
                return
            
            # 检查数据路径
            required_paths = [
                self.train_lr_var.get(),
                self.train_hr_var.get(),
                self.val_lr_var.get(),
                self.val_hr_var.get()
            ]
            
            for path in required_paths:
                if not path or not os.path.exists(path):
                    messagebox.showerror("错误", f"数据路径不存在: {path}")
                    return
            
            # 更新配置
            self.update_config_from_ui()
            
            # 保存用户设置的epoch数
            user_epochs = self.config['training']['num_epochs']
            
            # 强制清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # 更新按钮状态
            self.start_button.config(state=tk.DISABLED)
            self.incremental_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.NORMAL)
            
            # 创建训练管理器
            self.train_manager = TrainingManager(config=self.config)
            
            # 添加回调函数
            self.train_manager.add_callback('on_training_start', self.on_training_start)
            self.train_manager.add_callback('on_training_end', self.on_incremental_training_end)
            self.train_manager.add_callback('on_epoch_start', self.on_epoch_start)
            self.train_manager.add_callback('on_epoch_end', self.on_epoch_end)
            self.train_manager.add_callback('on_progress_update', self.on_progress_update)
            self.train_manager.add_callback('on_error', self.on_error)
            self.train_manager.add_callback('on_memory_status', self.on_memory_update)
            
            # 准备训练
            if self.train_manager.prepare_training():
                # 读取检查点信息以显示当前状态
                checkpoint = torch.load(self.pretrained_model_path, map_location=self.config['training']['device'])
                current_epoch = checkpoint.get('epoch', 0)
                best_psnr = checkpoint.get('best_psnr', 0)
                
                self.log_message(f"检查点信息: 当前epoch={current_epoch}, 最佳PSNR={best_psnr:.2f}")
                self.log_message(f"开始增量训练 - 将从第{current_epoch + 1}个epoch继续训练{user_epochs}个epoch")
                
                # 使用增量训练方法，传入检查点路径
                success = self.train_manager.start_incremental_training(
                    checkpoint_path=self.pretrained_model_path
                )
                
                if not success:
                    raise Exception("增量训练启动失败")
                    
            else:
                raise Exception("训练准备失败")
            
        except Exception as e:
            self.log_message(f"启动增量训练失败: {str(e)}")
            messagebox.showerror("错误", f"启动增量训练失败: {str(e)}")
            
            # 恢复按钮状态
            self.start_button.config(state=tk.NORMAL)
            self.incremental_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.DISABLED)
    
    def on_incremental_training_end(self, training_history):
        """增量训练结束回调"""
        self.status_label.config(text="增量训练完成")
        self.start_button.config(state=tk.NORMAL)
        self.incremental_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.DISABLED)
        self.log_message("增量训练已完成")
            
    def run(self):
        """运行GUI"""
        self.root.mainloop()

def main():
    """主函数"""
    app = TrainingGUI()
    app.run()

if __name__ == "__main__":
    main()


    def create_advanced_settings_frame(self, parent):
        """创建高级设置框架"""
        advanced_frame = ttk.LabelFrame(parent, text="图像自动调整设置", padding="10")
        advanced_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 自动调整开关
        self.auto_resize_var = tk.BooleanVar(value=True)
        auto_resize_check = ttk.Checkbutton(advanced_frame, text="启用图像自动调整", 
                                          variable=self.auto_resize_var)
        auto_resize_check.pack(anchor=tk.W, pady=(0, 5))
        
        # 缩放倍数设置
        scale_frame = ttk.Frame(advanced_frame)
        scale_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(scale_frame, text="HR/LR缩放倍数:").pack(side=tk.LEFT)
        self.target_scale_var = tk.StringVar(value="2")
        scale_entry = ttk.Entry(scale_frame, textvariable=self.target_scale_var, width=10)
        scale_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # 最大LR尺寸设置
        max_size_frame = ttk.Frame(advanced_frame)
        max_size_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(max_size_frame, text="LR图像最大尺寸:").pack(side=tk.LEFT)
        self.max_lr_size_var = tk.StringVar(value="512")
        max_size_entry = ttk.Entry(max_size_frame, textvariable=self.max_lr_size_var, width=10)
        max_size_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # 尺寸倍数强制设置
        divisible_frame = ttk.Frame(advanced_frame)
        divisible_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(divisible_frame, text="尺寸必须为此数的倍数:").pack(side=tk.LEFT)
        self.enforce_divisible_var = tk.StringVar(value="8")
        divisible_entry = ttk.Entry(divisible_frame, textvariable=self.enforce_divisible_var, width=10)
        divisible_entry.pack(side=tk.LEFT, padx=(5, 0))