#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI超分辨率工具集 - 主启动器
"""

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import os

class MainLauncher:
    """主启动器"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI超分辨率工具集")
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        
        # 设置窗口居中
        self.center_window()
        
        # 创建界面
        self.create_widgets()
    
    def center_window(self):
        """窗口居中"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(main_frame, text="AI超分辨率工具集", 
                               font=("Arial", 18, "bold"))
        title_label.pack(pady=(0, 30))
        
        # 工具按钮区域
        tools_frame = ttk.LabelFrame(main_frame, text="选择工具", padding="20")
        tools_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # 视频分帧工具
        video_btn = ttk.Button(tools_frame, text="视频分帧工具", 
                              command=self.launch_video_extractor,
                              width=30)
        video_btn.pack(pady=(0, 10))
        
        video_desc = ttk.Label(tools_frame, 
                              text="从高分辨率和低分辨率视频中提取对齐的训练帧",
                              font=("Arial", 9), foreground="gray")
        video_desc.pack(pady=(0, 20))
        
        # 训练工具
        train_btn = ttk.Button(tools_frame, text="模型训练工具", 
                              command=self.launch_training,
                              width=30)
        train_btn.pack(pady=(0, 10))
        
        train_desc = ttk.Label(tools_frame, 
                              text="训练超分辨率模型，支持实时监控和参数调整",
                              font=("Arial", 9), foreground="gray")
        train_desc.pack(pady=(0, 20))
        
        # 验证工具
        val_btn = ttk.Button(tools_frame, text="模型验证工具", 
                            command=self.launch_validation,
                            width=30)
        val_btn.pack(pady=(0, 10))
        
        val_desc = ttk.Label(tools_frame, 
                            text="验证模型性能，查看超分辨率效果和指标",
                            font=("Arial", 9), foreground="gray")
        val_desc.pack(pady=(0, 20))
        
        # 底部按钮
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X)
        
        # 帮助按钮
        help_btn = ttk.Button(bottom_frame, text="帮助", command=self.show_help)
        help_btn.pack(side=tk.LEFT)
        
        # 退出按钮
        exit_btn = ttk.Button(bottom_frame, text="退出", command=self.root.quit)
        exit_btn.pack(side=tk.RIGHT)
    
    def launch_video_extractor(self):
        """启动视频分帧工具"""
        try:
            script_path = os.path.join(os.path.dirname(__file__), "video_frame_extractor.py")
            subprocess.Popen([sys.executable, script_path])
        except Exception as e:
            messagebox.showerror("错误", f"启动视频分帧工具失败: {str(e)}")
    
    def launch_training(self):
        """启动训练工具"""
        try:
            script_path = os.path.join(os.path.dirname(__file__), "train_gui.py")
            subprocess.Popen([sys.executable, script_path])
        except Exception as e:
            messagebox.showerror("错误", f"启动训练工具失败: {str(e)}")
    
    def launch_validation(self):
        """启动验证工具"""
        try:
            script_path = os.path.join(os.path.dirname(__file__), "validation_gui.py")
            subprocess.Popen([sys.executable, script_path])
        except Exception as e:
            messagebox.showerror("错误", f"启动验证工具失败: {str(e)}")
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """AI超分辨率工具集使用说明:

1. 视频分帧工具:
   - 选择高分辨率和低分辨率视频
   - 设置分帧参数（间隔、最大帧数等）
   - 自动生成训练和验证数据集

2. 模型训练工具:
   - 配置训练参数
   - 实时监控训练进度
   - 查看训练曲线和日志

3. 模型验证工具:
   - 加载训练好的模型
   - 验证模型性能
   - 查看超分辨率效果

使用流程:
视频分帧 → 模型训练 → 模型验证

注意事项:
- 确保安装了所需的依赖包
- 视频文件格式支持: mp4, avi, mov, mkv等
- 建议使用GPU进行训练以提高速度"""
        
        messagebox.showinfo("帮助", help_text)
    
    def run(self):
        """运行应用"""
        self.root.mainloop()

def main():
    """主函数"""
    app = MainLauncher()
    app.run()

if __name__ == "__main__":
    main()