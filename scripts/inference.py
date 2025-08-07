#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from src.inference.predictor import ImageSuperResolution

def main():
    # 强制检查CUDA可用性
    if not torch.cuda.is_available():
        print("❌ 错误：此版本仅支持GPU运行，但未检测到CUDA设备！")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description='Super Resolution Inference - GPU专用版')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, help='Output path (optional)')
    
    args = parser.parse_args()
    
    # 创建推理器 - 强制使用CUDA
    sr_processor = ImageSuperResolution(args.model, 'cuda')
    
    # 处理图像
    output_image = sr_processor.process_image(args.input)
    
    # 保存结果
    output_path = args.output or f"{os.path.splitext(args.input)[0]}_2x.png"
    output_image.save(output_path)
    print(f"✅ 结果已保存到: {output_path}")

if __name__ == "__main__":
    main()
