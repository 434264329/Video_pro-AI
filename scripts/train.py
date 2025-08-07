#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from src.training.trainer import Trainer
from config.config import load_config

def main():
    parser = argparse.ArgumentParser(description='AI 图像超分辨率模型训练')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--create-config', action='store_true', help='创建示例配置文件')
    
    args = parser.parse_args()
    
    if args.create_config:
        config = load_config()
        config_path = "training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"✅ 示例配置文件已创建: {config_path}")
        return
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建训练器
    trainer = Trainer(
        train_lr_dir=config['data']['train_lr_dir'],
        train_hr_dir=config['data']['train_hr_dir'],
        val_lr_dir=config['data']['val_lr_dir'],
        val_hr_dir=config['data']['val_hr_dir'],
        batch_size=config['training']['batch_size'],
        lr=config['training']['learning_rate'],
        device=config['training']['device']
    )
    
    # 开始训练
    trainer.train(config['training']['num_epochs'], config['paths']['save_dir'])

if __name__ == "__main__":
    main()
