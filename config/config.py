import json
import os
from typing import Dict, Any

def load_config(config_path: str = "config/config.json") -> Dict[str, Any]:
    """加载配置文件"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return get_default_config()

def save_config(config: Dict[str, Any], config_path: str = "config/config.json") -> None:
    """保存配置文件"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        "model": {
            "num_blocks": 8,
            "num_features": 128,  
            "scale": 2
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 3e-4,
            "num_epochs": 5,
            "device": "cuda",
            "mixed_precision": True,
            "save_frequency": 1,
            "validation_frequency": 5,
            "early_stopping_patience": 20
        },
        "data": {
            "train_lr_dir": "data/train/lr",
            "train_hr_dir": "data/train/hr",
            "val_lr_dir": "data/val/lr",
            "val_hr_dir": "data/val/hr",
            "num_workers": 4,
            "pin_memory": True
        },
        "optimizer": {
            "type": "Adam",
            "betas": [0.9, 0.999],
            "weight_decay": 0,
            "scheduler": {
                "type": "MultiStepLR",
                "milestones": [50, 80, 120, 160],
                "gamma": 0.5
            }
        },
        "loss": {
            "l1_weight": 1.0,
            "perceptual_weight": 0.1,
            "gan_weight": 0.01
        },
        "memory": {
            "gradient_accumulation_steps": 1,
            "max_cache_size": 100,
            "enable_checkpointing": False,
            "clear_cache_frequency": 10
        },
        "paths": {
            "save_dir": "checkpoints",
            "checkpoint_dir": "checkpoints",
            "log_dir": "logs",
            "output_dir": "outputs"
        }
    }

def load_low_memory_config() -> Dict[str, Any]:
    """加载低显存GPU配置"""
    config = load_config()
    
    # 低显存GPU设置
    config["training"].update({
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "mixed_precision": True
    })
    
    config["model"].update({
        "num_blocks": 4,
        "num_features": 32
    })
    
    config["data"].update({
        "num_workers": 2,
        "pin_memory": False
    })
    
    config["memory"].update({
        "gradient_accumulation_steps": 16,
        "max_cache_size": 50,
        "enable_checkpointing": True,
        "clear_cache_frequency": 5
    })
    
    config["loss"].update({
        "l1_weight": 1.0,
        "perceptual_weight": 0.05,
        "gan_weight": 0.005
    })
    
    return config

def load_high_memory_config() -> Dict[str, Any]:
    """加载高显存GPU配置"""
    config = load_config()
    
    # 高显存GPU设置
    config["training"].update({
        "batch_size": 8,
        "gradient_accumulation_steps": 1,
        "mixed_precision": True
    })
    
    config["model"].update({
        "num_blocks": 8,
        "num_features": 64
    })
    
    config["data"].update({
        "num_workers": 8,
        "pin_memory": True
    })
    
    config["memory"].update({
        "gradient_accumulation_steps": 1,
        "max_cache_size": 200,
        "enable_checkpointing": False,
        "clear_cache_frequency": 10
    })
    
    return config

def load_rtx_4090_config() -> Dict[str, Any]:
    """加载RTX 4090专用优化配置"""
    config = load_config()
    
    # RTX 4090专用设置
    config["training"].update({
        "batch_size": 12,
        "learning_rate": 1e-4,  # 🔥 降低学习率以适应新特征数
        "mixed_precision": True,
        "gradient_accumulation_steps": 1
    })
    
    config["model"].update({
        "num_blocks": 8,
        "num_features": 72  # 🔥 从64改为72
    })
    
    config["data"].update({
        "num_workers": 12,
        "pin_memory": True
    })
    
    config["memory"].update({
        "gradient_accumulation_steps": 1,
        "max_cache_size": 500,
        "enable_checkpointing": False,
        "clear_cache_frequency": 20
    })
    
    config["loss"].update({
        "l1_weight": 1.0,
        "perceptual_weight": 0.1,
        "gan_weight": 0.01
    })
    
    # RTX 4090专用优化
    config["rtx_4090"] = {
        "tf32_enabled": True,
        "compile_enabled": True,
        "auto_batch_size": True,
        "max_batch_size": 16,
        "memory_fraction": 0.95,
        "optimization_level": "O2"
    }
    
    return config

def create_4gb_config() -> Dict[str, Any]:
    """创建4GB显存专用配置"""
    config = get_default_config()
    
    # 4GB显存极限优化
    config["training"].update({
        "batch_size": 1,
        "learning_rate": 5e-5,
        "mixed_precision": True,
        "gradient_accumulation_steps": 16
    })
    
    config["model"].update({
        "num_blocks": 3,
        "num_features": 24
    })
    
    config["data"].update({
        "num_workers": 1,
        "pin_memory": False
    })
    
    config["memory"].update({
        "gradient_accumulation_steps": 32,
        "max_cache_size": 25,
        "enable_checkpointing": True,
        "clear_cache_frequency": 3
    })
    
    config["loss"].update({
        "l1_weight": 1.0,
        "perceptual_weight": 0.02,
        "gan_weight": 0.002
    })
    
    return config

def create_6gb_config() -> Dict[str, Any]:
    """创建6GB显存专用配置"""
    config = get_default_config()
    
    # 6GB显存优化
    config["training"].update({
        "batch_size": 2,
        "learning_rate": 1e-4,
        "mixed_precision": True,
        "gradient_accumulation_steps": 4
    })
    
    config["model"].update({
        "num_blocks": 4,
        "num_features": 32
    })
    
    config["data"].update({
        "num_workers": 2,
        "pin_memory": True
    })
    
    config["memory"].update({
        "gradient_accumulation_steps": 8,
        "max_cache_size": 50,
        "enable_checkpointing": True,
        "clear_cache_frequency": 5
    })
    
    return config

def create_8gb_config() -> Dict[str, Any]:
    """创建8GB显存专用配置"""
    config = get_default_config()
    
    # 8GB显存优化
    config["training"].update({
        "batch_size": 4,
        "learning_rate": 1e-4,
        "mixed_precision": True,
        "gradient_accumulation_steps": 2
    })
    
    config["model"].update({
        "num_blocks": 6,
        "num_features": 48
    })
    
    config["data"].update({
        "num_workers": 4,
        "pin_memory": True
    })
    
    config["memory"].update({
        "gradient_accumulation_steps": 4,
        "max_cache_size": 100,
        "enable_checkpointing": False,
        "clear_cache_frequency": 8
    })
    
    return config

def create_12gb_config() -> Dict[str, Any]:
    """创建12GB显存专用配置"""
    config = get_default_config()
    
    # 12GB显存优化
    config["training"].update({
        "batch_size": 6,
        "learning_rate": 1e-4,
        "mixed_precision": True,
        "gradient_accumulation_steps": 1
    })
    
    config["model"].update({
        "num_blocks": 6,
        "num_features": 64
    })
    
    config["data"].update({
        "num_workers": 6,
        "pin_memory": True
    })
    
    config["memory"].update({
        "gradient_accumulation_steps": 2,
        "max_cache_size": 150,
        "enable_checkpointing": False,
        "clear_cache_frequency": 12
    })
    
    return config

def auto_config_by_gpu_memory(gpu_memory_gb: float) -> Dict[str, Any]:
    """根据GPU显存自动选择配置"""
    if gpu_memory_gb < 4.5:
        return create_4gb_config()
    elif gpu_memory_gb < 6.5:
        return create_6gb_config()
    elif gpu_memory_gb < 8.5:
        return create_8gb_config()
    elif gpu_memory_gb < 12.5:
        return create_12gb_config()
    elif gpu_memory_gb >= 20:  # RTX 4090等高端卡
        return load_rtx_4090_config()
    else:
        return load_high_memory_config()