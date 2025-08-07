import torch
import numpy as np
from PIL import Image

def calculate_psnr(img1, img2):
    """计算PSNR"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_ssim(img1, img2):
    """计算SSIM (简化版本)"""
    try:
        from skimage.metrics import structural_similarity as ssim
        import numpy as np
        
        # 转换为灰度图
        if isinstance(img1, Image.Image):
            arr1 = np.array(img1.convert('L'), dtype=np.float32)
            arr2 = np.array(img2.convert('L'), dtype=np.float32)
        else:
            # 假设是tensor - 移动到CPU进行计算
            arr1 = img1.detach().cpu().numpy()
            arr2 = img2.detach().cpu().numpy()
        
        return ssim(arr1, arr2, data_range=255)
    except ImportError:
        # 如果没有skimage，返回简化的相似度指标
        return calculate_psnr(img1, img2).item() / 50.0  # 简化映射

def batch_psnr(sr_imgs, hr_imgs):
    """批量计算PSNR"""
    psnr_values = []
    for sr, hr in zip(sr_imgs, hr_imgs):
        psnr = calculate_psnr(sr, hr)
        psnr_values.append(psnr.item())
    return np.mean(psnr_values)
