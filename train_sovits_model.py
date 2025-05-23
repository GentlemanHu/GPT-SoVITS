#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 西班牙语和葡萄牙语SoVITS模型训练脚本
"""

import os
import sys
import argparse
import logging
import json
import torch
import shutil
import numpy as np
import random
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 添加GPT-SoVITS路径到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'GPT_SoVITS'))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('train_sovits_model')

def load_json_config(config_file):
    """
    加载JSON配置文件
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    """
    保存检查点
    """
    logger.info(f"保存模型和优化器状态到 {filepath}")
    torch.save({
        "iteration": iteration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "learning_rate": learning_rate
    }, filepath)

def scan_checkpoint(cp_dir, prefix):
    """
    扫描检查点目录
    """
    pattern = os.path.join(cp_dir, prefix + "????????")
    cp_list = sorted(glob.glob(pattern))
    if len(cp_list) == 0:
        return None
    return cp_list[-1]

def set_seed(seed):
    """
    设置随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser(description='训练GPT-SoVITS的SoVITS模型')
    parser.add_argument('--language', type=str, required=True,
                        choices=['es', 'pt'],
                        help='语言 (es: 西班牙语, pt: 葡萄牙语)')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='数据目录，包含预处理后的数据')
    parser.add_argument('--config', type=str, default='GPT_SoVITS/configs/s2.json',
                        help='配置文件路径 (默认: GPT_SoVITS/configs/s2.json)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批处理大小 (默认: 16)')
    parser.add_argument('--max-epochs', type=int, default=100,
                        help='最大训练轮数 (默认: 100)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备 (默认: cuda 如果可用，否则 cpu)')
    parser.add_argument('--dialect', type=str, default=None,
                        help='方言 (brazilian, european, latin_american)')
    
    args = parser.parse_args()
    
    # 设置路径
    output_dir = Path(f"checkpoints/{args.language}_sovits")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    config = load_json_config(args.config)
    
    # 修改配置
    config['train']['batch_size'] = args.batch_size
    config['train']['epochs'] = args.max_epochs
    
    # 设置随机种子
    set_seed(config['train']['seed'])
    
    # 导入GPT-SoVITS模块
    try:
        import GPT_SoVITS.utils as utils
        from GPT_SoVITS.module.models import SynthesizerTrn
        from GPT_SoVITS.module.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
        from GPT_SoVITS.module.data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
    except ImportError:
        try:
            import utils
            from module.models import SynthesizerTrn
            from module.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
            from module.data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
        except ImportError:
            logger.error("无法导入SoVITS模块，请确保GPT-SoVITS环境配置正确")
            logger.error("请检查以下路径是否存在:")
            logger.error(f"  - {os.path.join(current_dir, 'GPT_SoVITS', 'module')}")
            logger.error(f"  - {os.path.join(current_dir, 'module')}")
            sys.exit(1)
    
    # 创建数据集
    train_dataset = TextAudioSpeakerLoader(
        os.path.join(args.data_dir, '2-name2text.txt'),
        config['data']
    )
    
    # 创建数据加载器
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    # 创建模型
    model = SynthesizerTrn(
        config['data']['filter_length'] // 2 + 1,
        config['train']['segment_size'] // config['data']['hop_length'],
        n_speakers=config['data']['n_speakers'],
        **config['model']
    )
    
    # 移动模型到设备
    model = model.to(args.device)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        config['train']['learning_rate'],
        betas=config['train']['betas'],
        eps=config['train']['eps']
    )
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, 
        gamma=config['train']['lr_decay']
    )
    
    # 创建TensorBoard日志
    writer = SummaryWriter(log_dir=output_dir)
    
    # 训练参数
    epochs = config['train']['epochs']
    iterations = 0
    start_epoch = 0
    
    # 查找检查点
    checkpoint_path = output_dir / "checkpoint.pth"
    if checkpoint_path.exists():
        logger.info(f"从检查点恢复训练: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        iterations = checkpoint["iteration"]
        start_epoch = iterations // len(train_loader)
        scheduler.last_epoch = iterations
    
    # 开始训练
    logger.info("开始训练SoVITS模型...")
    model.train()
    
    for epoch in range(start_epoch, epochs):
        for batch_idx, batch in enumerate(train_loader):
            # 移动数据到设备
            x, x_lengths, spec, spec_lengths, y, y_lengths, speakers = [
                x.to(args.device) if isinstance(x, torch.Tensor) else x
                for x in batch
            ]
            
            # 前向传播
            y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q) = model(
                x, x_lengths, spec, spec_lengths, speakers
            )
            
            # 计算损失
            mel = spec_to_mel_torch(
                spec,
                config['data']['filter_length'],
                config['data']['n_mel_channels'],
                config['data']['sampling_rate'],
                config['data']['mel_fmin'],
                config['data']['mel_fmax']
            )
            
            y_mel = commons.slice_segments(
                mel, ids_slice, config['train']['segment_size'] // config['data']['hop_length']
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                config['data']['filter_length'],
                config['data']['n_mel_channels'],
                config['data']['sampling_rate'],
                config['data']['hop_length'],
                config['data']['win_length'],
                config['data']['mel_fmin'],
                config['data']['mel_fmax']
            )
            
            # 计算损失
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * config['train']['c_mel']
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config['train']['c_kl']
            
            loss = loss_mel + loss_kl
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # 更新迭代计数
            iterations += 1
            
            # 记录日志
            if iterations % config['train']['log_interval'] == 0:
                logger.info(
                    f"Epoch: {epoch}, Iteration: {iterations}, Loss: {loss.item():.4f}, "
                    f"Mel Loss: {loss_mel.item():.4f}, KL Loss: {loss_kl.item():.4f}, "
                    f"LR: {scheduler.get_last_lr()[0]:.7f}"
                )
                
                # 写入TensorBoard
                writer.add_scalar("Loss/total", loss.item(), iterations)
                writer.add_scalar("Loss/mel", loss_mel.item(), iterations)
                writer.add_scalar("Loss/kl", loss_kl.item(), iterations)
                writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], iterations)
            
            # 保存检查点
            if iterations % config['train']['eval_interval'] == 0:
                save_checkpoint(
                    model, optimizer, scheduler.get_last_lr()[0], iterations,
                    checkpoint_path
                )
                
                # 保存带有迭代次数的检查点
                save_checkpoint(
                    model, optimizer, scheduler.get_last_lr()[0], iterations,
                    output_dir / f"checkpoint_{iterations:08d}.pth"
                )
    
    # 保存最终模型
    save_checkpoint(
        model, optimizer, scheduler.get_last_lr()[0], iterations,
        output_dir / "best_model.pth"
    )
    
    logger.info("SoVITS模型训练完成")

if __name__ == '__main__':
    main()
