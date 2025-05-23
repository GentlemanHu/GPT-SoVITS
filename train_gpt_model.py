#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 西班牙语和葡萄牙语GPT模型训练脚本
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import shutil
from pathlib import Path
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

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
logger = logging.getLogger('train_gpt_model')

def load_yaml_config(config_file):
    """
    加载YAML配置文件
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_newest_ckpt(ckpt_dir):
    """
    获取最新的检查点文件
    """
    import re
    
    # 获取所有检查点文件
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
    
    if not ckpt_files:
        return None
    
    # 提取步数
    def extract_step(filename):
        match = re.search(r'step=(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0
    
    # 按步数排序
    ckpt_files.sort(key=extract_step, reverse=True)
    
    return ckpt_files[0]

def my_model_ckpt(config, if_save_latest, if_save_every_weights, half_weights_save_dir, exp_name, **kwargs):
    """
    创建模型检查点回调
    """
    callbacks = []
    
    # 创建检查点回调
    ckpt_callback = ModelCheckpoint(**kwargs)
    callbacks.append(ckpt_callback)
    
    return ckpt_callback

def main():
    parser = argparse.ArgumentParser(description='训练GPT-SoVITS的GPT模型')
    parser.add_argument('--language', type=str, required=True,
                        choices=['es', 'pt'],
                        help='语言 (es: 西班牙语, pt: 葡萄牙语)')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='数据目录，包含预处理后的数据')
    parser.add_argument('--config', type=str, default='GPT_SoVITS/configs/s1longer.yaml',
                        help='配置文件路径 (默认: GPT_SoVITS/configs/s1longer.yaml)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批处理大小 (默认: 16)')
    parser.add_argument('--max-epochs', type=int, default=100,
                        help='最大训练轮数 (默认: 100)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备 (默认: cuda 如果可用，否则 cpu)')
    parser.add_argument('--dialect', type=str, default=None,
                        help='方言 (brazilian, european, latin_american)')
    parser.add_argument('--bert-model', type=str, default=None,
                        help='BERT模型路径 (默认: None)')
    
    args = parser.parse_args()
    
    # 设置路径
    output_dir = Path(f"checkpoints/{args.language}_gpt")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    config = load_yaml_config(args.config)
    
    # 修改配置
    config['train']['batch_size'] = args.batch_size
    config['train']['epochs'] = args.max_epochs
    config['train_semantic_path'] = os.path.join(args.data_dir, '6-name2semantic.tsv')
    config['train_phoneme_path'] = os.path.join(args.data_dir, '2-name2text.txt')
    config['output_dir'] = str(output_dir)
    
    # 设置BERT模型
    if args.bert_model:
        config['bert_path'] = args.bert_model
    elif args.language == 'es':
        config['bert_path'] = 'pretrained_models/spanish-bert'
    elif args.language == 'pt':
        config['bert_path'] = 'pretrained_models/portuguese-bert'
    
    # 设置随机种子
    seed_everything(config['train']['seed'], workers=True)
    
    # 创建检查点回调
    ckpt_callback = my_model_ckpt(
        config=config,
        if_save_latest=config['train'].get('if_save_latest', True),
        if_save_every_weights=config['train'].get('if_save_every_weights', False),
        half_weights_save_dir=config['train'].get('half_weights_save_dir', ''),
        exp_name=config['train'].get('exp_name', 'default'),
        save_top_k=-1,
        monitor="top_3_acc",
        mode="max",
        save_on_train_epoch_end=True,
        every_n_epochs=config['train'].get('save_every_n_epoch', 10),
        dirpath=ckpt_dir,
    )
    
    # 创建TensorBoard日志
    logger_tb = TensorBoardLogger(name=output_dir.stem, save_dir=output_dir)
    
    # 设置环境变量
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["USE_LIBUV"] = "0"
    
    # 导入GPT-SoVITS模块
    try:
        from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
        from GPT_SoVITS.AR.data.data_module import Text2SemanticDataModule
    except ImportError:
        try:
            from AR.models.t2s_lightning_module import Text2SemanticLightningModule
            from AR.data.data_module import Text2SemanticDataModule
        except ImportError:
            logger.error("无法导入AR模块，请确保GPT-SoVITS环境配置正确")
            logger.error("请检查以下路径是否存在:")
            logger.error(f"  - {os.path.join(current_dir, 'GPT_SoVITS', 'AR')}")
            logger.error(f"  - {os.path.join(current_dir, 'AR')}")
            sys.exit(1)
    
    # 创建训练器
    trainer = Trainer(
        max_epochs=config['train']['epochs'],
        accelerator="gpu" if args.device == "cuda" else "cpu",
        limit_val_batches=0,
        devices=-1 if args.device == "cuda" else 1,
        benchmark=False,
        fast_dev_run=False,
        strategy=DDPStrategy(process_group_backend="nccl" if os.name != 'nt' else "gloo")
        if args.device == "cuda"
        else "auto",
        precision=config['train']['precision'],
        logger=logger_tb,
        num_sanity_val_steps=0,
        callbacks=[ckpt_callback],
        use_distributed_sampler=False,
    )
    
    # 创建模型
    model = Text2SemanticLightningModule(config, output_dir)
    
    # 创建数据模块
    data_module = Text2SemanticDataModule(
        config,
        train_semantic_path=config['train_semantic_path'],
        train_phoneme_path=config['train_phoneme_path'],
    )
    
    # 查找最新的检查点
    try:
        newest_ckpt_name = get_newest_ckpt(ckpt_dir)
        if newest_ckpt_name:
            ckpt_path = ckpt_dir / newest_ckpt_name
            logger.info(f"从检查点恢复训练: {ckpt_path}")
        else:
            ckpt_path = None
    except Exception as e:
        logger.warning(f"查找检查点失败: {e}")
        ckpt_path = None
    
    # 开始训练
    logger.info("开始训练GPT模型...")
    trainer.fit(model, data_module, ckpt_path=ckpt_path)
    
    # 复制最佳模型
    try:
        newest_ckpt_name = get_newest_ckpt(ckpt_dir)
        if newest_ckpt_name:
            src_path = ckpt_dir / newest_ckpt_name
            dst_path = output_dir / "best_model.pth"
            shutil.copy2(src_path, dst_path)
            logger.info(f"已将最佳模型复制到 {dst_path}")
    except Exception as e:
        logger.warning(f"复制最佳模型失败: {e}")
    
    logger.info("GPT模型训练完成")

if __name__ == '__main__':
    main()
