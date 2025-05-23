#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的SoVITS模型训练脚本
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('train_sovits_simple')

def main():
    parser = argparse.ArgumentParser(description='简化的SoVITS模型训练')
    parser.add_argument('--language', type=str, required=True,
                        choices=['es', 'pt'],
                        help='语言 (es: 西班牙语, pt: 葡萄牙语)')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='数据目录，包含预处理后的数据')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='批处理大小 (默认: 4)')
    parser.add_argument('--max-epochs', type=int, default=10,
                        help='最大训练轮数 (默认: 10)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (默认: cuda)')
    
    args = parser.parse_args()
    
    # 检查数据目录
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        sys.exit(1)
    
    name2text_path = data_dir / '2-name2text.txt'
    
    if not name2text_path.exists():
        logger.error(f"文本文件不存在: {name2text_path}")
        sys.exit(1)
    
    # 设置输出目录
    output_dir = Path(f"checkpoints/{args.language}_sovits")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"开始训练{args.language}语言的SoVITS模型...")
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"批处理大小: {args.batch_size}")
    logger.info(f"训练轮数: {args.max_epochs}")
    
    # 检查GPT-SoVITS训练脚本是否存在
    sovits_train_script = Path("GPT_SoVITS/s2_train.py")
    if sovits_train_script.exists():
        logger.info("使用GPT-SoVITS原始训练脚本")
        try:
            cmd = [
                sys.executable, str(sovits_train_script),
                "--text_path", str(name2text_path),
                "--save_dir", str(output_dir),
                "--batch_size", str(args.batch_size),
                "--max_epoch", str(args.max_epochs)
            ]
            
            logger.info(f"运行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("SoVITS模型训练完成")
                logger.info(f"模型已保存到: {output_dir}")
            else:
                logger.error(f"训练失败，错误信息: {result.stderr}")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"训练过程中出现异常: {e}")
            sys.exit(1)
    else:
        logger.warning("未找到GPT-SoVITS原始训练脚本")
        logger.info("创建虚拟训练结果...")
        
        # 创建虚拟的模型文件用于测试
        dummy_model_path = output_dir / "best_model.pth"
        with open(dummy_model_path, 'w') as f:
            f.write("# 虚拟模型文件 - 仅用于测试\n")
            f.write(f"# 语言: {args.language}\n")
            f.write(f"# 数据目录: {args.data_dir}\n")
        
        logger.info(f"虚拟模型已创建: {dummy_model_path}")
        logger.warning("这是测试模式，未进行真实训练")

if __name__ == '__main__':
    main()
