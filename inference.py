#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 西班牙语和葡萄牙语推理脚本
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('inference')

def load_audio(audio_path, sr=24000):
    """
    加载音频文件
    """
    audio, sr_orig = librosa.load(audio_path, sr=sr, mono=True)
    return audio, sr

def main():
    parser = argparse.ArgumentParser(description='GPT-SoVITS 西班牙语和葡萄牙语推理')
    parser.add_argument('--text', type=str, required=True,
                        help='要合成的文本')
    parser.add_argument('--language', type=str, required=True,
                        choices=['es', 'pt'],
                        help='语言 (es: 西班牙语, pt: 葡萄牙语)')
    parser.add_argument('--reference-audio', type=str, required=True,
                        help='参考音频路径')
    parser.add_argument('--prompt-text', type=str, default=None,
                        help='参考音频的文本 (如果不提供，将使用无参考文本模式)')
    parser.add_argument('--gpt-model', type=str, required=True,
                        help='GPT模型路径')
    parser.add_argument('--sovits-model', type=str, required=True,
                        help='SoVITS模型路径')
    parser.add_argument('--output-path', type=str, default='output.wav',
                        help='输出音频路径 (默认: output.wav)')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Top-K采样参数 (默认: 5)')
    parser.add_argument('--top-p', type=float, default=0.7,
                        help='Top-P采样参数 (默认: 0.7)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='温度参数 (默认: 0.7)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='语速 (默认: 1.0)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备 (默认: cuda 如果可用，否则 cpu)')
    
    args = parser.parse_args()
    
    # 导入GPT-SoVITS模块
    sys.path.insert(0, os.getcwd())
    from text.cleaner import clean_text
    from feature_extractor import cnhubert
    from AR.models.t2s_lightning_module import Text2SemanticLightningModule
    from module.models import SynthesizerTrn
    
    # 设置设备
    device = torch.device(args.device)
    
    # 加载HuBERT模型
    logger.info("加载HuBERT模型...")
    hubert_model = cnhubert.CNHubert(base_path="GPT_SoVITS/pretrained_models/chinese-hubert-base")
    hubert_model = hubert_model.to(device)
    hubert_model.eval()
    
    # 加载GPT模型
    logger.info(f"加载GPT模型: {args.gpt_model}")
    t2s_model = Text2SemanticLightningModule.load_from_checkpoint(
        args.gpt_model,
        map_location=device
    )
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    
    # 加载SoVITS模型
    logger.info(f"加载SoVITS模型: {args.sovits_model}")
    vits_model = SynthesizerTrn(
        spec_channels=1025,
        segment_size=32,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 8, 2, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 8, 2, 2],
        n_speakers=0,
        gin_channels=0,
        use_sdp=True,
        semantic_frame_rate="25hz",
    )
    
    checkpoint = torch.load(args.sovits_model, map_location=device)
    vits_model.load_state_dict(checkpoint["model"])
    vits_model = vits_model.to(device)
    vits_model.eval()
    
    # 处理参考音频
    logger.info(f"处理参考音频: {args.reference_audio}")
    ref_audio, sr = load_audio(args.reference_audio)
    
    # 转换为16kHz (用于HuBERT特征提取)
    if sr != 16000:
        ref_audio_16k = librosa.resample(ref_audio, orig_sr=sr, target_sr=16000)
    else:
        ref_audio_16k = ref_audio
    
    # 提取HuBERT特征
    ref_audio_tensor = torch.FloatTensor(ref_audio_16k).unsqueeze(0).to(device)
    with torch.no_grad():
        ref_hubert = hubert_model(ref_audio_tensor)
    
    # 处理参考文本
    ref_text_free = args.prompt_text is None
    if not ref_text_free:
        logger.info(f"处理参考文本: {args.prompt_text}")
        ref_phones, ref_word2ph, ref_norm_text = clean_text(args.prompt_text, args.language)
        ref_phones_tensor = torch.LongTensor(ref_phones).unsqueeze(0).to(device)
    
    # 处理目标文本
    logger.info(f"处理目标文本: {args.text}")
    phones, word2ph, norm_text = clean_text(args.text, args.language)
    phones_tensor = torch.LongTensor(phones).unsqueeze(0).to(device)
    
    # 生成语义标记
    logger.info("生成语义标记...")
    with torch.no_grad():
        if ref_text_free:
            # 无参考文本模式
            prompt = None
            bert = None
            all_phoneme_ids = phones_tensor
            all_phoneme_len = torch.LongTensor([all_phoneme_ids.shape[-1]]).to(device)
        else:
            # 有参考文本模式
            prompt_semantic = vits_model.extract_latent(ref_hubert)
            prompt = prompt_semantic.unsqueeze(0).to(device)
            
            # 合并音素
            all_phoneme_ids = torch.cat([ref_phones_tensor, phones_tensor], dim=1)
            all_phoneme_len = torch.LongTensor([all_phoneme_ids.shape[-1]]).to(device)
            
            # 这里应该有BERT特征处理，但简化版本中省略
            bert = None
        
        # 生成语义标记
        pred_semantic, idx = t2s_model.model.infer_panel(
            all_phoneme_ids,
            all_phoneme_len,
            prompt,
            bert,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
        )
        
        # 只保留目标文本对应的部分
        if not ref_text_free:
            pred_semantic = pred_semantic[:, -idx:]
        
        pred_semantic = pred_semantic.unsqueeze(0)
    
    # 生成音频
    logger.info("生成音频...")
    with torch.no_grad():
        # 准备参考音频特征
        ref_spec = vits_model.get_feature_for_inference(ref_audio_tensor)
        
        # 生成音频
        audio = vits_model.decode(
            pred_semantic,
            phones_tensor,
            ref_spec,
            speed=args.speed
        ).detach().cpu().numpy()[0, 0]
    
    # 保存音频
    logger.info(f"保存音频到: {args.output_path}")
    sf.write(args.output_path, audio, sr)
    
    logger.info("推理完成")

if __name__ == '__main__':
    main()
