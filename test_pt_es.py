#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试葡萄牙语和西班牙语支持的完整功能
"""

import os
import sys
import argparse
import logging
import torch
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('test_pt_es')

def main():
    parser = argparse.ArgumentParser(description='测试葡萄牙语和西班牙语支持')
    parser.add_argument('--language', type=str, required=True,
                        choices=['es', 'pt'],
                        help='语言 (es: 西班牙语, pt: 葡萄牙语)')
    parser.add_argument('--test-type', type=str, default='all',
                        choices=['bert', 'g2p', 'segmenter', 'inference', 'all'],
                        help='测试类型 (默认: all)')
    parser.add_argument('--sample-audio', type=str, default=None,
                        help='样本音频路径 (用于推理测试)')
    parser.add_argument('--sample-text', type=str, default=None,
                        help='样本文本 (用于推理测试)')
    parser.add_argument('--output-dir', type=str, default='./test_output',
                        help='输出目录 (默认: ./test_output)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备 (默认: cuda 如果可用，否则 cpu)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置样本文本
    if args.sample_text is None:
        if args.language == 'es':
            args.sample_text = "Hola, esto es una prueba. Me llamo Claude y soy un asistente de inteligencia artificial."
        else:
            args.sample_text = "Olá, isto é um teste. Meu nome é Claude e sou um assistente de inteligência artificial."
    
    # 运行测试
    if args.test_type in ['bert', 'all']:
        test_bert(args)
    
    if args.test_type in ['g2p', 'all']:
        test_g2p(args)
    
    if args.test_type in ['segmenter', 'all']:
        test_segmenter(args)
    
    if args.test_type in ['inference', 'all']:
        if args.sample_audio is None:
            logger.warning("没有提供样本音频，跳过推理测试")
        else:
            test_inference(args)
    
    logger.info("测试完成")

def test_bert(args):
    """
    测试BERT模型
    """
    logger.info(f"测试{args.language}语言的BERT模型...")
    
    from transformers import AutoTokenizer, AutoModel
    
    # 设置BERT模型路径
    if args.language == 'es':
        bert_path = "pretrained_models/spanish-bert"
    else:
        bert_path = "pretrained_models/portuguese-bert"
    
    try:
        # 加载分词器和模型
        tokenizer = AutoTokenizer.from_pretrained(bert_path)
        model = AutoModel.from_pretrained(bert_path)
        
        # 测试文本
        inputs = tokenizer(args.sample_text, return_tensors="pt")
        outputs = model(**inputs)
        
        # 获取最后一层隐藏状态
        last_hidden_states = outputs.last_hidden_state
        
        logger.info(f"BERT输出形状: {last_hidden_states.shape}")
        logger.info(f"BERT测试通过")
        
        # 保存结果
        torch.save(last_hidden_states, os.path.join(args.output_dir, f"bert_{args.language}_output.pt"))
        
        return True
    except Exception as e:
        logger.error(f"BERT测试失败: {e}")
        return False

def test_g2p(args):
    """
    测试G2P(石墨音素)转换
    """
    logger.info(f"测试{args.language}语言的G2P转换...")
    
    # 导入相应的G2P模块
    if args.language == 'es':
        from text.spanish import g2p, normalize_text
    else:
        from text.portuguese import g2p, normalize_text
    
    try:
        # 规范化文本
        normalized_text = normalize_text(args.sample_text)
        logger.info(f"规范化文本: {normalized_text}")
        
        # 转换为音素
        phones, word2ph = g2p(normalized_text)
        
        logger.info(f"音素: {phones}")
        logger.info(f"词到音素映射: {word2ph}")
        logger.info(f"G2P测试通过")
        
        # 保存结果
        with open(os.path.join(args.output_dir, f"g2p_{args.language}_output.txt"), 'w', encoding='utf-8') as f:
            f.write(f"原始文本: {args.sample_text}\n")
            f.write(f"规范化文本: {normalized_text}\n")
            f.write(f"音素: {' '.join(phones)}\n")
            f.write(f"词到音素映射: {word2ph}\n")
        
        return True
    except Exception as e:
        logger.error(f"G2P测试失败: {e}")
        return False

def test_segmenter(args):
    """
    测试语言分割器
    """
    logger.info("测试语言分割器...")
    
    from text.LangSegmenter.langsegmenter import LangSegmenter
    
    try:
        # 准备混合语言文本
        if args.language == 'es':
            mixed_text = "Hello, mi nombre es Claude. I speak both español and English."
        else:
            mixed_text = "Hello, meu nome é Claude. I can speak português and English."
        
        # 分割文本
        segments = LangSegmenter.getTexts(mixed_text)
        
        logger.info(f"混合文本: {mixed_text}")
        for i, segment in enumerate(segments):
            logger.info(f"片段 {i+1}: 语言={segment['lang']}, 文本={segment['text']}")
        
        logger.info("语言分割器测试通过")
        
        # 保存结果
        with open(os.path.join(args.output_dir, "segmenter_output.txt"), 'w', encoding='utf-8') as f:
            f.write(f"混合文本: {mixed_text}\n")
            for i, segment in enumerate(segments):
                f.write(f"片段 {i+1}: 语言={segment['lang']}, 文本={segment['text']}\n")
        
        return True
    except Exception as e:
        logger.error(f"语言分割器测试失败: {e}")
        return False

def test_inference(args):
    """
    测试推理功能
    """
    logger.info("测试推理功能...")
    
    # 检查GPT和SoVITS模型
    gpt_model_path = f"checkpoints/{args.language}_gpt/best_model.pth"
    sovits_model_path = f"checkpoints/{args.language}_sovits/best_model.pth"
    
    if not os.path.exists(gpt_model_path):
        logger.warning(f"未找到GPT模型: {gpt_model_path}")
        return False
    
    if not os.path.exists(sovits_model_path):
        logger.warning(f"未找到SoVITS模型: {sovits_model_path}")
        return False
    
    try:
        # 导入推理模块
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
        logger.info(f"加载GPT模型: {gpt_model_path}")
        t2s_model = Text2SemanticLightningModule.load_from_checkpoint(
            gpt_model_path,
            map_location=device
        )
        t2s_model = t2s_model.to(device)
        t2s_model.eval()
        
        # 加载SoVITS模型
        logger.info(f"加载SoVITS模型: {sovits_model_path}")
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
        
        checkpoint = torch.load(sovits_model_path, map_location=device)
        vits_model.load_state_dict(checkpoint["model"])
        vits_model = vits_model.to(device)
        vits_model.eval()
        
        # 处理参考音频
        logger.info(f"处理参考音频: {args.sample_audio}")
        ref_audio, sr = librosa.load(args.sample_audio, sr=24000, mono=True)
        
        # 转换为16kHz (用于HuBERT特征提取)
        if sr != 16000:
            ref_audio_16k = librosa.resample(ref_audio, orig_sr=sr, target_sr=16000)
        else:
            ref_audio_16k = ref_audio
        
        # 提取HuBERT特征
        ref_audio_tensor = torch.FloatTensor(ref_audio_16k).unsqueeze(0).to(device)
        with torch.no_grad():
            ref_hubert = hubert_model(ref_audio_tensor)
        
        # 处理目标文本
        logger.info(f"处理目标文本: {args.sample_text}")
        phones, word2ph, norm_text = clean_text(args.sample_text, args.language)
        phones_tensor = torch.LongTensor(phones).unsqueeze(0).to(device)
        
        # 生成语义标记
        logger.info("生成语义标记...")
        with torch.no_grad():
            # 无参考文本模式
            prompt = None
            bert = None
            all_phoneme_ids = phones_tensor
            all_phoneme_len = torch.LongTensor([all_phoneme_ids.shape[-1]]).to(device)
            
            # 生成语义标记
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                top_k=5,
                top_p=0.7,
                temperature=0.7,
            )
            
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
                speed=1.0
            ).detach().cpu().numpy()[0, 0]
        
        # 保存音频
        output_path = os.path.join(args.output_dir, f"test_{args.language}_output.wav")
        logger.info(f"保存音频到: {output_path}")
        sf.write(output_path, audio, 24000)
        
        logger.info("推理测试通过")
        return True
    except Exception as e:
        logger.error(f"推理测试失败: {e}")
        return False

if __name__ == '__main__':
    main()
