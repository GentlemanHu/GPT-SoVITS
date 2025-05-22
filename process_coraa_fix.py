def process_coraa(extract_dir, output_dir, dialect):
    """
    处理CORAA数据集 (巴西葡萄牙语)
    注意: CORAA数据集需要单独申请下载
    """
    logger.info("处理CORAA数据集...")
    
    # 检查是否存在CORAA数据集的实际音频文件
    # CORAA数据集通常包含以下结构:
    # - audio/ 目录包含音频文件
    # - transcriptions/ 目录包含转录文件
    
    # 查找可能的目录结构
    possible_dirs = []
    for root, dirs, files in os.walk(extract_dir):
        for d in dirs:
            if any(keyword in d.lower() for keyword in ['audio', 'wav', 'speech', 'coraa']):
                possible_dirs.append(os.path.join(root, d))
    
    logger.info(f"找到可能的音频目录: {possible_dirs}")
    
    # 查找转录文件
    transcript_files = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith(('.txt', '.csv', '.tsv')) and any(keyword in file.lower() for keyword in ['transcript', 'text', 'label']):
                transcript_files.append(os.path.join(root, file))
    
    logger.info(f"找到可能的转录文件: {transcript_files}")
    
    # 如果没有找到音频和转录文件，说明这不是完整的CORAA数据集
    if not possible_dirs or not transcript_files:
        logger.warning("未找到CORAA数据集的音频文件和转录文件")
        logger.warning("CORAA数据集需要从官方网站单独申请和下载")
        logger.warning("正在创建示例数据集以便继续流程...")
        
        # 创建示例数据集
        import subprocess
        subprocess.run([
            sys.executable, 'create_sample_data.py', 
            '--output-dir', output_dir, 
            '--language', 'pt'
        ])
        return
    
    # 如果找到了文件，继续处理
    audio_dirs = possible_dirs
    
    # 创建metadata.csv
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    logger.info(f"创建metadata.csv: {metadata_path}")
    
    with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
        # 写入标题
        metadata_file.write("file_path|speaker|text|language\n")
        
        # 处理每个转录文件
        count = 0
        for transcript_file in transcript_files:
            logger.info(f"处理转录文件: {transcript_file}")
            
            try:
                with open(transcript_file, 'r', encoding='utf-8') as trans_file:
                    for line in trans_file:
                        if not line.strip() or line.startswith('#'):
                            continue
                        
                        # 尝试不同的分隔符
                        parts = None
                        if '\t' in line:
                            parts = line.strip().split('\t')
                        elif '|' in line:
                            parts = line.strip().split('|')
                        elif ',' in line:
                            parts = line.strip().split(',')
                        else:
                            # 尝试空格分隔
                            parts = line.strip().split(' ', 1)
                        
                        if not parts or len(parts) < 2:
                            continue
                        
                        filename = parts[0]
                        text = parts[1]
                        
                        # 查找音频文件
                        audio_path = None
                        for audio_dir in audio_dirs:
                            for root, dirs, files in os.walk(audio_dir):
                                for file in files:
                                    if filename in file or file.startswith(filename):
                                        audio_path = os.path.join(root, file)
                                        break
                                if audio_path:
                                    break
                            if audio_path:
                                break
                        
                        if not audio_path:
                            continue
                        
                        # 复制并转换音频文件
                        wav_filename = f"{filename}.wav"
                        wav_path = os.path.join(output_dir, wav_filename)
                        
                        # 使用ffmpeg转换到wav
                        try:
                            subprocess.run([
                                'ffmpeg', '-i', audio_path, '-ar', '24000', '-ac', '1',
                                '-hide_banner', '-loglevel', 'error', wav_path
                            ], check=True)
                        except subprocess.CalledProcessError:
                            logger.warning(f"转换音频文件失败: {audio_path}")
                            continue
                        
                        # 提取说话者ID
                        speaker = f"speaker_{filename.split('_')[0] if '_' in filename else 'default'}"
                        
                        # 写入metadata
                        metadata_file.write(f"{wav_filename}|{speaker}|{text}|pt\n")
                        count += 1
                        
                        # 限制数量
                        if count >= 100:  # 限制为100个样本用于测试
                            break
            
            except Exception as e:
                logger.error(f"处理转录文件 {transcript_file} 时出错: {e}")
                continue
            
            if count >= 100:
                break
        
        # 如果没有成功处理任何文件，创建示例数据集
        if count == 0:
            logger.warning("未能处理任何音频文件，创建示例数据集...")
            # 关闭当前文件
            metadata_file.close()
            
            # 创建示例数据集
            import subprocess
            subprocess.run([
                sys.executable, 'create_sample_data.py', 
                '--output-dir', output_dir, 
                '--language', 'pt'
            ])
            return
    
    logger.info(f"处理了 {count} 个音频文件")
