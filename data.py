import os
import re
import torch
import sentencepiece as spm

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

corpus_file = 'cleaned_corpus.txt'
model_prefix = 'spm_chinese'

def clean_text(text):
    # 去除特殊字符
    text = re.sub(r'[�◆★【】▲▼■●]', '', text)
    # 合并连续空格
    text = re.sub(r'\s+', ' ', text)
    # 过滤短文本
    if len(text) < 20:
        return None
    return text.strip()

if not os.path.exists(f"{model_prefix}.model"):
    
    if not os.path.exists(corpus_file):
        
        dataset = load_dataset('Delius/ChineseWebNovel', split='train')
        
        # 应用清洗并保存
        with open(corpus_file, "w", encoding="utf-8") as f:
            for example in dataset:
                cleaned = clean_text(example["Response"])
                if cleaned:
                    f.write(cleaned + "\n")
    
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=32000,
        character_coverage=0.9995,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        model_type='unigram',
        user_defined_symbols=['<sep>', '<cls>']  # 自定义特殊符号
    )

def create_tokenizer():
    return spm.SentencePieceProcessor(f"{model_prefix}.model")

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = tokenizer.encode(line.strip()) + [tokenizer.eos_id()]
                for i in range(0, len(tokens), max_len):
                    text = tokens[i:i+max_len]
                    if len(text) > 1:
                        self.data.append(text)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_batch(batch, tokenizer):
    data = []
    
    for item in batch:
        data.append(torch.LongTensor(item))
    
    return pad_sequence(data, batch_first=True, padding_value=tokenizer.pad_id())

def create_dataloader(tokenizer, batch_size, max_len=512, shuffle=False, drop_last=False):
    dataset = TextDataset(corpus_file, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        collate_fn=lambda batch: collate_batch(batch, tokenizer))
