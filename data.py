import torch
import re
import sentencepiece as spm

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

dataset = load_dataset('Delius/ChineseWebNovel', split='train')

def clean_text(text):
    # 去除特殊字符
    text = re.sub(r'[�◆★【】▲▼■●]', '', text)
    # 合并连续空格
    text = re.sub(r'\s+', ' ', text)
    # 过滤短文本
    if len(text) < 20:
        return None
    return text.strip()

corpus_file = 'cleaned_corpus.txt'
model_prefix = 'spm_chinese'

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

# 加载分词器
sp = spm.SentencePieceProcessor(f"{model_prefix}.model")

class TranslationDataset(Dataset):
    def __init__(self, sentences):
        super().__init__()
        self.sentences = sentences
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, index):
        if index >= 0 and index < self.__len__():
            return self.sentences[index]
        return

def collate_batch(batch, vocab, max_len=512):
    data = []
    
    for sent in batch:
        # 处理数据（添加特殊标记）
        tokens = chinese_tokenizer(sent) + ['<eos>'] 
        tokens = [vocab.get(t, vocab['<unk>']) for t in tokens[:max_len]]
        data.append(torch.LongTensor(tokens))
    
    return pad_sequence(data, batch_first=True, padding_value=vocab['<pad>'])

def create_dataloader(vocab, batch_size, shuffle=False, drop_last=False, max_len=512):
    dataset = TranslationDataset(sentences)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        collate_fn=lambda batch: collate_batch(batch, vocab, max_len))
