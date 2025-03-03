import torch
import jieba

from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# 原始数据
sentences = [
    '我爱学习人工智能',
    '深度学习改变世界',
    '自然语言处理很强大',
    '神经网络非常复杂'
]

def filter_examples(example):   
    return (
        1 < len(example) < 150 and   # 控制中文长度
        not any(c in example for c in ['�', '�'])  # 过滤非法字符
    )

# 数据清洗
sentences = list(filter(filter_examples, sentences))

# 定义特殊标记
SPECIAL_TOKENS = ['<pad>', '<sos>', '<eos>', '<unk>']

# 中文处理函数
def chinese_tokenizer(text):
    """中文按字符分割"""
    return list(jieba.cut(text.strip()))

# 构建词汇表
def build_vocab(sentences, tokenizer):
    """构建词汇表"""
    counter = Counter()
    for sent in sentences:
        counter.update(tokenizer(sent))
    vocab = {token: i+len(SPECIAL_TOKENS) for i, token in enumerate(sorted(counter))}
    # 添加特殊标记
    for i, token in enumerate(SPECIAL_TOKENS):
        vocab[token] = i
    return vocab

def create_vocab():
    # 生成词汇表
    return build_vocab(sentences, chinese_tokenizer)

# 预测结果解码
def decode_sequence(ids, vocab):
    idx2token = {v: k for k, v in vocab.items()}
    return ''.join([idx2token.get(i, '<unk>')
                     for i in ids if i not in [vocab['<pad>'], vocab['<sos>'], vocab['<eos>']]])

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
        tokens = ['<sos>'] + chinese_tokenizer(sent) + ['<eos>'] 
        tokens = [vocab.get(t, vocab['<unk>']) for t in tokens[:max_len]]
        data.append(torch.LongTensor(tokens))
    
    return pad_sequence(data, batch_first=True, padding_value=vocab['<pad>'])

def create_dataloader(vocab, batch_size, shuffle=False, drop_last=False, max_len=512):
    dataset = TranslationDataset(sentences)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        collate_fn=lambda batch: collate_batch(batch, vocab, max_len))
