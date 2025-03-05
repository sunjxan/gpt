import os
import torch

from data import create_vocab, chinese_tokenizer, decode_sequence
from GPT import GPT

def process_data(model, sentence, tokenizer, vocab, device='cpu'):
    """处理输入数据并生成编码器输出"""
    tokens = tokenizer(sentence)[-model.max_seq_len:]
    input_ids = [vocab.get(t, vocab['<unk>']) for t in tokens]
    return torch.LongTensor(input_ids).unsqueeze(0).to(device)

def get_probs(model, input_ids, vocab, temperature=1.0, top_k=None):
    """获取下一个token的概率分布"""
    input_ids = input_ids[:, -model.max_seq_len:]
    mask = model.generate_mask(input_ids, vocab['<pad>'])
    with torch.no_grad():
        output = model(input_ids, mask)
    # 应用温度缩放
    output = output[:, -1] / temperature
    # Top-k 过滤
    if top_k is not None and 0 < top_k <= output.size(-1):
        indices_to_remove = output < torch.topk(output, top_k)[0][..., -1, None]
        output[indices_to_remove] = float('-inf')
    return torch.softmax(output, dim=-1)

def greedy_decode(model, sentence, tokenizer, vocab, max_len=50, device='cpu'):
    model.eval()
    
    input_ids = process_data(model, sentence, tokenizer, vocab, device=device)
    end_token = vocab['<eos>']
    
    for _ in range(max_len):
        probs = get_probs(model, input_ids, vocab)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if next_token.item() == end_token:
            break
    
    return input_ids[0].cpu().tolist()

def sampling_decode(model, sentence, tokenizer, vocab, max_len=50,
                    temperature=1.0, top_k=1, device='cpu'):
    model.eval()
    
    input_ids = process_data(model, sentence, tokenizer, vocab, device=device)
    end_token = vocab['<eos>']
    
    for _ in range(max_len):
        probs = get_probs(model, input_ids, vocab, temperature, top_k)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if next_token.item() == end_token:
            break
    
    return input_ids[0].cpu().tolist()

if __name__ == '__main__':
    vocab = create_vocab()
    
    # 创建模型
    model = GPT(len(vocab))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    ckpt_path = './checkpoints/checkpoint_best.pth'
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    sentence = input('请输入中文句子：\n')
    print('input:', sentence)
    
    predictions = greedy_decode(model, sentence, chinese_tokenizer, vocab, device=device)
    print('greedy decode:', decode_sequence(predictions, vocab))
    
    # 技术文档生成（高确定性）
    predictions = sampling_decode(model, sentence, chinese_tokenizer, vocab, max_len=50, temperature=0.7, top_k=1, device=device)
    print('sampling decode(高确定性):', decode_sequence(predictions, vocab))
    
    # 创意写作（高多样性）
    predictions = sampling_decode(model, sentence, chinese_tokenizer, vocab, max_len=50, temperature=1.2, top_k=3, device=device)
    print('sampling decode(高多样性):', decode_sequence(predictions, vocab))
    
    # 平衡模式
    predictions = sampling_decode(model, sentence, chinese_tokenizer, vocab, max_len=50, temperature=0.9, top_k=2, device=device)
    print('sampling decode(平衡模式):', decode_sequence(predictions, vocab))
