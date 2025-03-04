import os
import torch

from data import create_vocab, chinese_tokenizer, decode_sequence
from GPT import GPT

def process_data(sentence, tokenizer, vocab, max_len=512, device='cpu'):
    """处理输入数据并生成编码器输出"""
    tokens = tokenizer(sentence)
    input_ids = [vocab.get(t, vocab['<unk>']) for t in tokens[-max_len:]]
    return torch.LongTensor(input_ids).unsqueeze(0).to(device)

def get_probs(model, input_ids, vocab, temperature=1.0):
    """获取下一个token的概率分布"""
    mask = model.generate_mask(input_ids, vocab['<pad>'])
    with torch.no_grad():
        output = model(input_ids, mask)
    output = output[:, -1] / temperature
    return torch.log_softmax(output, dim=-1)

def greedy_decode(model, sentence, tokenizer, vocab, max_len=50, device='cpu'):
    model.eval()
    
    input_ids = process_data(sentence, tokenizer, vocab, device=device)
    end_token = vocab['<eos>']

    for _ in range(max_len - 1):
        probs = get_probs(model, input_ids, vocab)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if next_token.item() == end_token:
            break

    return input_ids[0].cpu().tolist()

def normalize_scores(seq, score, len_penalty=0.75):
    penalty = ((5 + len(seq)) / (5 + 1)) ** len_penalty  # Google的GNMT公式
    return score / penalty

def beam_search_decode(model, sentence, tokenizer, vocab,
                       max_len=50, beam_width=5, len_penalty=0.75, device='cpu'):  # len_penalty长度惩罚系数
    model.eval()
    
    input_ids = process_data(sentence, tokenizer, vocab, device=device)

    # 初始化：序列、分数、完成状态
    end_token = vocab['<eos>']
    beam = [ ( input_ids, 0.0 ) ]  # (tokens, score)
    # 存储完整序列
    completed = []
    
    for _ in range(max_len - 1):
        candidates = []
        
        # 遍历当前所有候选
        for seq, score in beam:
            # 如果序列已结束，直接保留
            if seq[0, -1].item() == end_token:
                candidates.append( (seq, score) )
                continue
                
            # 获取下一个token的概率分布
            probs = get_probs(model, input_ids, vocab)
            top_probs, top_indices = torch.topk(probs, beam_width, dim=-1)
            
            for i in range(beam_width):
                token = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([seq, token], dim=-1)
                new_score = score + top_probs[0, i].item()
                candidates.append( (new_seq, new_score) )
        
        # 按分数排序，保留top beam_width个
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:beam_width]
        
        # 分离已完成的序列
        new_beam = []
        for seq, score in candidates:
            if seq[0, -1].item() == end_token:
                completed.append( (seq[0].cpu().tolist(), normalize_scores(seq, score, len_penalty)) )  # 长度归一化
            else:
                new_beam.append( (seq, score) )
        beam = new_beam
        
        if not beam:
            break  # 所有候选均已完成
            
    # 合并未完成的序列
    completed += [ (seq[0].cpu().tolist(), normalize_scores(seq, score, len_penalty)) for seq, score in beam ]
    
    # 按归一化分数排序返回
    return sorted(completed, key=lambda x: x[1], reverse=True)

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

    beam_search_result = beam_search_decode(model, sentence, chinese_tokenizer, vocab, device=device)
    print('beam search decode:', decode_sequence(beam_search_result[0][0], vocab))
