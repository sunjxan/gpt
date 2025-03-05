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

def sampling_decode(model, sentence, tokenizer, vocab, max_len=50,
                    temperature=1.0, top_k=None, device='cpu'):
    model.eval()
    
    input_ids = process_data(sentence, tokenizer, vocab, device=device)
    end_token = vocab['<eos>']
    
    for _ in range(max_len):
        probs = get_probs(model, input_ids, vocab)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)
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
    
    beam_search_result = beam_search_decode(model, sentence, chinese_tokenizer, vocab, device=device)
    print('beam search decode:', decode_sequence(beam_search_result[0][0], vocab))
