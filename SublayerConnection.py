import torch
import torch.nn as nn

class SublayerConnection(nn.Module):
    """
    残差连接与层归一化（对应Transformer中的Add & Norm操作）
    结构：x -> Sublayer -> Dropout -> Add -> LayerNorm
    输入输出形状保持不变：(batch_size, seq_len, d_model)
    """
    def __init__(self, d_model, dropout=0.1, norm_first=True):
        super().__init__()
        # 层归一化，对最后一个维度d_model进行归一化
        self.norm = nn.LayerNorm(d_model)  # 参数shape: (d_model,)
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first
    
    def forward(self, x, sublayer):
        """
        参数:
            x: 输入张量，shape为(batch_size, seq_len, d_model)
            sublayer: 子层函数（如多头注意力或前馈网络）
        返回:
            输出张量，shape与输入x相同
        """
        if self.norm_first:
            x = x + self.dropout(sublayer(self.norm(x)))
        else:
            x = self.norm(x + self.dropout(sublayer(x)))
        return x
