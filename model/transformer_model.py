"""
Transformer 时序预测模型
- 多头注意力捕捉因子间关系
- GPU加速训练和推理
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class FactorAttention(nn.Module):
    """因子间注意力 (Cross-Factor Attention)"""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm(x + attn_out)
        # FFN
        x = self.norm2(x + self.ffn(x))
        return x


class AlphaTransformer(nn.Module):
    """
    Alpha预测Transformer
    
    输入: (batch, seq_len, n_factors) 时序因子矩阵
    输出: (batch, 2) -> [预期收益, 预期波动率]
    """
    
    def __init__(self, n_factors: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, seq_len: int = 60, dropout: float = 0.1):
        super().__init__()
        
        self.n_factors = n_factors
        self.d_model = d_model
        self.seq_len = seq_len
        
        # 因子嵌入
        self.factor_embedding = nn.Sequential(
            nn.Linear(n_factors, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, seq_len, dropout)
        
        # Transformer层
        self.layers = nn.ModuleList([
            FactorAttention(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        
        # 时序聚合 (Attention Pooling)
        self.temporal_attn = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1),
        )
        
        # 多任务输出头
        self.return_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),  # 预期收益
        )
        
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),  # 预期波动率
            nn.Softplus(),  # 确保波动率为正
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, n_factors)
        returns: (pred_return, pred_vol) each (batch, 1)
        """
        # 因子嵌入
        h = self.factor_embedding(x)  # (B, T, D)
        
        # 位置编码
        h = self.pos_encoding(h)
        
        # 因果mask (只看过去)
        mask = torch.triu(
            torch.ones(self.seq_len, self.seq_len, device=x.device),
            diagonal=1
        ).bool()
        
        # Transformer编码
        for layer in self.layers:
            h = layer(h, mask=mask)
        
        # Attention Pooling 时序聚合
        attn_weights = self.temporal_attn(h)  # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(h * attn_weights, dim=1)  # (B, D)
        
        # 多任务输出
        pred_return = self.return_head(context)      # (B, 1)
        pred_vol = self.volatility_head(context)      # (B, 1)
        
        return pred_return, pred_vol


class ModelManager:
    """模型管理器 - 训练/推理/更新"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda:0'))
        
        self.model = AlphaTransformer(
            n_factors=config.get('n_factors', 50),
            d_model=config['transformer']['d_model'],
            nhead=config['transformer']['nhead'],
            num_layers=config['transformer']['num_layers'],
            seq_len=config['transformer']['seq_len'],
            dropout=config['transformer']['dropout'],
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        logger.info(f"模型加载到 {self.device}, "
                    f"参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def train_epoch(self, dataloader) -> dict:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for batch_x, batch_y_ret, batch_y_vol in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y_ret = batch_y_ret.to(self.device)
            batch_y_vol = batch_y_vol.to(self.device)
            
            pred_ret, pred_vol = self.model(batch_x)
            
            # 多任务损失
            loss_ret = F.mse_loss(pred_ret.squeeze(), batch_y_ret)
            loss_vol = F.mse_loss(pred_vol.squeeze(), batch_y_vol)
            
            # IC损失 (最大化截面IC)
            loss_ic = -self._pearson_corr(pred_ret.squeeze(), batch_y_ret)
            
            loss = loss_ret + 0.5 * loss_vol + 0.3 * loss_ic
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        self.scheduler.step()
        
        return {
            'avg_loss': total_loss / max(n_batches, 1),
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    @torch.no_grad()
    def predict(self, factor_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        实时推理
        factor_matrix: (n_stocks, seq_len, n_factors)
        returns: (pred_returns, pred_vols) each (n_stocks,)
        """
        self.model.eval()
        x = torch.FloatTensor(factor_matrix).to(self.device)
        
        pred_ret, pred_vol = self.model(x)
        
        return (
            pred_ret.squeeze().cpu().numpy(),
            pred_vol.squeeze().cpu().numpy()
        )
    
    @staticmethod
    def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算Pearson相关系数"""
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        corr = (x_centered * y_centered).sum() / (
            torch.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum()) + 1e-8
        )
        return corr
    
    def save(self, path: str):
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }, path)
        
    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
