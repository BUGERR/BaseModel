import torch
import os
import config
from tqdm import tqdm
import torch.nn as nn

from torchsummary import summary

from data import load_data, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN

from modules.transformer import Transformer # 80,689,152
from models.model import Transformer_my # 56,287,232


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

src_lang = 'en'
tgt_lang = 'de'

tokenizer, train_dataloader, valid_dataloader = load_data(
    src_lang, tgt_lang, ['train', 'validation']
)

model_L = Transformer(
    src_pad_idx=tokenizer.token_to_id(PAD_TOKEN),
    tgt_pad_idx=tokenizer.token_to_id(PAD_TOKEN),
    src_vocab_size=tokenizer.get_vocab_size(),
    tgt_vocab_size=tokenizer.get_vocab_size(),
    max_len=config.max_len,
    hidden_size=config.d_model,
    ffn_hidden=config.d_hidden,
    num_attention_heads=config.n_head,
    num_hidden_layers=config.n_layer,
    dropout=config.p_dropout
)

model_B = Transformer_my(
    pad_idx=tokenizer.token_to_id(PAD_TOKEN),
    vocab_size=tokenizer.get_vocab_size(),
    max_len=config.max_len,
    d_model=config.d_model,
    d_hidden=config.d_hidden,
    n_head=config.n_head,
    n_layer=config.n_layer,
    p_dropout=config.p_dropout
)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model_L):,} trainable parameters")
print(f"The model has {count_parameters(model_B):,} trainable parameters")


summary(model_L, [(20, ), (20, )], dtypes=[torch.long, torch.long])
summary(model_B, [(20, ), (20, )], dtypes=[torch.long, torch.long])