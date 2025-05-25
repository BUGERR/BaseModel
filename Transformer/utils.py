import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

from typing import Sequence, Union
import config
import data
from modules.transformer import make_pad_mask, Transformer
from tokenizers import Tokenizer


class WarmupScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        hidden_size: int,
        warmup_step: int,
        last_epoch: int = -1,
    ) -> None:
        self.hidden_size = hidden_size
        self.warmup_step = warmup_step
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.hidden_size**-0.5
            * min(
                (self.last_epoch + 1) ** -0.5,
                (self.last_epoch + 1) * self.warmup_step**-1.5,
            )
        ]
    
@torch.no_grad
def greedy_search(model, enc_output, src_mask, max_len, sos_idx, eos_idx, pad_idx):
    '''
    - 贪心推理过程：做翻译，给一句话，根据这句话编码后的输出，求翻译后的句子
    - enc_output: [bs, seq_len, d_model]
    '''
    device = enc_output.device
    batch_size, seq_len, d_model = enc_output.shape

    # 解码器最初的输入，开始 token
    # sos: [bs, 1]
    ys = torch.ones(batch_size, 1, dtype=torch.long, device=device).fill_(
        sos_idx
    )
    # 记录 batch 里的某句话是否翻译结束了，首次预测为 eos 则意味着结束翻译，后续补0
    ended = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for i in range(max_len - 1):
        # dec_output: [bs, seq_len - 1, vocab_size]
        # 只要每 decoder 输出的最后一个 token
        # logits: [bs, vocab_size]
        logits = model.decode(ys, enc_output, src_mask)[:, -1]

        # next_words: [bs]
        next_words = torch.argmax(logits, dim=1)

        # sos: [bs, 2]
        ys = torch.cat([ys, next_words.unsqueeze(1)], dim=1)
        
        # 如果下一个词第一次出现 eos，这句话已经翻译结束，意味着如果后续还预测为其他，做 padding 处理
        ended = ended | (next_words == eos_idx)

        # 因为 batch 里的句子长短不一致，短的句子会提前 eos，然后做 pad 来等长句子结束。
        # 如果一句话已经结束了，且不是第一个 eos 标记，则把最后一个 token 按 padding 处理
        '''
        例如：
        <start> hide new secretions from the parental units <extract>
        <start> goes to absurd lengths <extract> <pad> <pad> <pad>
        '''
        ys[ended & (ys[:, -1] != eos_idx), -1] = pad_idx

        # 如果每句话都输出过 eos，停止循环
        if ended.all():
            break
    
    # 如果 batch 里有句话一直没预测为 eos，且到了长度上限，则让那句没结束的话结束，已经结束的继续补 0
    if i == max_len - 2:  # reach max length
        ys[~ended, -1] = eos_idx
        ys[ended, -1] = pad_idx

    # 注意输出是包含初始的特殊 token sos，但不影响 bleu 分数的计算，因为用了 tokenizer.decode_batch
    # 去掉了特殊词元，只留下了文本部分（写法上是否再维护一个变量 pred_tokens 更好？无所谓，因为只用来计算 bleu

    return ys


@torch.no_grad
def sample(
    model, 
    enc_output, 
    src_mask, 
    temperature, 
    top_k,
    top_p, 
    max_len, 
    sos_idx, 
    eos_idx, 
    pad_idx, 
):
    '''
    - top_k + top_p
    - enc_output: [bs, seq_len_en, d_model]
    - logits: [bs, seq_len++, vocab_size]
    '''
    device = enc_output.device
    batch_size, seq_len, d_model = enc_output.shape
    sos = torch.ones(batch_size, 1, dtype=torch.long, device=device).fill_(sos_idx)
    ended = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for i in range(max_len - 1):
        # [bs, vocab_size]
        logits = model.decode(sos, enc_output, src_mask)[:, -1]

        logits = logits / temperature

        # Top-k
        # 取 logits 最后一维 vocab_size 中前 k 大的值，其余填为无穷小
        if top_k > 0:
            top_k_values, _ = torch.topk(logits, top_k)
            
            # 这前 k 大值中最小的，即倒一，记得加一维
            min_top_k_values = top_k_values[:, -1].unsqueeze(-1)
            
            # 其余填为无穷小
            logits = torch.where(
                logits < min_top_k_values, 
                torch.full_like(logits, float("-inf")), 
                logits, 
            )

        # Top-p (nucleus) sampling
        #  按降序对 logits 排序，选概率累积到 p 的
        if top_p < 1.0:
            # 降序值，和对应在原 logits 中位置的索引号
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            # 计算张量沿指定维度的累积和，返回该位置和前面词概率的求和 [bs, vocab_size]
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # 待移除位置的矩阵 [bs, vocab_size], dtype = bool
            indices_to_remove = cumulative_probs > top_p

            # 右移操作，保证端点值能取到。例如：前三个词概率和刚好为 p，则三个都候选
            indices_to_remove[:, 1:] = indices_to_remove[:, :-1].clone()

            # 第一列为 False，表示最大概率值不会被移除，保证有值
            indices_to_remove[:, 0] = 0

            # 将 remove 中的值写入到 sorted_indices 索引指定的位置
            indices_to_remove = indices_to_remove.scatter(
                1, sorted_indices, indices_to_remove
            )

            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        # 从 softmax 概率分布中随机采样，后续操作同贪心
        next_words = torch.multinomial(probs, num_samples=1).squeeze(1)

        sos = torch.cat([sos, next_words.unsqueeze(1)], dim=1)

        ended  = ended | (next_words == eos_idx)

        sos[ended & (sos[:, -1] != eos_idx), -1] = pad_idx

        if ended.all():
            break
    
    if i == max_len - 2:
        sos[~ended, -1] = eos_idx
        sos[ended, -1] = pad_idx

    return sos

@torch.no_grad
def translate_sentence(
        sentences: Union[Sequence[str], str], 
        model: Transformer, 
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        max_len=config.max_len, 
        do_sample=False, 
        temperature=None, 
        top_k=None,
        top_p=None, 
):
    '''
    - *greedy decoding* do_sample=False, calling ['utils.greedy_search']
    - *numtinomial sampling*: do_sample=True, calling ['utils.sample']
    '''
    if isinstance(sentences, str):
        sentences = [sentences]

    device = next(model.parameters()).device

    sos_idx = tgt_tokenizer.token_to_id(data.SOS_TOKEN)
    eos_idx = tgt_tokenizer.token_to_id(data.EOS_TOKEN)
    pad_idx = tgt_tokenizer.token_to_id(data.PAD_TOKEN)

    src_tensor = torch.LongTensor(
        [encoding.ids for encoding in src_tokenizer.encode_batch(sentences)]
    ).to(device)

    enc_output = model.encode(src_tensor)
    src_mask = make_pad_mask(src_tensor, src_tokenizer.token_to_id(data.PAD_TOKEN))

    if do_sample == False:
        tgt_tokens = greedy_search(
            model, enc_output, src_mask, max_len, sos_idx, eos_idx, pad_idx
        )
    elif do_sample == True:
        temperature = temperature if temperature is not None else config.temperature
        top_k = top_k if top_k is not None else config.top_k
        top_p = top_p if top_p is not None else config.top_p

        tgt_tokens = sample(
            model, enc_output, src_mask, temperature, top_k, top_p, max_len, sos_idx, eos_idx, pad_idx
        )

    return ["".join(s) for s in tgt_tokenizer.decode_batch(tgt_tokens.cpu().numpy())]



