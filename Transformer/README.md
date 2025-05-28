### 早期实验，检验模型代码是否正确（还是不要重复造轮子，自己写的容易错）
#### 这版的模型是自己写的，很多小错，不建议用。
- previous error1：evaluate 的 gt 没加 []，导致 bleu 算不了

``` python
# 注意两者区别，因为 gt 是 str 句子，要转成 list。而 pred_sentence 本身就是 list。
all_predictions.append("".join(pred_sentence))
all_gt.append(["".join(gt_sentence)])
```

- error2：decoder 的 enc_dec_attention 没定义，和 tgt_self_attention 用重了
- error3：Transformer 的 src_embedding 和 tgt_embedding 初始化了，但后续忘了用
- error4：decoder 的 forward 里应保持输入变量名 tgt 不变进行操作，我之前里面给写成 x 了，前后不一致

``` python
  def forward(self, tgt, enc_src, tgt_mask, src_mask):
      for layer in self.layers:
          tgt = layer(tgt, enc_src, tgt_mask, src_mask)
```

- error5：ScaledDotProductAttention 里 mask 注意是 -inf，之前没加负号

``` python
batch_size, num_attention_heads, seq_len, d_key = k.size()

# 1. Compute the dot product between query and key^T
k_t = k.transpose(-2, -1)
scores = q @ k_t / math.sqrt(d_key)

# 2. Apply mask (optional)
if mask is not None:
    scores = scores.masked_fill(mask.logical_not(), float("-inf"))

attn = F.softmax(scores, dim=-1)

out = attn @ v
```
