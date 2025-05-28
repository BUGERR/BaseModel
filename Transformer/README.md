### 早期实验，检验模型代码是否正确（还是不要重复造轮子，自己写的容易错）
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
