## 实验设置
- 小数据集不能 warmup 太快，通过减小 accumulate_grad_batches 来增加 warmup_step，让学习率增长变缓慢点，否则会梯度爆炸
- 合并了 src_emb 和 tgt_emb，既然英语德语已经共享词典了，直接放在同一个语义空间用向量表示了
- batchsize = 32，accumulate = 8
- epoch = 19（选 18 应该更好）
- warmup_epoch = 10
- 再减小 accumulate_grad_batches 会增 ppl

## 注意
- 如果是分别定义 src_emb 和 tgt_emb，然后通过赋值来 weight tying，可能会导致有问题，不等价于共用 emb。

``` python
# 错误示例
self.linear.weight = self.emb.tok_emb.weight
self.src_emb.tok_emb.weight = self.tgt_emb.tok_emb.weight
```
- 这里 src_emb 的权重是被赋予的，而不是自己学的，我猜测这种赋值方式不会让 encoder 学到源语言的语义特征


### bleu
<div style="text-align: center;">
  <img src="./images/bleu.png" alt="bleu" style="width: auto; height: auto;">
</div>

### ppl
<div style="text-align: center;">
  <img src="./images/ppl.png" alt="ppl" style="width: auto; height: auto;">
</div>

### lr
<div style="text-align: center;">
  <img src="./images/lr.png" alt="lr" style="width: auto; height: auto;">
</div>

### inference
- sample1 的翻译效果很炸裂
<div style="text-align: center;">
  <img src="./images/infer.png" alt="infer" style="width: auto; height: auto;">
</div>
