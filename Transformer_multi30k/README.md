- 关于原文模型实现细节的一些小实验，基本一比一复刻原文。（weight tying 实现应该有问题，不建议用）
- 例如：token embedding 乘了根号 d；共享了 encoder decoder embedding 权重。
- 尝试过拟合小数据集，确定最优模型架构，细节

## 实验设置
- 英语德语共享词典
- 原文 embedding 出来没加 ln，泛化性好。加 ln，训练更稳定
- models 和 modules 在模型结构上无差别，但性能有差异，困惑（models 性能更好，可能是 attention 写法不同）
- src 和 tgt 没必要 pad 到一样长，对性能无影响
- batchsize = 32，accumulate = 16
- epoch = 20
- warmup_epoch = 10
- 数据量较少，很快就过拟合了

### bleu
<div style="text-align: center;">
  <img src="./images/my_models_ln_16.png" alt="bleu" style="width: auto; height: auto;">
</div>

### ppl
<div style="text-align: center;">
  <img src="./images/ppl_my_models_ln_16.png" alt="bleu" style="width: auto; height: auto;">
</div>

### ppl | embedding 出来没加 ln
<div style="text-align: center;">
  <img src="./images/ppl_my_models_no_ln_16.png" alt="bleu" style="width: auto; height: auto;">
</div>

### lr
<div style="text-align: center;">
  <img src="./images/lr.png" alt="bleu" style="width: auto; height: auto;">
</div>

### 测试性能：embedding 出来没加 ln
<div style="text-align: center;">
  <img src="./images/epoch20_my_models_no_ln_16.png" alt="bleu" style="width: auto; height: auto;">
</div>

## [Multi30k](https://huggingface.co/datasets/bentrevett/multi30k)
- Train: 29000
- Validation: 1014
- Test: 1000
