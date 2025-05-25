import torch
import os
import config
from tqdm import tqdm
import torch.nn as nn

from data import load_data, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
from modules.transformer import Transformer
from models.model import Transformer

import sacrebleu
from utils import greedy_search, WarmupScheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

src_lang = 'en'
tgt_lang = 'de'

tokenizer, train_dataloader, valid_dataloader = load_data(
    src_lang, tgt_lang, ['train', 'validation']
)

model = Transformer(
    pad_idx=tokenizer.token_to_id(PAD_TOKEN),
    vocab_size=tokenizer.get_vocab_size(),
    max_len=config.max_len,
    d_model=config.d_model,
    d_hidden=config.d_hidden,
    n_head=config.n_head,
    n_layer=config.n_layer,
    p_dropout=config.p_dropout
).to(device)

sos_idx = tokenizer.token_to_id(SOS_TOKEN)
eos_idx = tokenizer.token_to_id(EOS_TOKEN)
pad_idx = tokenizer.token_to_id(PAD_TOKEN)

def split_batch(batch):
    '''
    batch: ([bs, seq_len_en], [bs, seq_len_de]) 注：完整的 <sos> + ... + <eos> + <pad>

    tgt: de --> [bs, seq_len_de - 1] 注：解码器输入，只要 <sos> + ... + <eos>

    gt: 解码器输出，是左移的 label, 只要 ... + <eos> + <pad>, 目的只是为了把上下对应位置的 token 错开
    '''
    src, tgt = batch
    tgt, gt = tgt[:, :-1], tgt[:, 1:]
    return src.to(device), tgt.to(device), gt.to(device)


def train(epoch, model, criterion, optimizer, scheduler):
    model.train()
    total_loss = 0
    step = 0
    optimizer.zero_grad()

    for batch in tqdm(train_dataloader, desc=f"Training epoch {epoch + 1}"):
        # tgt: decoder input
        # gt (ground truth): training target [bs, seq_len_de - 1]
        src, tgt, gt = split_batch(batch)

        # [bs, seq_len_de - 1, vocab_size]
        # teacher forcing：用真实的前面的数据作为 tgt 输入，来预测下一个词
        outputs = model(src, tgt)

        outputs = outputs.contiguous().view(-1, tokenizer.get_vocab_size())

        loss = criterion(outputs, gt.contiguous().view(-1)) / config.accumulate_grad_batches

        loss.backward()

        # 小 trick：梯度累积
        if (step + 1) % config.accumulate_grad_batches == 0 or (step + 1) == len(train_dataloader):
            nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        step += 1
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_dataloader)
    
    return avg_loss


@torch.no_grad()
def evaluate(model, criterion):
    model.eval()
    total_loss = 0

    # 存 id 转回 token 最后再拼回去的句子
    all_gt = []
    all_predictions = []

    for batch in tqdm(valid_dataloader, desc="Evaluating"):
        src, tgt, gt = split_batch(batch)

        enc_output = model.encode(src)
        src_mask = model.make_src_mask(src)
        dec_output = model.decode(tgt, enc_output, src_mask)

        loss = (
                criterion(
                    dec_output.contiguous().view(-1, tokenizer.get_vocab_size()),
                    gt.contiguous().view(-1),
                )
                / config.accumulate_grad_batches
        )
        total_loss += loss.item()

        pred_tokens = greedy_search(
            model, enc_output, src_mask, config.max_len, sos_idx, eos_idx, pad_idx
        )

        pred_sentence = tokenizer.decode_batch(pred_tokens.cpu().numpy())
        gt_sentence = tokenizer.decode_batch(gt.cpu().numpy())

        all_predictions.append("".join(pred_sentence))
        all_gt.append(["".join(gt_sentence)])

    avg_loss = total_loss / len(valid_dataloader)

    # bleu
    if len(all_predictions) > 0:
        bleu_score = sacrebleu.corpus_bleu(all_predictions, all_gt)
        avg_bleu = bleu_score.score
    else:
        avg_bleu = 0

    return avg_loss, avg_bleu


def training_loop(restore_epoch = 0):
    '''
    - 定义 optimizer, scheduler, criterion
    - restore 重加载
    '''
    # 定义优化器，
    optimizer = torch.optim.Adam(
        model.parameters(), betas=config.betas, eps=config.adam_eps
    )
    scheduler = WarmupScheduler(optimizer, config.d_model, config.warmup_step)
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.token_to_id(PAD_TOKEN),
        label_smoothing=config.eps_ls
    )

    # 重加载
    restore_ckpt_path = config.checkpoint_dir / f"en_de_{restore_epoch}.pth"

    if restore_epoch != -1 and os.path.exists(restore_ckpt_path):
        ckpt = torch.load(restore_ckpt_path)
        assert ckpt["epoch"] == restore_epoch
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        restore_epoch = 0

    best_bleu = 0
    for epoch in range(restore_epoch, config.epochs):
        avg_train_loss = train(epoch, model, criterion, optimizer, scheduler)
        avg_valid_loss, avg_bleu = evaluate(model, criterion)
        print(
            f"Epoch {epoch + 1}/{config.epochs}, Training Loss: {avg_train_loss: .4f}, Validation Loss: {avg_valid_loss:.4f}, BLEU Score: {avg_bleu:.2f}"
        )

        checkpoint_path = config.checkpoint_dir / "en_de_.pth"

        # if avg_bleu > best_bleu:
        #     best_bleu = avg_bleu
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            checkpoint_path,
        )

if __name__ == '__main__':
    training_loop(-1)