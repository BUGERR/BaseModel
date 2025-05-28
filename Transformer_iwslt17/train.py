import math
import time
import torch
import os
import config
from tqdm import tqdm
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from data import load_data, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
from models.model import Transformer_my
# from modules import Transformer # 56,287,232

import sacrebleu
from utils import greedy_search, WarmupScheduler

from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

src_lang = 'en'
tgt_lang = 'de'

tokenizer, train_dataloader, valid_dataloader = load_data(
    src_lang, tgt_lang, ['train', 'validation']
)


model = Transformer_my(
    pad_idx=tokenizer.token_to_id(PAD_TOKEN),
    vocab_size=tokenizer.get_vocab_size(),
    max_len=config.max_len,
    d_model=config.d_model,
    d_hidden=config.d_hidden,
    n_head=config.n_head,
    n_layer=config.n_layer,
    p_dropout=config.p_dropout
).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_state_dict_shapes_and_names(model):
    # This part helped me figure out that I don't have positional encodings saved in the state dict
    print(model.state_dict().keys())

    # This part helped me see that src MHA was missing in the decoder since both it and trg MHA were referencing
    # the same MHA object in memory - stupid mistake, happens all the time, embrace the suck!
    for name, param in model.named_parameters():
        print(name, param.shape)
        if not param.requires_grad:
            raise Exception('Expected all of the params to be trainable - no param freezing used.')


print(f"The model has {count_parameters(model):,} trainable parameters")

analyze_state_dict_shapes_and_names(model)

summary(model, [(20, ), (20, )], dtypes=[torch.long, torch.long])


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

    # Use tensorboard to plot curves, e.g. perplexity, accuracy, learning rate

    print("[Info] Use Tensorboard")

    current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    log_dir = os.path.join(config.base_dir, current_time)

    tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))

    log_train_file = os.path.join(log_dir, 'train.log')
    log_valid_file = os.path.join(log_dir, 'valid.log')

    print('[Info] Training performance will be written to file: {} and {}'.format(
        log_train_file, log_valid_file))

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,ppl\n')
        log_vf.write('epoch,loss,ppl,bleu\n')

    # 重加载
    restore_ckpt_path = config.checkpoint_dir / "en_de_B.pth"

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
            f"Epoch {epoch + 1}/{config.epochs}, Training Loss: {avg_train_loss: .4f}, Validation Loss: {avg_valid_loss:.4f}, BLEU Score: {avg_bleu:.2f}, lr: {scheduler.get_lr()[0]:.1e}"
        )

        checkpoint_path = config.checkpoint_dir / "en_de_B.pth"

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

        train_ppl = math.exp(min(avg_train_loss, 100))
        valid_ppl = math.exp(min(avg_valid_loss, 100))

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f}\n'.format(
                epoch=epoch, loss=avg_train_loss,
                ppl=train_ppl))
            log_vf.write('{epoch}, {loss: 8.5f}, {ppl: 8.5f}, {bleu: .2f}\n'.format(
                epoch=epoch, loss=avg_valid_loss,
                ppl=valid_ppl, bleu=avg_bleu))

        tb_writer.add_scalars('ppl', {'train': train_ppl, 'val': valid_ppl}, epoch)
        tb_writer.add_scalar('learning_rate', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('bleu', avg_bleu, epoch)

if __name__ == '__main__':
    training_loop(-1)
