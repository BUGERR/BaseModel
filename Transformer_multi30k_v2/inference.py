import torch
import config

from data import load_data, PAD_TOKEN
from modules.transformer import Transformer
from models.model import Transformer_my
from utils import translate_sentence
import sacrebleu

from utils import  greedy_search, sample
from tqdm import tqdm

import torch.nn as nn
import data

# from altair_heatmap import viz_encoder_self, viz_decoder_self, viz_decoder_src

# settings
restore_epoch = 4
num_sample = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

src_lang = 'en'
tgt_lang = 'de'

tokenizer, test_loader = load_data(src_lang, tgt_lang, ["test"])

test_dataset = test_loader.dataset

model = Transformer(
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

# model = Transformer_my(
#     pad_idx=tokenizer.token_to_id(PAD_TOKEN),
#     vocab_size=tokenizer.get_vocab_size(),
#     max_len=config.max_len,
#     d_model=config.d_model,
#     d_hidden=config.d_hidden,
#     n_head=config.n_head,
#     n_layer=config.n_layer,
#     p_dropout=config.p_dropout
# )


# load model
state_dict = torch.load(config.checkpoint_dir / "en_de_L.pth")
model.load_state_dict(state_dict["model"])

# sample data
samples = test_dataset[torch.randint(0, len(test_dataset), (num_sample,))]

method = {
    "greedy-search": {"do_sample": False, },
    "sample": {
        "do_sample": True, 
        "top_k": config.top_k, 
        "top_p": config.top_p,
        "temperature": config.temperature, 
    },
}

pred = {
    method_name: translate_sentence(
        samples[src_lang], model, tokenizer, tokenizer, **args
    )
    for method_name, args in method.items()
}

gt = [[sentence] for sentence in samples[tgt_lang]]

# Calculate BLEU scores for each method
bleu_scores = {
    method_name: sacrebleu.corpus_bleu(
        pred_list, gt
    ).score
    for method_name, pred_list in pred.items()
}

for i in range(num_sample):
    print(f"\033[1mThe {i+1}th source sentence\033[0m: {''.join(samples[src_lang][i])}")
    print(f"\033[1mGround Truth\033[0m: {''.join(samples[tgt_lang][i])}")
    for method_name in method.keys():
        print(f"\033[1m{method_name}\033[0m: {pred[method_name][i]}")
    print()

# Print BLEU scores
for method_name, score in bleu_scores.items():
    print(f"\033[1mBLEU score for {method_name}\033[0m: {score:.2f}")


def split_batch(batch):
    src, tgt = batch
    tgt, gt = tgt[:, :-1], tgt[:, 1:]
    return src, tgt, gt

sos_idx = tokenizer.token_to_id(data.SOS_TOKEN)
eos_idx = tokenizer.token_to_id(data.EOS_TOKEN)
pad_idx = tokenizer.token_to_id(data.PAD_TOKEN)

@torch.no_grad()
def evaluate(model, criterion):
    model.eval()
    total_loss = 0
    all_references = []
    all_predictions = []

    for batch in tqdm(test_loader, desc="Evaluating"):
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
        # pred_tokens = sample(
        #     model, enc_output, src_mask, config.max_len, , sos_idx, eos_idx, pad_idx
        # )

        pred_sentence = tokenizer.decode_batch(pred_tokens.cpu().numpy())
        target_sentence = tokenizer.decode_batch(gt.cpu().numpy())
        all_predictions.append("".join(pred_sentence))
        all_references.append(["".join(target_sentence)])

    avg_loss = total_loss / len(test_loader)
    if len(all_predictions) > 0:
        bleu_score = sacrebleu.corpus_bleu(all_predictions, all_references)
        avg_bleu = bleu_score.score
    else:
        avg_bleu = 0
    return avg_loss, avg_bleu

criterion = nn.CrossEntropyLoss(
    ignore_index=tokenizer.token_to_id(PAD_TOKEN), label_smoothing=config.eps_ls
)
avg_valid_loss, avg_bleu = evaluate(model, criterion)
print(
    f"test Loss: {avg_valid_loss:.4f}, BLEU Score: {avg_bleu:.2f}"
)


# attn_vis
# src_tokens = samples[src_lang][1]
# src_tokens = tokenizer.encode(src_tokens).tokens
# print(src_tokens)
#
# tgt_tokens = pred["greedy-search"][1]
# tgt_tokens = tokenizer.encode(tgt_tokens).tokens
# print(tgt_tokens)
# viz_encoder_self(model, src_tokens).save('attention_vis/enc_self_attn.html')
#
# viz_decoder_self(model, tgt_tokens).save('attention_vis/dec_self_attn.html')
#
# viz_decoder_src(model, tgt_tokens, src_tokens).save('attention_vis/dec_enc_attn.html')

