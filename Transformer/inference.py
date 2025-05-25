import torch
import config

from data import load_data, PAD_TOKEN
from modules.transformer import Transformer
from models.model import Transformer
from utils import translate_sentence
import sacrebleu

from altair_heatmap import viz_encoder_self, viz_decoder_self, viz_decoder_src

# settings
restore_epoch = 4
num_sample = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

src_lang = 'en'
tgt_lang = 'de'

tokenizer, test_loader = load_data(src_lang, tgt_lang, ["test"])

test_dataset = test_loader.dataset

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


# load model
state_dict = torch.load(config.checkpoint_dir / "en_de_.pth")
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



# attn_vis
src_tokens = samples[src_lang][1]
src_tokens = tokenizer.encode(src_tokens).tokens
print(src_tokens)

tgt_tokens = pred["greedy-search"][1]
tgt_tokens = tokenizer.encode(tgt_tokens).tokens
print(tgt_tokens)
viz_encoder_self(model, src_tokens).save('attention_vis/enc_self_attn.html')

viz_decoder_self(model, tgt_tokens).save('attention_vis/dec_self_attn.html')

viz_decoder_src(model, tgt_tokens, src_tokens).save('attention_vis/dec_enc_attn.html')