from data import load_data, PAD_TOKEN
import torch
import config
from models.model import Transformer_my

from utils import translate_sentence
import sacrebleu

src_lang = "en"
tgt_lang = "de"

tokenizer, test_loader = load_data(src_lang, tgt_lang, ["test"])
device = torch.device("cuda:0")
dataset = test_loader.dataset

model = Transformer_my(
    pad_idx=tokenizer.token_to_id(PAD_TOKEN),
    vocab_size=tokenizer.get_vocab_size(),
    max_len=config.max_len,
    d_model=config.d_model,
    d_hidden=config.d_hidden,
    n_head=config.n_head,
    n_layer=config.n_layer,
    p_dropout=config.p_dropout
)

state_dict = torch.load(config.checkpoint_dir / "en_de_L.pth")
model.load_state_dict(state_dict["model"])

num_sample = 5
samples = dataset[torch.randint(0, len(dataset), (num_sample,))]["translation"]

method = {
    "greedy-search": {"num_beams": 1, "do_sample": False},
    "sample": {
        "num_beams": 1,
        "do_sample": True,
        "top_k": config.top_k,
        "top_p": config.top_p,
        "temperature": config.temperature,
    },
}


# print([samples[i][src_lang] for i in range(num_sample)])
# print([[samples[i][tgt_lang]] for i in range(num_sample)])

pred = {
    method_name: translate_sentence(
        [samples[i][src_lang] for i in range(num_sample)], model, tokenizer, tokenizer, **args
    )
    for method_name, args in method.items()
}

references = [[samples[i][tgt_lang]] for i in range(num_sample)]

# Calculate BLEU scores for each method
bleu_scores = {
    method_name: sacrebleu.corpus_bleu(
        pred_list, references
    ).score
    for method_name, pred_list in pred.items()
}

for i in range(num_sample):
    print(f"\033[1mThe {i+1}th source sentence\033[0m: {''.join(samples[i][src_lang])}")
    print(f"\033[1mGround Truth\033[0m: {''.join(samples[i][tgt_lang])}")
    for method_name in method.keys():
        print(f"\033[1m{method_name}\033[0m: {pred[method_name][i]}")
    print()

# Print BLEU scores
for method_name, score in bleu_scores.items():
    print(f"\033[1mBLEU score for {method_name}\033[0m: {score:.2f}")
