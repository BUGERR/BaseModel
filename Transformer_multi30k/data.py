import config
import os
import datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
from typing import Optional, Sequence
import torch
from torch.utils.data import DataLoader

SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"

special_tokens = [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN]


def build_tokenizer(dataset, force_reload):
    '''
    pipeline: check cache --> 空格预分词 --> 初始化 BpeTrainer --> 定义 batch_iterator
               --> train_from_iterator --> enable padding 和 trancation --> 后处理
    '''
    tokenizer_path = config.dataset_dir / f"tokenizer.json"
    if os.path.exists(tokenizer_path) and not force_reload:
        tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN)).from_file(str(tokenizer_path))
    else:
        tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            # vocab_size = 37000,
            min_frequency=2,
            show_progress=True,
            special_tokens=special_tokens
        )
        print('Training tokenizer ...')

        def batch_iterator(batch_size, dataset):
            train_data = dataset["train"]
            scaled_batch_size = batch_size // 2
            for i in range(0, train_data.num_rows, scaled_batch_size):
                batch = train_data[i: i + scaled_batch_size]
                yield batch['en'] + batch['de']

        tokenizer.train_from_iterator(batch_iterator(config.batch_size, dataset), trainer)
        tokenizer.enable_padding(
            pad_id=tokenizer.token_to_id(PAD_TOKEN),
            pad_token=PAD_TOKEN
        )
        tokenizer.enable_truncation(max_length=config.max_len)
        tokenizer.post_processor = TemplateProcessing(
            # 单句的模板
            single=f"{SOS_TOKEN} $A {EOS_TOKEN}",
            # 输入是成对句子的模版，一般用不到
            # pair=,
            # 指定特殊 token 和词库对应 id
            special_tokens=[
                (SOS_TOKEN, tokenizer.token_to_id(SOS_TOKEN)),
                (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN))
            ]
        )

        tokenizer.save(str(tokenizer_path))
    return tokenizer


def load_data(
        src_lang, tgt_lang, splits: Optional[Sequence[str]] = None, force_reload=False
):
    if sorted((src_lang, tgt_lang)) != ["de", "en"]:
        raise ValueError("Available language options are ('de','en') and ('en', 'de')")

    all_splits = ["train", "validation", "test"]
    if splits is None:
        splits = all_splits
    elif not set(splits).issubset(all_splits):
        raise ValueError(f"Splits should only contain some of {all_splits}")

    dataset = datasets.load_dataset("bentrevett/multi30k")

    tokenizer = build_tokenizer(dataset, force_reload)

    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for item in batch:
            src_batch.append(item[src_lang])
            tgt_batch.append(item[tgt_lang])

        src_batch = tokenizer.encode_batch(src_batch)
        tgt_batch = tokenizer.encode_batch(tgt_batch)

        src_tensor = torch.LongTensor([item.ids for item in src_batch])
        tgt_tensor = torch.LongTensor([item.ids for item in tgt_batch])

        return src_tensor, tgt_tensor

    dataloaders = [
        DataLoader(
            dataset=dataset[split],
            batch_size=config.batch_size,
            collate_fn=collate_fn,
            shuffle=split == 'train'
        )
        for split in splits
    ]

    return (tokenizer, *dataloaders)


