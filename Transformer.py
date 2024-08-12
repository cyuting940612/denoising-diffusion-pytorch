import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
from torch.nn.functional import log_softmax, pad
import time
import spacy
import os
from os.path import exists
import torchtext.datasets as datasets
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import DataProcess_Transformer as dpt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, TensorDataset, random_split




if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA GPU is available. Using CUDA device.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS backend is available. Using MPS device.")
else:
    device = torch.device("cpu")
    print("CUDA and MPS are not available. Using CPU.")



class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def make_model(
    src_vocab, tgt_vocab, N=8, d_model=128, d_ff=512, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def load_tokenizers():

    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en

def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]

def safe_tokenize(text, tokenizer, replacement_token="<unk>"):
    tokens = []
    for tok in tokenizer.tokenizer(text):
        try:
            tok.text.encode('utf-8')
            tokens.append(tok.text)
        except UnicodeDecodeError:
            tokens.append(replacement_token)
    return tokens


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


# def safe_iterator(iterator, encoding='utf-8', errors='ignore'):
#     for line in iterator:
#         try:
#             yield line.decode(encoding, errors=errors).strip()
#         except AttributeError:
#             yield line.strip()
def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return safe_tokenize(text, spacy_de)

    def tokenize_en(text):
        return safe_tokenize(text, spacy_en)

    print("Building German Vocabulary ...")
    train = datasets.Multi30k(root='.data', split='train', language_pair=('de', 'en'))
    val = datasets.Multi30k(root='.data', split='valid', language_pair=('de', 'en'))

    # train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    train = datasets.Multi30k(root='.data', split='train', language_pair=('de', 'en'))
    val = datasets.Multi30k(root='.data', split='valid', language_pair=('de', 'en'))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0])  # <s> token id
    eos_id = torch.tensor([1])  # </s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)

def create_dataloaders(
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
):
    # def create_dataloaders(batch_size=12000):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_iter = datasets.Multi30k(root='.data', split='train', language_pair=('de', 'en'))
    valid_iter = datasets.Multi30k(root='.data', split='valid', language_pair=('de', 'en'))

    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss


def data_gen(V, batch_size, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)

def create_dataloaders(
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
):
    # def create_dataloaders(batch_size=12000):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_iter, valid_iter, test_iter = datasets.Multi30k(
        language_pair=("de", "en")
    )

    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader

class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=1110):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            # self.tgt = tgt[:-1]
            # self.tgt_y = tgt[1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

    def to(self, device):
        """Move the batch to the specified device."""
        self.src = self.src.to(device)
        self.src_mask = self.src_mask.to(device)
        if hasattr(self, 'tgt'):
            self.tgt = self.tgt.to(device)
            self.tgt_y = self.tgt_y.to(device)
            self.tgt_mask = self.tgt_mask.to(device)
        if hasattr(self, 'ntokens'):
            self.ntokens = self.ntokens.to(device)
        return self
class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed

def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0

    # # Convert the iterator to a list to count the length
    # data_list = list(data_iter)
    # total_batches = len(data_list)
    # print("Total number of batches:", total_batches)

    for i, batch in enumerate(data_iter):
        batch.to(device)
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )


        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None

def train_worker(
    train_dataloader,
    valid_dataloader,
    config,
    is_distributed=False,
):
    # print(f"Train worker process using GPU: {gpu} for training", flush=True)
    # torch.cuda.set_device(gpu)

    # pad_idx = vocab_tgt["<blank>"]
    pad_idx = 1110
    d_model = 64
    model = make_model(1112, 1112, N=8).to(device)
    # model.cuda(gpu)
    module = model
    is_main_process = True
    if is_distributed:
        print("break")
        # dist.init_process_group(
        #     "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        # )
        # model = DDP(model, device_ids=[gpu])
        # module = model.module
        # is_main_process = gpu == 0

    criterion = LabelSmoothing(
        size=1112, padding_idx=pad_idx, smoothing=0.1
    )
    # criterion.cuda(gpu)

    # train_dataloader, valid_dataloader = create_dataloaders(
    #     vocab_src=vocab_src,
    #     vocab_tgt=vocab_tgt,
    #     spacy_de = spacy_de,
    #     spacy_en=spacy_en,
    #     batch_size=config["batch_size"],
    #     max_padding=config["max_padding"],
    #     is_distributed=is_distributed,
    # )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        # if is_distributed:
        #     train_dataloader.sampler.set_epoch(epoch)
        #     valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        # GPUtil.showUtilization()
        # if is_main_process:
        file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
        torch.save(module.state_dict(), file_path)
        # torch.cuda.empty_cache()

        # print(f"[Epoch {epoch} Validation ====", flush=True)
        # model.eval()
        # sloss = run_epoch(
        #     (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
        #     model,
        #     SimpleLossCompute(module.generator, criterion),
        #     DummyOptimizer(),
        #     DummyScheduler(),
        #     mode="eval",
        # )
        # print(sloss)
        # torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal_large.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)

config = {
    "batch_size": 32,
    "distributed": False,
    "num_epochs": 2,
    "accum_iter": 10,
    "base_lr": 1.0,
    "max_padding": 97,
    "warmup": 3000,
    "file_prefix": "multi30k_model_",
}


# def load_trained_model():
#
#     model = make_model(1112, 1112, N=6)
#     model.load_state_dict(torch.load("multi30k_model_final.pt"))
#     return model

# def inference_test():
#
#     test_model = load_trained_model()
#     test_model.eval()
#     src = torch.LongTensor([[0,1,2,3,4,5,6,7,8,9,10]])
#     src_mask = torch.ones(1, 1, 11)
#
#     ys = 40*torch.ones(1, 1).type_as(src)
#
#     for i in range(97):
#         memory = test_model.encode(src, src_mask)
#         out = test_model.decode(
#             memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
#         )
#         prob = test_model.generator(out[:, -1])
#         _, next_word = torch.max(prob, dim=1)
#         next_word = next_word.data[0]
#         ys = torch.cat(
#             [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
#         )
#         # src = torch.cat(
#         #     [src, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
#         # )
#         # src_mask = torch.ones(1, 1, 2+i+1)
#
#     print("Example Untrained Model Prediction:", ys)


# def run_tests():
#     for _ in range(1):
#         inference_test()


class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        label_data = self.labels[idx]
        return torch.tensor(input_data, dtype=torch.long), torch.tensor(label_data, dtype=torch.long)

# Example data
input, target = dpt.data_process()  # Replace with your actual data
dataset = CustomDataset(input,target)
# dataset = TensorDataset(X_tensor_week, y_tensor_week)
# dataset = TensorDataset(input, target)

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

training_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
testing_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# spacy_de,spacy_en = load_tokenizers()
# vocab_src, vocab_tgt =load_vocab(spacy_de, spacy_en)

# train_worker(training_dataloader,testing_dataloader, config=config, is_distributed=False)
# run_tests()

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        monitor = prob.detach().numpy()
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys

def top_k_sampling(prob, k):
    top_k_prob, top_k_indices = torch.topk(prob, k)
    top_k_prob = F.softmax(top_k_prob, dim=-1)
    next_word = top_k_indices[torch.multinomial(top_k_prob, num_samples=1)]
    return next_word.item()

def decode_with_top_k_sampling(model, src, src_mask, max_len, start_symbol, k=2):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        prob = torch.squeeze(prob)
        next_word = top_k_sampling(prob, k)
        ys = torch.cat(
            [ys, torch.tensor([[next_word]]).type_as(src.data)], dim=1
        )
    return ys


def top_p_sampling(prob, p=0.9):
    sorted_prob, sorted_indices = torch.sort(prob, descending=True)
    cumulative_prob = torch.cumsum(F.softmax(sorted_prob, dim=-1), dim=-1)
    top_p_indices = sorted_indices[cumulative_prob <= p]

    # Ensure at least one token is selected
    if len(top_p_indices) == 0:
        top_p_indices = sorted_indices[:1]
        top_p_prob = F.softmax(sorted_prob[:1], dim=-1)
    else:
        top_p_prob = F.softmax(sorted_prob[cumulative_prob <= p], dim=-1)

    next_word = top_p_indices[torch.multinomial(top_p_prob, num_samples=1)]
    return next_word.item()
def decode_with_top_p_sampling(model, src, src_mask, max_len, start_symbol, p=0.9):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        prob = torch.squeeze(prob)
        next_word = top_p_sampling(prob, p)
        ys = torch.cat(
            [ys, torch.tensor([[next_word]]).type_as(src.data)], dim=1
        )
    return ys

def check_outputs(
    valid_dataloader,
    model,
    n_examples=15,
    pad_idx=1110,
):
    results = [()] * n_examples
    result_list = []
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        # rb = Batch(b[0].reshape(1,97), b[1].reshape(1,97), pad_idx)
        # # greedy_decode(model, rb.src, rb.src_mask, 97, 40)[0]
        #
        # src_tokens = [
        #    x for x in rb.src[0]
        # ]
        # tgt_tokens = [
        #     x for x in rb.tgt[0]
        # ]
        #
        # print(
        #     "Source Text (Input)        : "
        #     + str(src_tokens)
        # )
        # print(
        #     "Target Text (Ground Truth) : "
        #     + str(tgt_tokens)
        # )
        # model_out = greedy_decode(model, rb.src, rb.src_mask, 97, 40)[0]
        # print("Model Output               : " + str(model_out))
        # results[idx] = (rb, src_tokens, tgt_tokens, model_out)

        input_tensor = torch.LongTensor([[0]])
        target_tensor = torch.LongTensor([[40]])
        # input_tensor = b[0].view(1, 97)
        # target_tensor = b[1].view(1,97)

        rb = Batch(input_tensor, target_tensor, pad_idx)
        # greedy_decode(model, rb.src, rb.src_mask, 97, 40)[0]

        # src_tokens = [
        #    x for x in rb.src[0]
        # ]
        # tgt_tokens = [
        #     x for x in rb.tgt[0]
        # ]

        # print(
        #     "Source Text (Input)        : "
        # )
        # for i, tensor in enumerate(src_tokens):
        #     print(tensor)
        # print(
        #     "Target Text (Ground Truth) : "
        # )
        # for i, tensor in enumerate(tgt_tokens):
        #     print(tensor)
        for i in range(1,97):
            model_out = greedy_decode(model, rb.src, rb.src_mask, i+1, 8)
            next_int = model_out[0,-1]
            rb.src = torch.cat(
                [rb.src, torch.zeros(1, 1).type_as(rb.src.data).fill_(next_int)], dim=1
            )
            new_value_tensor = torch.tensor([[[True]]])
            rb.src_mask = torch.cat((rb.src_mask, new_value_tensor), dim=-1)

        # model_txt = (
        #     " ".join(
        #         [vocab_tgt[x] for x in model_out]
        # )

        # print("Model Output               : " )
        # # for i, tensor in enumerate(model_out):
        # print(rb.src)
        result_list.append(rb.src)
    stacked_tensor = torch.stack(result_list)

    final_data = stacked_tensor.numpy()
    final_data = final_data[:,:,1:]
    final_data = final_data.reshape(96*n_examples, 1)
    file_path = 'transformer_weekday.csv'
    np.savetxt(file_path, final_data, delimiter=',', fmt='%.2f')

    results[idx] = (rb,  rb.src)


    return results


def run_model_example(n_examples=100):

    print("Preparing Data ...")

    print("Loading Trained Model ...")

    model = make_model(1112, 1112, N=8)
    model.load_state_dict(
        torch.load("multi30k_model_final_large.pt", map_location=torch.device("mps"))
    )

    print("Checking Model Outputs:")
    example_data = check_outputs(
        test_dataset, model, n_examples=n_examples
    )
    return model, example_data


run_model_example()









# class Config:
#     vocab_size = 30522
#     n_positions = 512
#     n_ctx = 512
#     n_embd = 768
#     n_layer = 12
#     n_head = 12
#
# config = Config()
#
# class Attention(nn.Module):
#     def __init__(self, config):
#         super(Attention, self).__init__()
#         self.n_head = config.n_head
#         self.split_size = config.n_embd
#         self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
#         self.c_proj = nn.Linear(config.n_embd, config.n_embd)
#         self.attn_dropout = nn.Dropout(0.1)
#         self.resid_dropout = nn.Dropout(0.1)
#
#     def split_heads(self, x, k=False):
#         new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
#         x = x.view(*new_x_shape)
#         if k:
#             return x.permute(0, 2, 3, 1)
#         else:
#             return x.permute(0, 2, 1, 3)
#
#     def forward(self, x, layer_past=None):
#         x = self.c_attn(x)
#         query, key, value = x.split(self.split_size, dim=2)
#         query = self.split_heads(query)
#         key = self.split_heads(key, k=True)
#         value = self.split_heads(value)
#
#         attn_weights = torch.matmul(query, key) / (float(value.size(-1)) ** 0.5)
#         attn_weights = nn.Softmax(dim=-1)(attn_weights)
#         attn_weights = self.attn_dropout(attn_weights)
#
#         attn = torch.matmul(attn_weights, value)
#         attn = attn.permute(0, 2, 1, 3).contiguous()
#         attn = attn.view(*attn.size()[:-2], self.split_size)
#         attn = self.c_proj(attn)
#         attn = self.resid_dropout(attn)
#         return attn
#
# class Block(nn.Module):
#     def __init__(self, config):
#         super(Block, self).__init__()
#         self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-5)
#         self.attn = Attention(config)
#         self.ln_2 = nn.LayerNorm(config.n_embd, eps=1e-5)
#         self.mlp = nn.Sequential(
#             nn.Linear(config.n_embd, 4 * config.n_embd),
#             nn.GELU(),
#             nn.Linear(4 * config.n_embd, config.n_embd),
#             nn.Dropout(0.1),
#         )
#
#     def forward(self, x):
#         a = self.ln_1(x)
#         x = x + self.attn(a)
#         m = self.ln_2(x)
#         m = self.mlp(m)
#         x = x + m
#         return x
#
#
# class TransformerModel(nn.Module):
#     def __init__(self, config):
#         super(TransformerModel, self).__init__()
#         # self.label_embedding = nn.Embedding(config.n_label, config.n_embd)
#         self.wte = nn.Embedding(config.vocab_size, config.n_embd)
#         self.wpe = nn.Embedding(config.n_positions, config.n_embd)
#         self.drop = nn.Dropout(0.1)
#         self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
#         self.ln_f = nn.LayerNorm(config.n_embd, eps=1e-5)
#         self.decoder = nn.Linear(config.n_embd, config.vocab_size)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, input_ids, position_ids=None):
#         if position_ids is None:
#             position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device)
#             position_ids = position_ids.unsqueeze(0).expand(input_ids.size(0), -1)
#         # label_emb = self.label_embedding(labels).unsqueeze(1)
#         # (1, batch_size, model_dim)
#         input_embeds = self.wte(input_ids)
#
#         position_embeds = self.wpe(position_ids)
#         hidden_states = input_embeds + position_embeds
#         # hidden_states = torch.cat([label_emb, hidden_states], dim=1)
#         hidden_states = self.drop(hidden_states)
#
#         for block in self.h:
#             hidden_states = block(hidden_states)
#
#         hidden_states = self.ln_f(hidden_states)
#         decoded = self.decoder(hidden_states)
#         output = self.softmax(decoded)
#
#         return output


