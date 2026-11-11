#!/usr/bin/env python3
"""Builds the tiny ONNX models under fixtures/ml/ (random weights) that
lucene-ml's tests and the plugin's ml e2e run, and their ml-commons model
zips: text embedding (BERT and DistilBERT shapes), sparse encoding (a masked
LM head max-pooled over tokens) and a cross-encoder. The same weights are also
traced as TorchScript (`.pt`), in the shapes ml-commons' PyTorch translators
feed and read: a sentence-transformers model (a dict of arrays in and out), a
Hugging Face model (positional arrays, a tuple out), a sparse model (a dict
in, `{"output": ...}` out) and a cross-encoder (positional, a tuple out).

An existing tokenizer.json is kept, so models added later share the
vocabulary the recorded vectors were made with.

Needs torch, transformers and tokenizers (pip). The expected vectors come from
a stock OpenSearch node: scripts/ml-golden.py.

Usage: scripts/gen-ml-fixture.py [out_dir]   (default fixtures/ml)
"""
import hashlib
import json
import os
import sys
import zipfile
from typing import Dict

import torch
from tokenizers import BertWordPieceTokenizer
from transformers import (BertConfig, BertForMaskedLM, BertForSequenceClassification, BertModel,
                          DistilBertConfig, DistilBertModel)

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "neural search finds documents by meaning rather than by exact words",
    "a model turns text into a dense vector of numbers",
    "opensearch indexes documents into shards and segments",
    "lucene reads postings, doc values and stored fields",
    "machine learning models are registered, deployed and used for prediction",
    "café naïve résumé über straße",
]


def export(module, names, out, path):
    ids = torch.ones(1, 6, dtype=torch.long)
    args = tuple(torch.zeros_like(ids) if n == "token_type_ids" else ids for n in names)
    dyn = {n: {0: "b", 1: "s"} for n in names}
    for o in out:
        dyn[o] = {0: "b", 1: "s"} if o == "last_hidden_state" else {0: "b"}
    torch.onnx.export(module, args, path, input_names=list(names), output_names=list(out),
                      dynamic_axes=dyn, opset_version=15, dynamo=False)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "fixtures/ml"
    os.makedirs(out, exist_ok=True)
    tokenizer = os.path.join(out, "tokenizer.json")
    if not os.path.exists(tokenizer):
        corpus = os.path.join(out, "corpus.txt")
        with open(corpus, "w") as f:
            f.write("\n".join(CORPUS * 4))
        tok = BertWordPieceTokenizer(lowercase=True, strip_accents=True)
        tok.train([corpus], vocab_size=400, min_frequency=1)
        os.remove(corpus)
        tok.save(tokenizer)
    vocab = len(json.load(open(tokenizer))["model"]["vocab"])
    torch.manual_seed(11)

    bert = BertModel(BertConfig(vocab_size=vocab, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                                intermediate_size=64, max_position_embeddings=512), add_pooling_layer=False).eval()

    class Bert(torch.nn.Module):
        def __init__(self, m, pooled):
            super().__init__()
            self.m, self.pooled = m, pooled

        def forward(self, input_ids, attention_mask, token_type_ids):
            h = self.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
            if not self.pooled:
                return h
            m = attention_mask.unsqueeze(-1).float()
            return h, (h * m).sum(1) / m.sum(1)

    distil = DistilBertModel(DistilBertConfig(vocab_size=vocab, dim=32, n_layers=2, n_heads=4, hidden_dim=64,
                                              max_position_embeddings=512)).eval()

    class Distil(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    small = dict(vocab_size=vocab, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                 intermediate_size=64, max_position_embeddings=512)
    mlm = BertForMaskedLM(BertConfig(**small)).eval()

    class Sparse(torch.nn.Module):
        """Per-vocabulary weights: max over the text's tokens of log(1 + relu(logit)), shifted so most
        are zero."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            logits = self.m(input_ids=input_ids, attention_mask=attention_mask).logits - 0.15
            w = torch.log1p(torch.relu(logits)) * attention_mask.unsqueeze(-1).float()
            return w.max(dim=1).values

    cross = BertForSequenceClassification(BertConfig(num_labels=1, **small)).eval()

    class Cross(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits

    three = ("input_ids", "attention_mask", "token_type_ids")
    models = {
        "tiny-bert": (Bert(bert, False), three, ("last_hidden_state",)),
        "tiny-bert-pooled": (Bert(bert, True), three, ("last_hidden_state", "sentence_embedding")),
        "tiny-distilbert": (Distil(distil), three[:2], ("last_hidden_state",)),
        # The input names ml-commons' SentenceTransformerTranslator gives a sparse encoding model.
        "tiny-sparse": (Sparse(mlm), ("input1.input_ids", "input1.attention_mask"), ("output",)),
        "tiny-cross-encoder": (Cross(cross), three, ("logits",)),
    }
    for name, (module, names, outputs) in models.items():
        onnx = os.path.join(out, f"{name}.onnx")
        export(module, names, outputs, onnx)
        z = os.path.join(out, f"{name}.zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
            f.write(onnx, f"{name}.onnx")
            f.write(os.path.join(out, "tokenizer.json"), "tokenizer.json")
        digest = hashlib.sha256(open(z, "rb").read()).hexdigest()
        print(name, os.path.getsize(z), digest)

    class SentenceTransformer(torch.nn.Module):
        """What sentence-transformers' traced models return: the features dict, the arrays it was given
        and the token and sentence embeddings."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, features: Dict[str, torch.Tensor]):
            ids, mask = features["input_ids"], features["attention_mask"]
            h = self.m(input_ids=ids, attention_mask=mask, token_type_ids=torch.zeros_like(ids)).last_hidden_state
            m = mask.unsqueeze(-1).float()
            return {"input_ids": ids, "attention_mask": mask, "token_embeddings": h,
                    "sentence_embedding": (h * m).sum(1) / m.sum(1)}

    class Positional(torch.nn.Module):
        def __init__(self, m, n):
            super().__init__()
            self.m, self.n = m, n

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            if self.n == 2:
                return (self.m(input_ids, attention_mask, torch.zeros_like(input_ids)),)
            return (self.m(input_ids, attention_mask, token_type_ids),)

    class DictSparse(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input: Dict[str, torch.Tensor]):
            return {"output": self.m(input["input_ids"], input["attention_mask"])}

    ids = torch.ones(1, 6, dtype=torch.long)
    traced = {
        "tiny-st": (SentenceTransformer(bert), ({"input_ids": ids, "attention_mask": ids},)),
        "tiny-hf": (Positional(Bert(bert, False), 2), (ids, ids)),
        "tiny-sparse-ts": (DictSparse(Sparse(mlm)), ({"input_ids": ids, "attention_mask": ids},)),
        "tiny-cross-ts": (Positional(Cross(cross), 3), (ids, ids, torch.zeros_like(ids))),
    }
    for name, (module, args) in traced.items():
        pt = os.path.join(out, f"{name}.pt")
        with torch.no_grad():
            torch.jit.trace(module.eval(), args, strict=False).save(pt)
        z = os.path.join(out, f"{name}.zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
            f.write(pt, f"{name}.pt")
            f.write(os.path.join(out, "tokenizer.json"), "tokenizer.json")
        digest = hashlib.sha256(open(z, "rb").read()).hexdigest()
        print(name, os.path.getsize(z), digest)


if __name__ == "__main__":
    main()
