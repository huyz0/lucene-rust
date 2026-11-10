#!/usr/bin/env python3
"""Builds the tiny ONNX text-embedding models under fixtures/ml/ (random
weights, BERT and DistilBERT shapes) that lucene-ml's tests and the plugin's
ml e2e run, and their ml-commons model zips.

Needs torch, transformers and tokenizers (pip). The expected vectors come from
a stock OpenSearch node: scripts/ml-golden.py.

Usage: scripts/gen-ml-fixture.py [out_dir]   (default fixtures/ml)
"""
import hashlib
import os
import sys
import zipfile

import torch
from tokenizers import BertWordPieceTokenizer
from transformers import BertConfig, BertModel, DistilBertConfig, DistilBertModel

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
    args = tuple(ids if n != "token_type_ids" else torch.zeros_like(ids) for n in names)
    dyn = {n: {0: "b", 1: "s"} for n in names}
    for o in out:
        dyn[o] = {0: "b", 1: "s"} if o == "last_hidden_state" else {0: "b"}
    torch.onnx.export(module, args, path, input_names=list(names), output_names=list(out),
                      dynamic_axes=dyn, opset_version=15, dynamo=False)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "fixtures/ml"
    os.makedirs(out, exist_ok=True)
    corpus = os.path.join(out, "corpus.txt")
    with open(corpus, "w") as f:
        f.write("\n".join(CORPUS * 4))
    tok = BertWordPieceTokenizer(lowercase=True, strip_accents=True)
    tok.train([corpus], vocab_size=400, min_frequency=1)
    os.remove(corpus)
    tok.save(os.path.join(out, "tokenizer.json"))
    vocab = tok.get_vocab_size()
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

    three = ("input_ids", "attention_mask", "token_type_ids")
    models = {
        "tiny-bert": (Bert(bert, False), three, ("last_hidden_state",)),
        "tiny-bert-pooled": (Bert(bert, True), three, ("last_hidden_state", "sentence_embedding")),
        "tiny-distilbert": (Distil(distil), three[:2], ("last_hidden_state",)),
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


if __name__ == "__main__":
    main()
