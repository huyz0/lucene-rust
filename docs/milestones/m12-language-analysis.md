# M12 — Language analysis modules

> **Goal:** Lucene's language-specific analysis modules -- ICU, Japanese,
> Korean, Chinese, Polish, dictionary morphology, phonetic and OpenNLP --
> with the same tokens as Java.

| | |
|---|---|
| **Effort** | L |
| **Depends on** | [M11](m11-analysis-common.md) |
| **Unblocks** | analysing CJK and other scripts without a JVM |
| **Status** | not started |

---

## Why this milestone exists

These modules are small in code but heavy in data (and, for ICU and OpenNLP,
in external libraries). Sizes from the 10.5.0 sources jars:

| Module | Files | Lines of code | What it carries |
|---|---|---|---|
| `analysis-icu` | 26 | 1,230 | ICU4J normalisation, folding, script-aware tokenization, collation keys |
| `analysis-kuromoji` | 59 | 5,015 | Japanese morphological analysis over the IPADIC dictionary |
| `analysis-nori` | 40 | 2,523 | Korean morphological analysis over mecab-ko-dic |
| `analysis-smartcn` | 21 | 1,399 | Chinese word segmentation (HMM, bundled dictionary) |
| `analysis-stempel` | 21 | 2,093 | Polish stemming tables |
| `analysis-morfologik` | 9 | 376 | FSA dictionaries (Morfologik) |
| `analysis-phonetic` | 10 | 449 | Commons Codec phonetic encoders |
| `analysis-opennlp` | 20 | 998 | OpenNLP models (sentence, POS, chunk, lemmatize) |

---

## Scope

### In scope

- All eight modules, reading the same dictionary and model files Lucene
  ships or loads, so a user's existing dictionaries work unchanged.
- ICU through a Rust ICU implementation (ICU4X), with a differential test
  proving the output matches ICU4J's for the operations Lucene uses; where
  it cannot, the gap is recorded per operation.

### Out of scope

- Training models or building dictionaries; only loading and using them.

---

## Tasks

- **T12.1** — Kuromoji and Nori: the Viterbi lattice, user dictionaries,
  n-best output, search-mode decompounding.
- **T12.2** — SmartCN and Stempel.
- **T12.3** — Morfologik FSA reading and phonetic encoders (reimplementing
  Commons Codec's algorithms, each against its Java output).
- **T12.4** — ICU over ICU4X.
- **T12.5** — OpenNLP: model loading and inference, or a recorded decision
  that it stays JVM-only if no faithful Rust runtime exists.

## Acceptance criteria

- [ ] Each module produces Lucene's token stream on a native-language corpus
      for its language(s), every attribute.
- [ ] Kuromoji and Nori match Lucene's n-best lattices, not just the best
      path.
- [ ] Every bundled dictionary and model is licence-audited and in `NOTICE`.
- [ ] Each analyzer is no slower than Lucene's.

## Risks and unknowns

- **ICU4X versus ICU4J.** The two implement the same Unicode standards but
  are separate codebases; differences in normalisation or break iteration
  are possible and must be found by the differential test, not assumed away.
- **OpenNLP has no Rust runtime.** T12.5 may end as a documented decision
  rather than a port.

## Exit artifacts

- Per-language differential corpora and generators
- `docs/parity.md` rows for all eight modules
