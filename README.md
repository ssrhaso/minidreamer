# microjpt

A pure Julia port of Karpathy's [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) - a single-file, zero-dependency GPT that trains and generates text from scratch. Built to explore Julia's mathematical expressiveness for ML: native matrix operations, manual backprop, no autograd, no frameworks.

### Files

| File            | Lines | Description                                         |
| --------------- | ----- | --------------------------------------------------- |
| `microjpt.jl` | 150   | Fully commented version - every operation annotated |
| `nanojpt.jl`  | 99    | Minimal version - same algorithm, no comments       |

Both files contain the complete algorithm: tokenizer, RMSNorm, multi-head causal attention, ReLU MLP, manual matrix backpropagation, Adam optimizer, and temperature-controlled inference.

### Running

Requires Julia 1.9+. No packages to install -  `Random` and `Downloads` are part of the standard library. On first run the dataset is downloaded automatically, if you choose you can replace this dataset with any other to test the application further. Open a Julia REPL and run:

```julia
include("microjpt.jl")    # annotated
include("nanojpt.jl")    # compact
```

### Benchmarks

Independent benchmarks by [@Entrpi](https://github.com/Entrpi) on an Apple M5 (single P-core), using the Karpathy names dataset with `n_head=4`.

**d16, names.txt, 10K training samples**

| Implementation                                                                                     | µs/sample   | Speedup vs CPython |
| -------------------------------------------------------------------------------------------------- | ------------ | ------------------ |
| [CPython 3.14](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) (autograd, batch=1) | 49,000       | 1×                |
| [microgpt.cpp](https://github.com/Charbel199/microgpt.cpp) (autograd, batch=1)                        | 270          | 181×              |
| [rust-microgpt](https://github.com/mplekh/rust-microgpt) (autograd, batch=1)                         | 118          | 415×              |
| **microjpt (explicit, batch=1)**                                                             | **31** | **1,581×**  |
| [EEmicroGPT](https://github.com/Entrpi/eemicrogpt) (explicit, batch=16)                               | 3.0          | 16,333×           |

**d64, names.txt, 1K training samples**

| Implementation                                                                                     | µs/sample    | Speedup vs CPython |
| -------------------------------------------------------------------------------------------------- | ------------- | ------------------ |
| [CPython 3.14](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) (autograd, batch=1) | 713,200       | 1×                |
| [rust-microgpt](https://github.com/mplekh/rust-microgpt) (autograd, batch=1)                          | 1,620         | 440×              |
| **microjpt (explicit, batch=1)**                                                             | **294** | **2,425×**  |
| [EEmicroGPT](https://github.com/Entrpi/eemicrogpt) SME2 (explicit, batch=16)                          | 36.8          | 19,380×           |

microjpt is the fastest batch=1 implementation with **3.8× faster than rust-microgpt at d16** and **5.5× faster at d64**. The explicit backprop collapses ~57K autograd tape nodes into ~20 matrix operations, the same insight driving EEmicroGPT's performance. The remaining gap to EEmicroGPT comes from batching (16 samples), f32 vs f64, and hand-optimized Neon/SME2 SIMD.

> 100 lines of dependency-free Julia, 1,581× faster than CPython, and only 7.6× off hand-tuned C.

### Blog

For a deeper walkthrough of the math behind each operation, see the companion post: (WIP)
