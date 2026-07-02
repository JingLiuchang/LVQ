# LVQ

Standalone C++ implementation of an IVF residual scan path using
Locally-adaptive Vector Quantization (LVQ).

This repository was implemented for experiments around the paper
[Similarity search in the blink of an eye with compressed indices][lvq-paper]
by Aguerrebere et al., PVLDB 2023.  The paper introduces LVQ as a
per-vector scalar quantization method with local scaling and optional
second-level residual codes for higher-recall reranking.

This code is not the Intel SVS implementation and does not implement the
paper's graph index.  In our benchmark stack, Faiss is used to train an
`IVF{nlist},Flat` coarse quantizer and assign database vectors to coarse
centroids.  This repository then compresses and scans residual vectors
`x - centroid` with the LVQ storage and distance-computation ideas.

[lvq-paper]: https://www.vldb.org/pvldb/vol16/p3433-aguerrebere.pdf

## Repository Layout

```text
.
|-- CMakeLists.txt
|-- include/
|   `-- ivf_lvq.h          # Core LVQ data layout, encode/decode, and scan code
`-- src/
    |-- CMakeLists.txt
    `-- LVQ/
        |-- index.cpp      # Builds an LVQ IVF residual index
        `-- search.cpp     # Runs nprobe sweeps over an LVQ IVF residual index
```

Generated directories are intentionally ignored:

```text
build/                    # CMake build output
DATA/                     # Benchmark-generated fvecs/ivecs/index files
```

## Implemented Variants

The `--compression` option selects the primary and secondary LVQ code widths:

| Name | Primary bits | Residual bits | Notes |
|------|--------------|---------------|-------|
| `LVQ4x0` | 4 | 0 | 4-bit local scalar codes, no secondary rerank codes |
| `LVQ8x0` | 8 | 0 | 8-bit local scalar codes, no secondary rerank codes |
| `LVQ4x4` | 4 | 4 | 4-bit primary codes plus 4-bit residual codes |
| `LVQ4x8` | 4 | 8 | 4-bit primary codes plus 8-bit residual codes |
| `LVQ8x8` | 8 | 8 | 8-bit primary codes plus 8-bit residual codes |

For `LVQ*x0`, search directly returns the top `k` candidates from the primary
scan.  For variants with residual bits, search first keeps
`ceil(k * rerank_factor)` candidates from the primary scan and reranks them with
the secondary residual code.  The search binary clamps `rerank_factor` to at
least `1.0`.

## Data Contract

The index builder expects a source directory containing:

```text
base.fvecs                # Database vectors
centroids.fvecs           # IVF coarse centroids, one row per list
cluster_ids.ivecs         # One assigned coarse-list id per database vector
```

The search binary expects:

```text
query.fvecs               # Query vectors
groundtruth.ivecs         # Ground-truth neighbor ids for recall evaluation
```

The benchmark wrapper in the parent repository prepares these files from the
configured dataset and Faiss coarse quantizer.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces:

```text
build/src/lvq_index
build/src/lvq_search
```

The code uses C++17 and OpenMP.  It also includes the benchmark tree's RESQ
utility headers for `.fvecs`/`.ivecs` loading.

## Usage

Build an index:

```bash
build/src/lvq_index \
  -s DATA/deep1M/nlist4096_LVQ8x0/ \
  -c 4096 \
  --compression LVQ8x0 \
  -o DATA/deep1M/nlist4096_LVQ8x0/ivf_lvq_4096_LVQ8x0.index
```

Run an `nprobe` sweep:

```bash
build/src/lvq_search \
  -i DATA/deep1M/nlist4096_LVQ8x0/ivf_lvq_4096_LVQ8x0.index \
  -q DATA/deep1M/nlist4096_LVQ8x0/query.fvecs \
  -g DATA/deep1M/nlist4096_LVQ8x0/groundtruth.ivecs \
  -k 10 \
  -p 5,10,25,50,100 \
  -w 0 \
  -m 2 \
  --rerank-factor 10.0
```

The search binary writes tab-separated `RESULT` rows:

```text
RESULT  nprobe  recall  qps  scanned  reranked
```

## Benchmark Integration

This repository is consumed as a submodule by our quantization benchmark:

- Benchmark repository: <https://github.com/JingLiuchang/bench>
- Current integration branch: `feature/p2-e2e`
- Parent-side entry point: `scripts/run_p2_lvq.py`
- Hydra config: `conf/method_p2/lvq.yaml`
- Batch wrapper: `batch-run/run_p2_lvq.sh`

The benchmark repository is still in a pivot/development state, but its P2 LVQ
path is now intentionally non-SVS: it uses this standalone C++ LVQ repository
for residual indexing and search.
