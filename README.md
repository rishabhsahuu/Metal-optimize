# Metal-optimize

Apple Silicon (Metal) experiments on [Strata](https://github.com/Niko1221/Strata): a branch that makes the Metal engine
faster on a small unified-memory Mac without changing the answers.

The target is the hard case: **Qwen3.8-Flash-Next IQ3_XXS** (47 GB, quality over Q2_0) on a 16 GB MacBook Air M4.
About 40 GB of experts have to share 12.7 GB of GPU working set, so the work is about which experts are where, not about
faster arithmetic.

Two rules hold for everything here:

- **Bitwise or off.** A change that claims no output change is checked against the pair it replaces in
  `metal/tests/fuse_parity.cpp`; a change that cannot be checked without a model is behind a switch that is off by
  default.
- **Number with its machine.** Every speed or memory number below says what it was measured on. Nothing is claimed for
  hardware it was not run on.

## The test machine

| | |
| --- | --- |
| Mac | MacBook Air M4 (Mac16,12), 17.2 GB unified memory, macOS 26.6.2 |
| Pack | Qwen3.8-Flash-Next **IQ3_XXS**, 47 GB, experts streamed from the internal SSD |
| SSD | 2.49 GB/s measured with `dd` (the `.experts` copy is on the internal disk) |
| Server | 32K context, DFlash2 drafter, windows up to 6 tokens, thinking budget 1,024 tokens |

## Results on that Air

| | |
| --- | --- |
| Before the arena preload fix | **3.2 tok/s** |
| After it, whole-request totals | **3.4–3.5 tok/s** (the live readout says 3.5–3.6; see below) |
| Expert arena | 3.37–3.42 GB, 1,728–1,752 slots of 24,576 |
| Experts preloaded from the profile | 1,490 at 2.17 GB/s (the run that worked); 0 in a later run with only 7.0 GB free at planning |
| Prompt cost | 69 tokens in 16.0 s: one full pass over the experts, about 40 GB at 2.5 GB/s |
| 1,388 generated | 3.5 tok/s, drafts accepted 736 of 1,149 (64%) |
| 1,024 generated | 3.4 tok/s, drafts accepted 413 of 747 (55%) |
| Reference, 32 GB M4 mini | 10.0 tok/s short, 9.2 story, 7.7 after 6K (arena 11.5–14.7 GB) |

**How to read these numbers.** The live `tok/s` in the server output is a 2-second window (`serve/server.py:313`), not
the request average, so it moves with how many drafts the last window accepted. Only the `N generated in X ms` line at
the end of a request compares across runs. The 3.2 → 3.6 step was read from the live line; whole-request totals after
the fix are 3.4–3.5, and the spread inside one process tracks draft acceptance (64% → 3.5, 55% → 3.4), not the kernel
batch.

**The prompt is a fixed pass, not a per-token cost.** A chunk routes nearly every expert in every layer, so 69 tokens
and 1,024 tokens both pay one read of about 40 GB. That is why a short prompt shows 4.3 tok/s and a 1,024-token prompt
would show about 64 tok/s for the same wall time.

**A 16 GB Air's expert tier is about 4 GB, not 10.** The window holds 8.13 GB before the arena (5.08 GB of Metal buffers — 2.84 dense, 0.68 the hyper-connections, 0.88 the drafter — plus host memory) and the GPU working set is 12.7 GB, so `arena_budget` can give the arena at most about 4 GB (12.7 − 8.1 − 0.5). The experts themselves are 47 GB, so a 4 GB arena holds about 9 % of them; that is why the exit report shows a 30 % hit rate and 1.6 GB read per window.

`--expert-cache` is not part of that number: the host code's device copies are given back at the first request (9.94 GB `given back (unused by the Metal window)`, its 3,921 slots included), which is also why every request's report says `GPU 0 hits this request`. Leaving the flag off only skips a 21 s, 6.87 GB startup fill.

**The dense weights can come from the shard instead of a copy.** `STRATA_METAL_DENSE_FILE=1` wraps the dense tensors' own pages of the GGUF as the Metal buffer (no copy: 292 runs, 2.84 GB of clean file pages) and leaves the 40 GiB of routed experts out of the wrap. On this Air the process then holds 5.41 GB instead of 8.11 GB, and the room left under the footprint cap rises from 4.1 GB to 6.9 GB. The arena does not grow by that much, because the same 2.84 GB still occupies RAM: hot file pages are not free memory, and `arena_avail` does not count them. The measured arena went from 3.27 GB (1,742 slots) to 3.56 GB (1,894 slots), +9 %. The wrapped path returned the same text as the copy path for two greedy prompts (24 tokens each); one 64-token request at the 3.56 GB arena took 20.7 s (3.1 tok/s, drafts 34 of 62), inside the run-to-run spread. The first attempt wrapped the whole 47 GB shard, and the first window's submit never returned: the driver makes the buffers a window binds resident, and a 9.5 GB chunk carries the experts the arena is streaming. Only the dense runs are wrapped now.

**What the SSD costs, and the ceiling.** Timing only, with `STRATA_METAL_STREAM_FAKE=1` (every miss is pointed at slot 0, so
nothing is read and the answers are wrong), the same engine on the same Air took **73–74 ms for a decode step**
(13.5 tok/s) once the arena was warm, and **19.4 s for a 2,512-token prompt** (129.8 tok/s). The real engine at a
4.00 GB arena (2,133 slots preloaded in 1.9 s, footprint 12.6 GB of the 12.7 GB cap) took 16.6 s for 64 tokens
(3.9 tok/s, drafts 36 of 51) and 47.2 s for 2,833 prompt tokens (60.1 tok/s) at 3.58 GB. So the expert reads are about
three quarters of a decode step: even with every expert resident this Air would not pass about 13 tok/s, and it cannot
hold the 40 GiB of experts that would need. The fake-mode step is one token (the drafter proposes nothing on its
garbage hidden states), so it is a floor for one step, not a whole-request average.

What the arena actually gets depends on `STRATA_METAL_KEEP_FREE_GB` and the memory free at the first request. With 4.6 GB free and 3.5 GB kept the budget was 1.0 GB, under the arena's floor (3.4 GB: a prompt chunk's layer needs 576 slots in each of three size classes), the arena was forced over the budget, and a 3,575-token prompt then failed on its second chunk. With the default 7 GB kept the plan is smaller and the fill refuses earlier — the safer setting on 16 GB. Same prompt, 2,833 tokens read in 47.2 s (60.1 tok/s) when macOS had 7.6 GB free.

## Nine changes so far

Six of the sixteen items in [the plan](docs/MAC_OPTIMIZATION_PLAN.md) are implemented (M2, M3, M4, M6, M7, M8). Rows 3
and 8 are the configuration that made the measured step possible; row 9 is a memory experiment, off by default.

| # | Change | Plan item | Default | Evidence | Effect |
| --- | --- | --- | --- | --- | --- |
| 1 | Learned expert profile for the streaming arena: per-layer routing counts, saved at exit | M7 | on with `STRATA_METAL_ADAPT=1` | `metal/tests/stream_policy_test.cpp` (8/8) | The arena fills from what the model actually routes, not from a static list |
| 2 | Heat-based victim: a cold expert leaves before a hot one | M8 | with M7 | `stream_policy_test.cpp` (8/8) | Replaces the blind LRU |
| 3 | The arena preload now runs (`STRATA_METAL_KEEP_FREE_GB` 7 → 3.5) | the fix under M7 | `run-iq3_xxs-tuned.sh` | 1,490 experts at 2.17 GB/s in one run | **3.2 → 3.4–3.5 tok/s** |
| 4 | `router_route` and `gemv_cat` in the decode window | M2 | on, `STRATA_METAL_FUSED_ROUTER=0` / `_GEMV_CAT=0` off | `fuse_parity`: logits, ids, w, plan bitwise, 1–8 tokens | One dispatch less a layer (~0.25 ms a window, static) |
| 5 | `gr_norm` folded into the mixer GEMVs and the MoE combine (`hcf.metal`) | M3 | **off**, `STRATA_METAL_HCF=1` | `fuse_parity`: bo, R′, xn bitwise, 1–8 tokens | −10.9 % on the HC read, measured on an M5 Max UD-Q4_K_XL (from `hcf.metal`); not measured here |
| 6 | The sampler rides in the window's command buffer | M4 | on | Same kernels, same buffers, same order | One command buffer fewer a window (<1 ms, static) |
| 7 | The prompt chunk yields to the live arena (halves in 128-row steps) | M6 | on, shrink only | Did not trigger at 3.37 GB (1.60 GB of buffers is under half) | A large chunk's buffers cannot evict the arena it is trying to use |
| 8 | The tuned configuration: 32K context, `--mmap-experts`, `STRATA_METAL_KEEP_FREE_GB` 7 (the default), no `--expert-cache` (its slots are given back at the first request anyway; off skips a 21 s fill), budget 1,024 | — | `run-iq3_xxs-tuned.sh` | The runs above | The numbers above |
| 9 | The dense weights wrapped from the shard instead of copied into Metal memory | — | **off**, `STRATA_METAL_DENSE_FILE=1` | The same two greedy prompts as the copy path returned the same text; 292 runs, 2.84 GB of file pages | Anonymous footprint 8.11 → 5.41 GB; arena 3.27 → 3.56 GB (+9 % slots); no speed change measured |

## Setup

### From the setup script

```sh
git clone https://github.com/rishabhsahuu/Metal-optimize.git
cd Metal-optimize
./setup.sh --model IQ3_XXS      # checks the Mac, builds build-metal/strata, downloads and prepares the pack
./run-iq3_xxs.sh
```

The script needs Python 3.10 or newer (it fetches a standalone 3.12 if the Mac only has 3.9), Apple's Command Line
Tools, and about 82 GB of free disk for the pack and its working space. It downloads the metal-cpp headers into
`third_party/metal-cpp` and builds with CMake and Ninja from `.venv`. Details: [docs/MAC.md](docs/MAC.md).

### Build an existing checkout

```sh
# metal-cpp headers: setup.sh puts them in third_party/metal-cpp
cmake -S metal -B build-metal -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DMETAL_CPP_DIR="$PWD/third_party/metal-cpp"
cmake --build build-metal --target strata -j 8
./build-metal/strata --version      # strata 0.1.59-metal
```

### Run the tuned configuration

`run-iq3_xxs-tuned.sh` starts the server on port 8080 with `strata-iq3_xxs.tuned.json`. Edit the three model paths in
that JSON to your own (`--pack`, `--native`, `--ple-gguf`); everything else is the tuning described above.

The same settings from the installation script's `strata` command:

```sh
strata launch --family qwen --model iq3_xxs --context 32k --yes --no-tui
```

Useful flags: `--mmap-experts`, `--spec N`, `--api-key-env NAME`, `--new-key`, `--speed faster`. Leave
`--expert-cache N` off: it is the host code's cache, and the Metal window's arena is the cache it uses.
The server prints its addresses and keeps running; `strata launch --status` shows it, `--stop` stops it.

### Tests

```sh
cmake --build build-metal --target fuse_parity stream_policy_test -j 8
./build-metal/fuse_parity           # ALL PASS: gemv_cat, router_route, gr_norm_comb, hcf, 1..8 tokens
./build-metal/stream_policy_test    # all passed
python3 tools/check_links.py        # 132 pages, all local links resolve
```

## Switches

All are off unless set. The defaults are the tuned path above.

| Variable | Effect |
| --- | --- |
| `STRATA_METAL_KEEP_FREE_GB=3.5` | Memory left to macOS. This is the gate the arena fill uses; 7 (the default) leaves too little for the preload on a 16 GB Mac |
| `STRATA_METAL_ADAPT=1` | The adaptive expert tier (M7 counts, M8 heat). `_ADAPT_SCAN`, `_ADAPT_PROFILE_SAVE` tune it |
| `STRATA_METAL_FUSED_ROUTER=0`, `STRATA_METAL_FUSED_GEMV_CAT=0` | Keep the two dispatch pairs instead of the fusions (M2) |
| `STRATA_METAL_HCF=1` | Fold `gr_norm` into the mixers and the MoE combine (M3) |
| `STRATA_METAL_PROMPT_CHUNK=N` | Pin the prompt chunk in rows; the default is automatic and yields to the arena (M6) |
| `STRATA_METAL_ARENA_GB=N` | Set the arena size explicitly |
| `STRATA_METAL_DENSE_FILE=1` | Streaming: wrap the dense tensors' pages of the shard instead of a 2.84 GB copy. Smaller anonymous footprint (8.11 → 5.41 GB here); the arena grew 3.27 → 3.56 GB on this Air, not by the whole copy, because the pages still occupy RAM |

## What is not here

- **M1, fp16 dense decode.** It can only touch the dense matrices when streaming, and it needs a 100-chunk KLD gate
  before it can be a default. Not measured.
- **M5, skip a fold when nothing changed.** The condition has no observable without a model trace, and skipping a fold
  that was needed would change tokens. Deferred, not forgotten.
- **M9–M16.** Expert-streaming scheduling, QSA attention, prompt-chunk growth and the rest of the queue in
  [the plan](docs/MAC_OPTIMIZATION_PLAN.md).
- **An A/B against the branch point.** The four kernel/scheduling changes remove microseconds from a ~290 ms window, so
  they are inside run-to-run noise on this Air. The measured step is the arena preload fix.

The ceiling is memory, not arithmetic: the arena can reach about 3.6 GB here (12.7 GB working set, ~8.1 GB held before
it, 3.5 GB kept for macOS out of 17.2 GB). The 32 GB mini's 11.5–14.7 GB arena is what 10 tok/s looks like.

## Docs

- [docs/MAC_OPTIMIZATION_PLAN.md](docs/MAC_OPTIMIZATION_PLAN.md): the ranked M1–M16 plan, what is in, what is not.
- [docs/MAC_ADAPTIVE_TIER.md](docs/MAC_ADAPTIVE_TIER.md): the adaptive expert tier design.
- [docs/MAC.md](docs/MAC.md): the upstream Mac guide, including the full switch list.
- [docs/DETAILS.md](docs/DETAILS.md): how the engine works, with the measured numbers it was tuned on.

## Built on Strata

This is a branch of [Strata](https://github.com/Niko1221/Strata) (MIT). The upstream README, with the installers, the
model guide and the desktop app, is kept in [README.upstream.md](README.upstream.md). The changes here are described by
[name and measurement](docs/MAC_OPTIMIZATION_PLAN.md); they are not a fork with a separate API.
