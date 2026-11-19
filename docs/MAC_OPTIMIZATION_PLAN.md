# Making the Mac engine faster: working plan

**Status: working document, branch `metal-optimize`. In the branch: the adaptive placement policy (M7's counts and profile
save, M8's heat-based victim, off by default behind `STRATA_METAL_ADAPT`); M2 (router_route + gemv_cat in the decode
window), M4 (the sampler in the window's command buffer) and M6 (the prompt chunk yields to the arena); and M3 (hcf
compiled and wired, off behind `STRATA_METAL_HCF=1`). `metal/tests/fuse_parity.cpp` now covers gemv_cat, router_route,
gr_norm_comb and hcf at every decode window size (1..8): ALL PASS. None of it has been measured on a Mac with a model,
so no speed claim is made and every new path has an off switch. Every other item below is unimplemented; M5 is deferred
because its condition ("the committed prefix did not move") has no observable without a model trace, and skipping a fold
that was needed would change tokens.** Every performance number quoted is from this repository's docs or code comments and
names what it was measured on; nothing was measured for this plan. Items marked *static analysis* are read from the code,
not run.

Two repositories are compared:

- **Strata** (`metal/`): the engine we change. Source of every measured Mac number below.
- **Basalt** (`/Users/rishabhsahu/coding/basalt`): a fork of Strata for one target, NVIDIA Blackwell. What it teaches is
  placement, scheduling and serving policy. Its execution tricks (FP4/FP8 tensor cores, TMA, PDL) have no Metal analogue
  and are listed under [what does not transfer](#what-does-not-transfer).

Companion document: [MAC_ADAPTIVE_TIER.md](MAC_ADAPTIVE_TIER.md) (the expert-placement part, in detail).

## Where a Mac's time goes

| Measurement | Value | Measured on | Source |
| --- | --- | --- | --- |
| One 2-token decode window | 111 ms: routed experts ~40, dense ~66 before the tile change | M4 Mac mini 32 GB, Gyro-S streamed | [MAC.md](MAC.md#history--older-measurements) |
| The same window with `STRATA_METAL_F16=1` | 56-60 ms -> 40-41 ms | M1 Max 34 GB, Q2_0 streamed | [MAC.md](MAC.md#switches) |
| SSD wait per token, experts' copy | 46.6 -> 31.5 ms | M4 Mac mini 32 GB, IQ3_XXS, 6K document | [MAC.md](MAC.md#history--older-measurements) |
| Prompt chunk 1024 / 2048 / 4096 | 700 / 920 / 1,100 tok/s at 4K | M5 Max 128 GB, UD-Q5 streamed | [MAC.md](MAC.md#history--older-measurements) |
| Dense GEMV tile change | 1.17 -> 0.99 ms | M5 Max, GDN out projection | `metal/src/engine/window.cpp:88` |
| Fused GDN input GEMM | 3.09 -> 2.87 ms a layer | M5 Max, T 2048 | `metal/src/engine/window.cpp:96` |
| One dispatch + barrier | ~7 us | decode chain | `metal/kernels/hcf.metal:11` |
| QSA attention load stalls | ~30 us a layer at a 1,000-cell context | decode | `metal/kernels/qsab.metal:2` |
| Expert layer, one token | 278 us (37.5 GB/s), IQ3_XXS 333 us, Q2_0 166 us | M4 Mac mini 32 GB, Gyro-S | [MAC.md](MAC.md#history--older-measurements) |

What that means, in two regimes:

- **Streamed Mac (16 to 48 GB).** The arena holds a few GB, every window misses on many experts, and each of the 48
  layers costs a host turn (a GPU signal, a host lookup, reads, and a GPU wait). Decode is dominated by the expert path
  and the per-layer handshake, not by arithmetic. The 111 ms window above splits roughly 40 % experts, 60 % dense.
- **Resident Mac (96 GB and up).** The experts come from the same memory as the dense weights, the handshake is still
  there but has no I/O to do, and the dense kernels dominate. Kernel-level wins matter most here; on a streamed Mac
  they matter less but are not free.

On the 16 GB MacBook Air, the arena is about 5 GB, so it sits at the far end of the first regime: the plan below is
ordered so that the items that help *that* machine come first.

## The queue

Effort is small (one file, no new kernel family), medium (a kernel family or a scheduling change), or large (buffers,
numerics). "Output" says whether tokens can change.

### P0: small, low risk, do first

| ID | Change | Files | Evidence it pays | Effort | Output |
| --- | --- | --- | --- | --- | --- |
| M1 | Turn on the fp16 decode windows for **streamed** packs | `window.cpp` `f16_auto_for` (`:167-173`), `:1271` | The function returns false when streaming, so a streamed Mac never gets it unless forced. Forced on an M1 Max: a 2-token window 56-60 -> 40-41 ms, mean KL 0.009 against fp32. **On a streamed pack only the dense GEMVs change**: the expert path's fp16 arm needs `!stream_` and Q2_0 (`:1271`) | small | Yes (fp16 math), so opt-in first, then a measured default |

**For an IQ3_XXS Air, read the table in a different order.** IQ3_XXS expert layers cost about twice Q2_0's (333 vs
166 us on the M4), so the streaming path, not the dense kernels, is where this pack's time goes. That promotes M9 (the
handshake), M7 and M8 (placement) ahead of M1, which on this pack can only touch the dense side. The rest of P0 stays:
they are bitwise, cheap and independent of the pack.

| M2 | Wire `router_route` and `gemv_cat` into the engine — **in the branch** | `router.metal`, `gemv.metal`, `window.cpp` | Both exist and are bitwise-checked against the two-dispatch form in `metal/tests/fuse_parity.cpp` (now 1..8 tokens); today only `token_chain` uses them. In the decode window for BF16/F32 routers; `STRATA_METAL_FUSED_ROUTER=0` / `_GEMV_CAT=0` keep the pairs | small | No |
| M3 | Compile `hcf.metal` and fold `gr_norm` into its producers — **in the branch, opt-in** | `msl_libs.h` (`lib_core`), `hcf.metal` | Was in no library list; now compiled and checked bitwise in `fuse_parity` (gemvh_ vs gemv 32,1,8,2 + gr_norm; moe_combine_hcf vs moe_combine_sh + gr_norm, 1..8 tokens). `STRATA_METAL_HCF=1` wires the two mixers and the MoE combine; the -10.9 % in `hcf.metal:9-12` is an M5 Max, UD-Q4_K_XL measurement (*static* here) | small | No |
| M4 | Append sampling to the window's command buffer — **in the branch** | `window.cpp:3433` | A whole `run_graph` per window for two small dispatches; greedy already reduces in-graph (`argmax_part`). The list is copied per window because SampArgs carries the token counter | small | No |
| M5 | Skip the fold when nothing changed | `window.cpp:2041-2063` | One `gdn_step` dispatch for each of the 36 GDN layers per fold pass, even when the committed prefix did not move (*static*) | small | No |
| M6 | Size the prompt chunk from the arena, not from RAM — **in the branch (shrink only)** | `window.cpp:2911-2919` | Was `phys >= 96 GB ? 4096 : 1024`. Now the measured buffers must stay under half the live arena, halving in 128-row steps first; growth still follows the phys rule, because the bisection up needs a measurement we do not have | small | No |

M1 is the single largest cheap win: it is a condition change, not a kernel, and the fallback is one environment variable.
It needs a quality gate first (100-chunk KLD and the 61-task suite on a streamed IQ3_XXS), because the KL cost
was 0.009 on the M1 Max. Gyro trellis packs have no fp16 decode (`x16_kind` returns -1 for the TQ types), so they are
unaffected and must be tested for no regression.

### P1: medium

| ID | Change | Files | Evidence it pays | Effort | Output |
| --- | --- | --- | --- | --- | --- |
| M7 | Learned expert profile (stage 1 of the adaptive tier) — **in the branch, opt-in** | `stream.h/.cpp`, `core/expert_cache` | The arena starts from a shipped profile built on another trace. On a small Mac the top of that ranking is most of the hit rate | small | No |
| M8 | Heat-based victim (stage 2) — **in the branch, opt-in** | `stream.cpp` `victim()`, new `stream_policy.h` | Pure LRU today; a topic excursion evicts hot experts | medium | No |
| M9 | Take the streaming handshake off the critical path | `window.cpp:1316`, `:3249`; `stream.cpp` `at_layer` | One cut per layer, and the GPU waits in `done_cv_`. Basalt's device-planned MoE does the same job without a host turn; step 1 (no cut when a layer is fully resident) is small, step 2 (look two layers ahead with `pred2`) is the structural one | medium | No |
| M10 | Use the batched QSA attention for short contexts | `qsab.metal`, `msl_libs.h` | Load stalls ~30 us a layer at 1,000 cells; the kernel is bitwise with `qsa_attn` (*static*) | small | No |
| M11 | Head: fused top-1 for sampled requests; a two-stage screen for large packs | `window.cpp`, `gemv.metal`, `sample.metal` | Basalt's two-stage head: 350 vs 439 us a window (NVFP4, 400 W + 180 W). Our head is read once a window; 248,320 rows x 2,560 at Q8_0 is ~675 MB, roughly 6-8 ms on an Air's memory bandwidth (*estimate from 120 GB/s theoretical*) | medium | Greedy no; a Q6_K proxy shifts the sampled tail (Basalt: KLD 0.0022) |
| M12 | Short-step flash attention for 1-3 row windows | `qsa.metal`, `stubs.cpp:253` | `qsa_flash` exists for prompts only; a decode window always takes the scalar path (*static*) | medium | No |

M9 is the highest-value structural item for the 16 GB Air, because at ~5 GB of arena almost every layer misses. It is
also the riskiest of P1: the arena must hold two layers' worth of slots at once, and a mistake shows up as a stall, not
a wrong token.

### P2: large, or gated on measurement that does not exist yet

| ID | Change | Evidence / reason to wait | Effort | Output |
| --- | --- | --- | --- | --- |
| M13 | `MAXT` 8 -> 16, two rows a slot | Basalt's window GEMM takes 32 rows with the same per-row bit argument. Needs per-slot draft state and every `NTOK` specialisation recompiled | large | No |
| M14 | Grouped expert GEMM across batch slots | `--parallel 4` gives 119-125 tok/s aggregate against 96-100 for one request (M5 Max, IQ3_S) yet window per slot reads the weights again | large | No |
| M15 | int8 prompt GEMM through MetalPerformancePrimitives | `metal/kernels/mm_i8.metal` is a probe that asks whether the matrix unit runs int8 faster than fp16; it is not built by `metal/CMakeLists.txt` and no result is recorded. Prefill only: decode is bandwidth-bound | large | Yes |
| M16 | GDN prompt recurrence as matmuls | Basalt expresses the 16-token update as matmuls; our scan is one lane per state column and the trellis profile says decode is ALU-bound. It changes the recurrence's rounding | large | Yes |

## What does not transfer

One line each, all verified in Basalt's own docs and code:

- **FP4 and FP8 tensor cores** (`mxf4nvf4`, E4M3) and **TMA**-fed GEMM: Apple GPUs have neither.
- **PDL** (programmatic dependent launch) and **CUDA graphs**: a Metal command buffer is the replacement for the second,
  and it is already used.
- **VMM** segmented expert cache, elastic KV, hot VRAM resize: no page-granular GPU mappings; our arena already shrinks
  by dropping chunks.
- **Page-locked host memory** and **PCIe split** (`--pcie-frac`): host and GPU memory are the same memory.
- **Peer GPUs**, layer splits, prompt splits: one GPU per Mac.
- **NVDEC / CUDA vision kernels**, **AVX2 CPU expert kernels**: no Metal path, and a CPU expert tier on a Mac would add
  a second reader of the same memory.
- **`cpu-share` for prompt experts**: on CUDA it exists because copying over PCIe costs more than the arithmetic. Here
  the expert is already in memory; keep the CPU out of it.

The reversal to watch: Metal *does* expose int8 matrix operations through MetalPerformancePrimitives, which Basalt's
CUDA-only view does not see. That is M15, and it is a prefill question, not a decode one.

## On a 16 GB MacBook Air, with IQ3_XXS

The target pack is IQ3_XXS: quality over minimum size. That changes the arithmetic in three places.

- **Disk comes first.** IQ3_XXS is a 47.0 GB file plus the 28.8 GB n-gram table: 75.8 GB to download, plus about 6 GB
  of working space (`REPACK_WORK_GB` in `strata_control/mac_fit.py`). That is ~82 GB, against 68 GB free on this
  machine. Free roughly 15 GB before starting or the install stops part-way.
- **It streams.** Resident needs about 55 GB (`model_need_gb`), against Metal's ~12.9 GB share of 16 GB. The arena lands
  near 5 GB, so most experts are always cold. The engine's own rule recommends Q2_0 at this size
  (`recommend_model`), so IQ3_XXS has to be chosen explicitly; the choice is kept with a warning (`docs/MAC.md`).
- **What it costs.** Measured on a 32 GB M4 with an 11.5-14.7 GB arena: 10.0 tok/s on a short question, 9.2 on a
  story, 7.7 after a 6K document, 65.8 tok/s on a 4K prompt. A 5.7-9.3 GB arena cost Q2_0 about 40 % of its decode
  (14.1 -> 8.3), so expect roughly 5-6 tok/s here before throttling. That is an extrapolation, not a measurement.
- **Per expert it is twice the work of Q2_0**: one expert layer takes 333 us against 166 us on the M4. The streaming
  path dominates, which is why M9, M7 and M8 come first for this pack.
- **No draft layer.** A streamed Mac below 96 GB runs without MTP (`stream_mtp`), and its 1.2 GB is worth more as
  arena.
- **Context.** The default is 32K, but every KV cell competes with the arena. 8-16K is the sane setting here; 36,864
  tokens costs 0.65 GB a sequence (`docs/MAC.md`).
- **What to expect in use.** 5-6 tok/s is roughly 200-270 words a minute, about reading speed. Usable for chat, slow
  for an agent that writes long files.

None of this makes the model fit. The honest framing for a 16 GB Air is "less slow", not "fast", and a fanless Air
throttles: every number must be taken twice, cold and after a few minutes, or it will flatter whichever build ran first.

## How to measure

Fixtures: this MacBook Air M4 16 GB, and a 32 GB M4 Mac mini (the engine's own streaming fixture, so the numbers line up
with [MAC.md](MAC.md)). IQ3_XXS on both, for the reason above; the arena pinned with `STRATA_METAL_ARENA_GB` so arms are
comparable. The mini's marks to beat: 10.0 tok/s short question, 9.2 story, 7.7 after a 6K document, 65.8 tok/s on a 4K
prompt.

- **Arms:** one change at a time against the branch point. Never two.
- **Metrics:** decode tok/s on a short question, a story, code and JSON; prompt tok/s at 4K and 16K; `at_layer` hit rate
  and prefetch hit rate from `ExpertStream::report()`; SSD MB per token; stall ms a window.
- **Attribution:** `STRATA_METAL_KPROF=1`, `STEPPROF`, `PROMPT_KPROF` and `STRATA_METAL_TRACE=1` already exist
  (`window.cpp:1772-1965`). Use them instead of guessing which kernel moved.
- **Protocol:** a fresh process per arm; interleaved pairs; medians of five; identical tokens against the branch point
  where the change claims no output change; the arena size and free disk recorded with every run; battery state noted.
- **Report the failures.** A change that is neutral is a result; a change measured on one machine once is not.

## Commits so far (branch `metal-optimize`)

1. `metal: adaptive expert placement (opt-in), with the policy test` — M7/M8, policy checked by
   `metal/tests/stream_policy_test.cpp`; the arena preload it needs was blocked by the keep-free rule on a 16 GB Air
   (see "On a 16 GB MacBook Air").
2. `metal: router_route and gemv_cat in the decode window (bitwise)` — M2, checked with `metal/tests/fuse_parity.cpp`.
3. `metal: the sampler rides in the window's command buffer` — M4.
4. `metal: the prompt chunk yields to the arena, not to the RAM size` — M6 (shrink only).
5. `metal: hcf compiled and wired into the HC reads, opt-in` — M3, kernels checked with `fuse_parity`.

The next four are M5 (deferred: no observable without a model trace), M9, M10 and M1, in that order for the Air.

M1 is a separate experiment, not a first commit: on a streamed IQ3_XXS it can only touch the dense GEMVs, and its effect
on this pack is unmeasured. Run it behind `STRATA_METAL_F16=1` with a 100-chunk KLD gate and the 61-task suite before it
is considered for a default.

Each one keeps the default path byte-identical until its measurement says otherwise, per `CONTRIBUTING.md`.

## Open questions

- Does fp16 decode help a 5 GB arena as much as the M1 Max's 64-slot one, and does it reach an IQ3_XXS pack's dense
  tensors at all? Only Q3_K-Q6_K, IQ4_XS and Q2_0 take the fp16 GEMV (`x16_kind`); the routed experts never do when
  streaming. Unmeasured.
- How much does the Air throttle over a 10-minute run, and does that hide kernel wins? Unmeasured.
- Is `stubs_gyro.cpp`'s "trellis unsupported" comment stale against the working path in `window.cpp:1252`? Worth
  confirming with the maintainer before touching the Gyro kernels.
- IQ3_XXS needs ~82 GB of free disk (75.8 GB of download plus working space) against 68 GB here, and the 40 GB experts'
  copy, the +17 % decode item in [MAC.md](MAC.md), does not fit beside it at all.
