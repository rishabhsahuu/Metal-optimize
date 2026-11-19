# The adaptive expert tier on the Mac (proposal)

**Status: the placement policy is in the branch and off by default. The counts, the heat-based victim and the profile save
are implemented (`metal/src/engine/stream.cpp`, `metal/src/engine/stream_policy.h`, with
`metal/tests/stream_policy_test.cpp` passing); the proactive rounds are not, and none of it has been measured on a Mac with
a model. Nothing on this page is a speed claim.** The Mac numbers quoted are from [MAC.md](MAC.md), each with what it was
measured on. The numbers a change would have to beat are named in [What to measure](#what-to-measure).

This is not a Basalt port. Basalt's speed on Blackwell comes from FP4 tensor cores, a discrete VRAM tier and CUDA; none of
those exist on a Mac. The one idea worth taking is the adaptive expert tier, and Strata already runs it on its CUDA and HIP
engines. This page is how to give the same behaviour to the Mac's SSD-streaming arena. It is one item in the wider
[MAC_OPTIMIZATION_PLAN.md](MAC_OPTIMIZATION_PLAN.md).

## What each engine has today

| | CUDA / HIP | Mac (Metal) |
| --- | --- | --- |
| Expert tiers | VRAM slots, a RAM copy, the CPU, an optional peer GPU | Arena slots in unified memory, everything else on the SSD |
| Placement at start | `--expert-profile` (STRP file) | The same STRP profile (`STRATA_METAL_EXPERT_PROFILE`, else `data/expert-profile.bin`) |
| Eviction | Router heat, the profile's rank | Pure LRU ([`stream.cpp`](../metal/src/engine/stream.cpp) `victim()`, from the size class's tail) |
| Read-ahead | Routing-aware prefetch, look-ahead | The next layers' predicted experts, on the I/O threads (`at_layer`, `ahead`) |
| Follows the conversation | `--adapt-every`, `--adapt-swaps`, `--adapt-decay`, `--adapt-min-gain`, `--adapt-async` ([DETAILS.md](DETAILS.md#low-ram-modes-chosen-by-setup)) | No |
| Learns a profile across runs | `--expert-profile-save` (#477) | No |

The Mac arena already has the hard parts: a slot pool in size classes, an LRU, a slot state machine (idle, queued,
reading), pinning per handshake (`stamp_`), six I/O threads, and a stepped fill that respects what macOS will give.
The missing part is the policy, not the machinery.

Basalt's public numbers are the reason to care about the cadence: on its own page, slowing its round from every window to
every 64 dropped a lone RTX 5090 from about 240 to 160 tok/s. A policy that runs on the wrong cadence is worse than none,
which is why stage 3 is last and default-off.

## What is different on the Mac (and why this is safer)

1. **One memory pool, one copy of the weights.** There is no GPU tier and CPU tier reading the same expert in
   different precision. A slot holds the expert's GGUF bytes unchanged; only the address changes
   ([`stream.h`](../metal/src/engine/stream.h)). So on the Mac, expert placement **cannot change a single token**.
   docs/MAC.md records the check that this rests on: the same greedy answers on 10 of 10 prompts between the streaming
   M4 and the resident M5 Max. The CUDA tier is not bit-exact from run to run; this one can be, and the port should keep
   that property.
2. **The swap is asymmetric.** Evicting costs nothing (drop the mapping). Bringing an expert in is an SSD read: one
   request with the experts' copy, three without (`read_part`). There is no PCIe copy, no pinned staging buffer, no
   second tier to copy from.
3. **The SSD is the bottleneck, shared by three consumers:** demand misses, the read-ahead, and any proactive round. A
   round that reads an expert nobody asks for costs exactly what a miss costs. This is stricter than on CUDA, where a
   swap moves bytes over PCIe beside compute.
4. **Only small Macs are affected.** A pack that fits is never streamed (`ExpertStream::wanted`), so this work only
   touches 16 to 48 GB Macs. On the 128 GB M5 Max it is dead code.

## Stage 1: learn the profile (lowest risk, do it first) — implemented

The arena's start ordering is what a small Mac lives on. On the 32 GB M4 the arena is 11.5 to 14.7 GB; on this 16 GB
Air it is about 5 GB, so most experts are always cold and the top of the profile is nearly all that matters. Today that
ordering is whatever profile shipped, built from someone else's trace.

- **Count.** In `at_layer`, after `want[]` is built, add one decayed count per routed (layer, expert):
  `heat = heat * decay + 1`. Counts are per window, never per clock, so the same session history gives the same numbers.
- **Write.** Reuse what already exists and is already linked into the Metal target
  (`metal/CMakeLists.txt`: `core/expert_cache.cpp`):
  `strata::core::rank_learned_profile()` then `strata::core::write_expert_profile()`. The writer is atomic (a `.tmp`
  renamed over the file), so a reader never sees half a profile, and the format is the same STRP v1 that
  `tools/make_profile.py` writes.
- **Read.** Replace the hand-rolled STRP header parse in `make_arena` with `strata::core::read_expert_profile()`. Same
  bytes, one parser for both engines.
- **Where it goes.** Not the pack's `data/expert-profile.bin`, which setup owns and may overwrite. A path in the data
  dir, chosen by `STRATA_METAL_ADAPT_PROFILE_SAVE`, or written only on clean shutdown when
  `STRATA_METAL_ADAPT_PROFILE_SAVE=1`.
- **Why it is safe.** The decode path does not change. The only effect is which experts are read first at the next
  start, and the reads return the same bytes.

This is the piece most likely to pay on a 16 GB Air, because it costs one write and no per-window work.

## Stage 2: pick the victim by heat, not by age — implemented

`victim()` walks the size class's LRU tail and takes the first idle, unpinned slot. That is right for a workload that
never changes and wrong at every topic change: a hot expert that went untouched for a few windows sits near the tail and
is evicted, then read again on the way back.

- Keep the LRU walk, but bounded: scan at most `S` slots from the tail (default 32) and take the lowest decayed heat,
  with the oldest winning ties. A full scan is O(slots) on the host turn, which is the wrong place for it.
- **Hysteresis.** An expert is only worth keeping over the one it replaces when its heat beats it by a margin
  (`adapt-min-gain`'s role on CUDA). Without it, two nearly equal experts swap back and forth and each swap is an SSD
  read. **Not in the branch:** the first cut only chooses among the coldest of the scanned slots, which protects a hot
  expert without needing the margin. The margin belongs with the rounds, where a swap is optional.
- **Never touch:** a slot with `state != 0`, a slot with `pin == stamp_` (the GPU is reading it this handshake), a
  retired slot (`retired_`), or a slot inside the prompt region (`region_slots_`).

## Stage 3: proactive rounds (default off)

Between decode windows, promote the hottest non-resident experts into the coldest slots. This is Basalt's round and
CUDA's `--adapt-every`, and on the Mac it is the stage most likely to lose:

- Only run when the I/O queues are empty (`urgent_q_`, `pre_q_`) and no slot is reading, so a round never delays a real
  miss.
- Suspend while a prompt region is held (`hold_region(true)`), and while `T > 8` chunks are running.
- Cap the round (`adapt-swaps`; Basalt uses 96, and 96 NVFP4 experts is about 250 MB each way there, where here it is a
  raw SSD read of the same bytes).
- Skip entirely when the arena has unused slots after an early-stop fill: it has spare places already.
- Decay after each round, and count rounds by windows, not by time.

On this Air the working set is far larger than the arena, so a round can do nothing but steal SSD bandwidth. That is a
real outcome and should be measured, not assumed away.

## Switches

All off unless set, in the style of the other Mac experiments:

| Variable | Default | In the branch | Effect |
| --- | --- | --- | --- |
| `STRATA_METAL_ADAPT` | 0 | yes | The master switch for stages 2 and 3 |
| `STRATA_METAL_ADAPT_DECAY` | 0.85 | yes | Count decay per window; lower follows a topic change sooner |
| `STRATA_METAL_ADAPT_SCAN` | 32 | yes | Slots the victim search may look at from the LRU tail (0: the plain LRU) |
| `STRATA_METAL_ADAPT_PROFILE_SAVE` | off | yes | `1` (beside the GGUF) or a path: write what was learned at exit, in the STRP format the arena already reads |
| `STRATA_METAL_ADAPT_EVERY` | 1 | no | Windows between rounds (stage 3) |
| `STRATA_METAL_ADAPT_SWAPS` | 32 | no | Pairs a round may move (stage 3) |
| `STRATA_METAL_ADAPT_MIN_GAIN` | 1.0 | no | The heat margin a newcomer needs over the victim (stage 3) |

The three unbuilt switches belong to the proactive rounds, which are the stage most likely to lose on a small Mac. The
decay and scan values are starting points copied from the CUDA defaults and Basalt's page, not values measured here.

## Where the code goes

- `metal/src/engine/stream.h`, `stream.cpp`: `heat_`, the round, the save/read, and the victim change.
- A new `metal/src/engine/stream_policy.h`: the victim choice, the round's admission test and the decay as pure functions
  over plain arrays, in the shape of `arena_budget.h`. That is what makes it testable on a PC.
- `metal/tests/stream_policy_test.cpp`: no Mac needed. Feed it a recorded sequence of per-window routings and check that
  (a) the heat ordering is what the counts say, (b) the margin prevents the ping-pong, (c) a round never proposes a
  pinned or reading slot, (d) the same history gives the same placement twice.
- `tools/replay_cache_policy.py` already replays a recorded routing trace against a policy; it is the natural place to
  score a policy offline before building it. A learned profile can also be scored on a trace it was not built from, which
  is the only way to know whether it transfers.
- `docs/MAC.md`: one row in the switches table and a line in the streaming section, once measured. Not before.

## What to measure

Two Macs: this MacBook Air M4 16 GB, and a 32 GB M4 Mac mini (`docs/MAC.md`'s streaming fixture). IQ3_XXS on both, with
the arena pinned with `STRATA_METAL_ARENA_GB` so the arms are comparable. IQ3_XXS needs about 55 GB of memory to stay
resident, so it streams on both; on the Air its download (75.8 GB with the n-gram table) needs about 15 GB of disk freed
first.

- **Arms:** A = today (static profile, LRU); B = A + learned profile from A's own run; C = B + heat victims; D = C + rounds.
- **Metrics:** decode tok/s (short question, story, code, JSON); prompt tok/s at 4K and 16K; the `at_layer` hit rate and
  prefetch hit rate from `report()`; SSD MB per token; stall ms per window; and the fill time in the start line.
- **Protocol:** a fresh process per arm, interleaved pairs, medians of five, identical tokens checked against arm A, and
  the disk's free space and the arena's actual size recorded with every run. An Air on battery is a different machine
  from one on the charger; say which.
- **Report the failures.** On the Air, no gain at all is the likely result and is worth writing down. A policy that costs
  33% when its cadence is wrong (Basalt's own 240 to 160) is a policy that needs a null result before it ships.

Until these runs exist, no sentence in this repository should say the adaptive tier makes a Mac faster.

## Limits this design does not fix

- It cannot make a 125B model fit 16 GB. It changes which experts wait on the SSD, not how many.
- It does not touch the prompt path's expert GEMMs, the dense weights or the draft layer.
- It gives the Mac nothing from Basalt's Blackwell work: no FP4 tensor cores, no TMA, no peer GPU, no NVFP4 layout.
