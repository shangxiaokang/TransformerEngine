..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

QwenAir QSA training kernel plan
================================

Status
------

This is an implementation plan, not a claim that a production QSA kernel is
available. The two current PyTorch APIs are numerical bring-up paths:
``qsa_block_sparse_attention`` explicitly computes FP32 scores, while
``qsa_indexed_sdpa_attention`` gathers selected K/V and calls PyTorch SDPA.
Both support autograd for Q, K, and V on unpadded causal self-attention.

An isolated wheel build from fork commit ``c4f14012`` succeeded in the same
B300 PyTorch 25.10 container with CUDA 13.0 and ``NVTE_CUDA_ARCHS=103a``.
The container's cuDNN Frontend 1.14.1 was too old for this source's declared
``>=1.29.0`` requirement, so 1.30.0 was installed only in a scratch target.
The matching native extension and Python package were then installed in that
isolated target. ``import transformer_engine.pytorch`` loaded that
extension, exposed ``DType.kFloat8E8M0`` and both QSA APIs, and completed a
BF16 QSA forward/backward smoke test on B300. The full Transformer Engine
L0 test suite remains untested with this wheel.

B300 job ``4801923`` used that isolated native wheel to run the downstream
Megatron Core QwenAir reference, explicit-gather QSA, and indexed-SDPA
regression suite and exited successfully. Two-GPU B300 job ``4801872`` then
combined indexed SDPA with the QwenAir PLE and expert-parallel paths for two
BF16/AdamW steps, distributed-checkpoint reload, and optimizer restart; it
also exited successfully. These jobs validate that the native TE package can
drive the current small-model training prototype. They do not validate the
177B target configuration, 262K context, or a production sparse kernel;
indexed SDPA still selected PyTorch's math backend.

On a B300 PCIe GPU with the PyTorch 25.10 container, all 188 standalone QSA
tests passed, including BF16 forward and backward comparisons with an
independent dense mask. Profiling showed that indexed SDPA selected
``aten::_scaled_dot_product_attention_math``. Forcing FlashAttention failed
because its implementation rejected the non-null per-token selection mask.
With a deliberately small test geometry (BF16, one batch, 24:2 GQA, head
dimension 64, four selected blocks, query chunk eight), backward peak
allocation above the input baseline was 4,730,880 bytes at sequence length
128 and 18,121,728 bytes at 512. This checks memory growth for the reference;
it does not predict performance at the target model geometry.

Exact selection contract
------------------------

The target full-attention layers use 24 query heads, two KV heads, head
dimension 256, four-token compressed key blocks, and a budget of 2,048
selected tokens (512 complete blocks) per query token. The indexer has four
query heads, one key head, and head dimension 128. The maximum context is
262,144 tokens.

For zero-based query position ``q``, a complete block ``b`` is eligible only
when ``4 * b + 3 <= q``. Selection is recomputed independently for every
query token; four queries in one block may choose different key blocks. The
current incomplete block contributes its visible ``q % 4 + 1`` tokens when
``(q + 1) % 4 != 0``. At ``q % 4 == 3`` it is complete and participates only
if top-k selects it. An absent selection is encoded as ``-1``. Duplicate,
future, or incomplete selected blocks are invalid. This is the contract of
the existing ``[B, S, K]`` TE API and must survive every optimized backend.

The reference indexer averages four raw key vectors in FP32 and converts the
pooled result to the input dtype before key RMSNorm and RoPE at the block's
first position. For each candidate block it computes four FP32 query/key dot
products, applies ReLU to each, sums the four values, divides by
``sqrt(128)``, and takes top-k over the *complete causal prefix*. Any GPU
indexer must preserve this order of operations until selection parity is
measured. In particular, applying top-k to a fixed-width masked array can
change tie behavior compared with top-k over the actual prefix.

Scale and bottlenecks
---------------------

At ``S = 262144``, ``K = 512``, and batch one:

* The selected ``int32[B, S, K]`` tensor occupies 512 MiB per QSA layer.
  An unsharded BF16 query tensor is 3 GiB; key and value tensors are 256 MiB
  each. Activation storage and gradients add to these figures.
* The existing streaming indexer invokes top-k once per four-token group:
  about 65,536 Python-triggered top-k operations per layer. Its 16-token
  score chunks add about 16,384 score operations. This launch count alone
  prevents a production-length training run.
* At the default query chunk of eight, indexed SDPA enters Python about
  32,768 times per layer. Each chunk gathers K/V and builds a boolean mask.
  The math backend can expand the 12:1 GQA ratio internally. The explicit
  BF16 K/V gather at the target geometry is about 32 MiB per chunk, before
  backend intermediates and backward saves.
* The causal indexer does roughly 8.8 trillion floating-point operations
  per layer to score all eligible blocks. QK plus PV for 2,048 selected
  tokens costs roughly 13.2 trillion operations in attention forward.
  These are arithmetic counts, not measured throughput.

A faster SDPA dispatch alone cannot solve the indexer launch count. Likewise,
a fused indexer alone leaves tens of thousands of Python attention chunks.
The production path must replace both.

Proposed TE interfaces
----------------------

The first production release should accept the same attention tensors and
selected-block convention as the current TE API, so Megatron can switch
backends without changing model semantics. A separate low-level selector
would accept already projected, normalized, and rotated indexer queries and
block keys; projection and model policy remain in Megatron.

.. code-block:: python

   qsa_select_key_blocks(
       index_query,      # [B, S, 4, 128]
       index_block_key,  # [B, floor(S / 4), 128] complete blocks
       *,
       topk=512,
   ) -> blocks            # int32 [B, S, topk], -1 padding

   qsa_fused_attention(
       query, key, value, blocks,
       *,
       scale=None,
   ) -> output            # [B, S, Hq, Dv]

Version one should require contiguous, unpadded causal self-attention,
``Hq % Hkv == 0``, matching Q/K/V dtypes, and dropout zero. The current
model's RoPE promotes Q/K to FP32 under BF16 autocast; its value is promoted
exactly to FP32 for the existing TE API. The fused backend must support this
actual FP32 path before replacing the model reference. Any faster BF16 path
needs a separately verified casting contract; it must not silently narrow
the FP32 RoPE result. Packed sequences, left padding, context parallelism,
and arbitrary block sizes require explicit later contracts and must fail
closed in the first release. Indices are discrete and do not carry autograd
gradients.

Kernel sequence
---------------

1. Pool indexer token keys once per layer, then apply the existing key
   RMSNorm and RoPE. Implement a tiled or persistent GPU selector that scores
   causal prefixes and computes per-token top-512 without a ``[S, S / 4]``
   score tensor or a host launch per four tokens. Start with exact FP32
   scoring. A hierarchical top-k may keep bounded tile candidates and merge
   them on device; candidate scratch, sorting work, and tie behavior must be
   measured before choosing it over a persistent per-query design.
2. Implement indexed attention forward in TE common CUDA/CuTe or Triton.
   Read each selected four-token K/V block indirectly and append the current
   incomplete tail. Group the 12 query heads sharing a KV head, use online
   FP32 softmax, and preserve the requested output dtype. Do not materialize
   gathered K/V, repeated GQA K/V, or a dense boolean attention mask. Launch
   over many query rows from one framework call rather than a Python loop per
   chunk.
3. Add backward. Compute dQ per query row. Accumulate dK/dV into FP32
   workspaces or use a key-centric reduction; BF16 atomics and highly shared
   selected blocks need explicit contention and accuracy tests. Recompute
   attention probabilities from Q/K and saved row normalization statistics
   when this reduces activation memory. Verify Q/K/V and projection-weight
   gradients against both existing references.
4. Add Megatron backend dispatch after standalone TE tests pass. Keep model
   projection, RoPE, gating, optimizer, and pipeline/expert parallel policy
   in Megatron. Validate one GPU, then EP+PLE multi-GPU training. Long-context
   context parallelism needs a separate data-movement design: compressed
   index keys can be shared across sequence ranks, while selected remote
   K/V blocks and their backward gradients need routing or controlled
   all-gather. Global block IDs and sequence offsets must be specified before
   enabling that backend.

Training acceptance has two distinct paths. For continuation from a
pretrained checkpoint, freeze the existing hard-top-k selector and verify
forward selection, attention and model gradients, optimizer updates, and
checkpoint reload. The current hard indices do not transmit language-model
loss gradients to indexer projection or norm parameters. From-scratch
selector training must wait for an authoritative auxiliary-loss or
differentiable-selection contract; no loss should be invented by the TE
kernel. Once specified, test nonzero indexer gradients and end-to-end
training separately.

Verification gates
------------------

* Check every four-token causal boundary, changing selections within one
  query block, empty selections, duplicate/future rejection, and GQA 24:2.
  Compare forward and all input/parameter gradients with the dense mask at
  small sequence lengths under an IEEE FP32 setting. Include BF16 autocast,
  FP32 weights, and TF32-enabled BF16 process settings.
* Compare selected block IDs with the frozen reference at small lengths and
  on sampled query windows from large lengths. Add adversarial equal-score
  cases; top-k ties must have an explicit policy before claiming exact
  checkpoint parity.
* Use profiler traces to prove the selector and attention no longer issue a
  number of host launches proportional to ``S / 4`` or ``S / 8``. Record
  forward/backward latency and allocated-memory peaks on B300 at 4K, 32K,
  and the 262K target. Confirm there is no sequence-square allocation.
* Run a real optimizer step and checkpoint reload with the QwenAir model,
  then repeat with the intended parallel configuration. A passing kernel
  microbenchmark alone is insufficient to claim the full model trains.

The current indexed SDPA API remains the numerical fallback until these
gates pass. It is not a production performance baseline for the 262K target.
