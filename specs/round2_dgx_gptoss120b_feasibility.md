# DGX Spark GPT-OSS-120B feasibility and runtime decision

Access date: 2026-08-12. This report supports packet preparation; current
Spark conformance still depends on the packet's local preflight and invented
fixture smoke.

## Decision

Use a packet-owned Ollama 0.32.5 Linux ARM64 runtime with the exact official
`gpt-oss:120b` manifest. Recommend connected acquisition after explicit
operator acknowledgement, with a separately checksummed offline runtime/model
store as the tested fallback. The packet must stop before project calls unless
the exact runtime, model, structured-output, seed, timing, memory, resume, and
privacy gates pass.

## Exact artifacts

- OpenAI identifies `gpt-oss-120b` as a 117B-parameter, 5.1B-active,
  Apache-2.0 open-weight model. Its MoE weights are natively MXFP4, it uses the
  Harmony response format, supports structured output and configurable
  reasoning, and is designed to fit on a single 80 GB GPU.
- Official Hugging Face repository: `openai/gpt-oss-120b`. The packet records
  the repository revision resolved during acquisition, but the scientific
  runtime identity is the exact Ollama manifest and blobs below.
- Ollama tag: `gpt-oss:120b`; manifest SHA-256
  `a951a23b46a1f6093dafee2ea481d634b4e31ac720a8a16f3f91e04f5a40ecd9`.
- Model blob: SHA-256
  `6be6d66a3f546d8c19b130dc41dc24b2fc159f84ffbc76a0ee0676205083cf5a`,
  65,369,799,840 bytes. Template, license, and parameter blobs are also frozen
  in the generation delta.
- Runtime: Ollama 0.32.5 official `ollama-linux-arm64.tar.zst`, SHA-256
  `aa7e06b5683ee66c4a3ec68ea7236db43b5a5d0821f0dfe2c5a215f4462bddf4`,
  1,542,011,985 bytes. It is extracted below the packet without `sudo`, a
  service install, an updater, or `curl | sh`.

## Spark and memory basis

NVIDIA documents a 20-core Arm CPU, GB10 Grace Blackwell, 128 GB unified
LPDDR5x memory, and support for models up to 200B parameters. Current Founders
Edition release notes list DGX OS 7.5.0, driver 580.159.03, CUDA 13.0.2, and
kernel 6.17; partner systems may update on a different schedule. The July 2026
release improves unified-memory OOM reporting. NVIDIA also documents that
`nvidia-smi` conventional memory usage is not authoritative on Spark. The
packet therefore reads total/available system memory from `/proc/meminfo`,
records the method, checks model residency separately, and requires at least
20 GB available after smoke. A model load alone is not a pass.

The 2026-07-23 DocSpectra report (planning evidence only) observed aarch64
Ubuntu 24.04.4, GB10, driver 580.142, CUDA 13.0, NVIDIA Container Toolkit
1.19.0, about 121.69 GiB RAM, 3.18 TiB free disk, and corrected Docker/NVIDIA
runtime access. The packet reruns a compatible sanitized probe. Hardware and
runtime evidence can be mapped back to DocSpectra, but SpecTech model-load,
prompt, structured-output, latency, and experiment conformance cannot.

## Interface and execution gates

Ollama exposes the exact tag through a local API, supports one-field JSON
schemas, `think=low`, seed and sampling options, per-call client deadlines,
usage counts, and explicit unload. The packet freezes temperature 1.0, top-k
64, top-p 0.95, repeat penalty 1.0, context 4096, 256 output tokens, and three
attempts per slot from the reviewed human-first base. Every attempt has a fresh
conversation and a newly namespaced derived seed. Low reasoning is the frozen
minimum because the reviewed GPT-OSS-20B interface required it; invented
120B smoke must reconfirm it before project calls. Reasoning text is discarded
after recording only count and SHA-256.

The launcher is bounded by an OS-level 20-hour timeout as well as internal
monotonic deadlines. The conservative primary projection includes acquisition
when it occurs after launcher start, cold load, 540 calls at generation-smoke
p95, 30 minutes validation/resume allowance, and 15 minutes collection. It
must be at most 18 hours. Optional rubric work is admitted only after exact
180/180 primary completion and only when 80 rubric-smoke-p95 calls plus ten
minutes still fit under 18 hours. Collection begins by hour 19.

## Acquisition and security

Connected mode permits only the registered GitHub release and Ollama registry
names after `--allow-approved-download`. Every artifact is resumed only within
the frozen budget and rehashed. Offline mode accepts one separately transferred
payload only after its external checksum and internal manifest pass. Scientific
calls are local-only after acquisition. The packet uses packet-owned paths,
does not require privileged containers or host networking, never mounts the
Docker socket into a scientific container, and returns only sanitized minimum
evidence. OneDrive remains a transfer channel, not an execution directory.

## Alternatives rejected

- OpenAI's PyTorch/Triton references are educational rather than production
  runners and introduce a larger pinned dependency surface on ARM64.
- vLLM/SGLang require a separately qualified ARM64/CUDA stack and do not offer
  a simpler, already-reviewed Harmony/provenance path for this packet.
- NVIDIA NIM has signed ARM64 artifacts and is a viable predeclared fallback,
  but brings NGC entitlement/license and container-image acquisition concerns.
  It is not silently substituted after any project call.
- External APIs are out of scope and would violate the local execution and
  data-egress boundary.

## Primary sources

- https://openai.com/index/gpt-oss-model-card/
- https://openai.com/index/introducing-gpt-oss/
- https://github.com/openai/gpt-oss
- https://huggingface.co/openai/gpt-oss-120b
- https://ollama.com/library/gpt-oss:120b
- https://ollama.com/blog/gpt-oss
- https://github.com/ollama/ollama/releases/tag/v0.32.5
- https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- https://docs.nvidia.com/dgx/dgx-spark/software.html
- https://docs.nvidia.com/dgx/dgx-spark/release-notes.html
- https://build.nvidia.com/spark/open-webui

The runtime choice is a feasibility inference from these sources plus prior
local GPT-OSS interface evidence. Only the packet's on-Spark gates establish
current operational conformance.
