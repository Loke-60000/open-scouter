---
title: Install
nav_order: 2
---

# Install

## Option 1: pull the prebuilt image (GPU)

```bash
docker pull ghcr.io/loke-60000/open-scouter:latest
```

## Option 2: run from source (GPU)

```bash
git clone --recurse-submodules https://github.com/Loke-60000/open-scouter open-scouter
cd open-scouter
docker compose -f docker/docker-compose.yml up vllm
```

Both need an NVIDIA GPU and the NVIDIA Container Toolkit (`--gpus all`).

## Option 3: build an axis on CPU (no GPU)

You do not need a GPU to generate an axis from a model. Install the generate
extra:

```bash
git clone --recurse-submodules https://github.com/Loke-60000/open-scouter open-scouter
cd open-scouter
pip install -e ".[generate]"
```

Now see [Usage](usage.html).
