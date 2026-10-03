# ACE-Step 1.5 Docker Image (Turbo + 0.6B LM)

Fork of [ValyrianTech/ace-step-1.5](https://github.com/ValyrianTech/ace-step-1.5) optimized for GPUs with 8GB VRAM.

The upstream image ships the **base** DiT model (50 diffusion steps) and the **1.7B** language model. That's fine if you have 32GB+ VRAM, but on an 8GB card like a T1000 or GTX 1070, you'll either OOM or wait forever.

This fork swaps in:

- **Turbo DiT model** (`acestep-v15-turbo`): 8 diffusion steps instead of 50, roughly 6x faster with minimal quality loss
- **0.6B LM** (`acestep-5Hz-lm-0.6B`): saves ~2.2GB VRAM compared to the 1.7B, still provides semantic planning ("thinking mode")

The base model and 1.7B LM are excluded from the image entirely to keep it smaller.

## What changed from upstream

3 files, all straightforward:

**Dockerfile:**
- Main model download now excludes `acestep-v15-base/*` and `acestep-5Hz-lm-1.7B/*` (upstream excludes turbo instead)
- Downloads `ACE-Step/acestep-5Hz-lm-0.6B` separately
- `ACESTEP_CONFIG_PATH` points to turbo, `ACESTEP_LM_MODEL_PATH` points to 0.6B
- Placeholder directory flipped from turbo to base (the startup check wants both dirs to exist)

**start.sh:**
- Defaults updated to match: turbo DiT + 0.6B LM

That's it. Everything else (API server, Gradio UI, endpoints) is identical to upstream.

## Hardware notes

Tested on an NVIDIA T1000 8GB (Turing, SM 75). Key constraints for this class of GPU:

- No bf16 support (Ampere+ only), runs in FP16
- No Flash Attention (SM 80+ only), must use `pt` backend instead of `vllm`
- INT8 quantization and CPU offloading auto-enabled by ACE-Step's tier detection
- ACE-Step caps generation at ~60 seconds with LM enabled at this VRAM tier

For even less VRAM usage, you can disable the LM entirely with `ACESTEP_INIT_LLM=false` (cuts VRAM to under 5GB, 30-50% faster, but loses the lyrics enhancement feature).

## Quick Start

### Prerequisites

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA support
- HuggingFace token (for downloading gated models during build)

### Using the pre-built image

The fastest way. No build needed:

```bash
docker compose up -d
```

API available at `http://localhost:8000`. Test it:

```bash
curl http://localhost:8000/health
```

### Building from source

If you want to customize the image:

```bash
docker build --secret id=HF_TOKEN,env=HF_TOKEN -t acestep-api:latest .
```

## Deployment

### Docker Compose

The included [`docker-compose.yml`](docker-compose.yml) runs the API server with GPU support. Edit environment variables as needed, then:

```bash
docker compose up -d
```

### Kubernetes

Kustomize manifests in [`deploy/kubernetes/`](deploy/kubernetes/). Review and adjust to your cluster (storage class, ingress domain, API key), then:

```bash
kubectl apply -k deploy/kubernetes/
```

The deployment runs `acestep-api` only (no Gradio UI) to keep VRAM usage down. Both the API server and the Gradio UI load the full model stack independently, so running both on an 8GB card will OOM.

Key things to customize:
- `secret.yaml`: your API key
- `pvc.yaml`: storage class for your cluster
- `ingress.yaml`: your domain (uncomment in `kustomization.yaml` to enable)
- `deployment.yaml`: resource limits, node selector, `MAX_CUDA_VRAM` if you're on 8GB VRAM

### Terraform

Full Terraform config in [`deploy/terraform/`](deploy/terraform/). Deploys the namespace, secret, PVC, deployment, service, and optionally an ingress:

```hcl
# Minimal
terraform init && terraform apply

# With ingress
terraform apply -var='domain=acestep.example.com' -var='tls_issuer=letsencrypt-prod'
```

Copy `terraform.tfvars.example` to `terraform.tfvars` to configure storage class, GPU settings, node selectors, etc.

Get your API key after deploy:

```bash
terraform output -raw api_key
```

## CLI Tool

```bash
python generate_music.py \
  --api-url http://localhost:8000 \
  --caption "Upbeat indie pop with jangly guitars and energetic vocals" \
  --lyrics "[Verse 1]\nWalking down the street\nMusic in my feet\n\n[Chorus]\nWe are alive tonight" \
  --duration 90 \
  --output my_song.mp3
```

Handles task submission, polling, and file download automatically. See `python generate_music.py --help` for all options.

## API Endpoints

See the [ACE-Step API documentation](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/API.md) for full details.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | List available models |
| `/release_task` | POST | Create music generation task |
| `/query_result` | POST | Batch query task results |
| `/format_input` | POST | Format and enhance lyrics/caption via LLM |
| `/v1/audio` | GET | Download audio file |

### Usage with curl

Generate a 30-second instrumental track:

```bash
API=http://localhost:8000

# 1. Submit a generation task
TASK=$(curl -s -X POST "$API/release_task" \
  -H "Content-Type: application/json" \
  -d '{
    "caption": "warm acoustic guitar with soft piano, gentle and nostalgic",
    "lyrics": "",
    "duration": 30
  }')

echo "$TASK"
TASK_ID=$(echo "$TASK" | jq -r '.data.task_id')

# 2. Poll until done (check every 10s)
# NOTE: the param is task_id_list, NOT task_ids. Wrong name silently returns empty.
while true; do
  RESULT=$(curl -s -X POST "$API/query_result" \
    -H "Content-Type: application/json" \
    -d "{\"task_id_list\": [\"$TASK_ID\"]}")

  STATUS=$(echo "$RESULT" | jq -r '.data[0].status // 0')
  [ "$STATUS" = "1" ] && echo "Done!" && break
  [ "$STATUS" = "2" ] && echo "Failed!" && break
  echo "Processing..."
  sleep 10
done

# 3. Download the first audio file
# The result field is a JSON string containing an array of generated files.
# Audio filenames are NOT the same as the task ID.
FILE_URL=$(echo "$RESULT" | jq -r '.data[0].result' | jq -r '.[0].file')
curl -o output.mp3 "$API$FILE_URL"
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ACESTEP_CONFIG_PATH` | `/app/checkpoints/acestep-v15-turbo` | Path to DiT model |
| `ACESTEP_LM_MODEL_PATH` | `/app/checkpoints/acestep-5Hz-lm-0.6B` | Path to LM model |
| `ACESTEP_OUTPUT_DIR` | `/app/outputs` | Generated audio output directory |
| `ACESTEP_DEVICE` | `cuda` | Device (cuda, cpu, mps) |
| `ACESTEP_LM_BACKEND` | `pt` | LLM backend (vllm, pt) |
| `ACESTEP_API_HOST` | `0.0.0.0` | Server host |
| `ACESTEP_API_PORT` | `8000` | Server port |
| `ACESTEP_INIT_LLM` | `true` | Set to `false` to disable LM entirely |

## VAE precision on older NVIDIA cards

The pinned upstream selects FP16 for the VAE on cards without BF16 support. On a Turing GPU,
a controlled five-second decode produced 0 finite samples out of 480,000 with FP16; the same
latents produced 480,000 finite samples with FP32. This can turn a completed generation into
silent audio even when the diffusion latents are valid.

The image applies `patches/vae-float32.patch` before installing ACE-Step. It keeps the VAE in
FP32 on those cards, including every CPU offload/reload. The diffusion model keeps its existing
precision. BF16-capable NVIDIA, ROCm, CPU and Mac selection stay as upstream defines them.

There is no extra environment variable to set. `ACESTEP_DTYPE` is not read by this upstream
VAE selector; casting the model once is also insufficient because offloading casts it again.

Build and deploy a new image digest to pick up the fix. An existing Terraform deployment pinned
to an older digest keeps the old behavior until its image reference changes.

To check the patch against an upstream checkout without downloading models:

```bash
ACESTEP_UPSTREAM_DIR=/path/to/ACE-Step-1.5 python3 -m unittest discover -s tests
```

The image build fails if the patch no longer applies after an upstream bump.

### 120-second generation on T1000

The T1000 can produce NaNs in cuDNN's FP16 input convolution at 3,000 latent
frames (120 seconds). The same finite weights and input work with native FP16
convolution; the accepted 88-second control also works with cuDNN.

`patches/input-conv-native.patch` bypasses cuDNN only for the DiT input projection
on NVIDIA cards without native BF16 support. It keeps FP16 weights, CPU offload
and the existing FP32 VAE patch. The cuDNN context restores its previous settings
after the projection, including when it raises. ACE's API runs one generation
job at a time; the cuDNN setting is process-wide during that short call.

## License

See the [ACE-Step 1.5 repository](https://github.com/ace-step/ACE-Step-1.5) for license information.

## Recovering from a failed GPU load

With CPU offload enabled, an allocation failure while moving model weights could leave the
already-moved tensors on the GPU. The next request then started with less free VRAM. The
image applies `patches/offload-load-failure.patch` before installation so the existing CPU
offload cleanup also runs when loading fails, not only after generation starts. The original
error still reaches the caller. Generation is not silently retried.

This does not reserve VRAM against other applications or make an oversized request fit.
GPU Operator time slicing does not provide memory isolation. Use the turbo model, allow CPU
offload, and leave enough memory for the requested duration and any other workloads.

The patch tests execute the pinned upstream context without model downloads. They cover
partially loaded DiT, VAE and text-encoder weights, normal completion, generation failure,
and a successful request after a failed load. Run the same test command shown above.
