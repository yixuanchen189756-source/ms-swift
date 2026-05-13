# MOPD Two-4090 Remote Teacher Runbook

This runbook verifies MOPD with two machines:

- Teacher machine: one RTX 4090, runs hidden-state teacher servers.
- Training machine: one RTX 4090, runs ms-swift MOPD training.

Start with `Qwen/Qwen3.5-0.8B` for all three teacher roles. After the path works, replace only the `codegen` teacher with a larger code teacher if the hardware can load it.

## Requirements

Both machines must use this modified ms-swift tree. Prefer mounting it into the container as:

```bash
-v /matt/MOPD/ms-swift:/workspace/ms-swift
```

Use the local source tree, not a system-installed `swift` command:

```bash
PYTHONPATH=/workspace/ms-swift python swift/cli/rlhf.py ...
```

The commands below assume:

```bash
/models/Qwen3.5-0.8B
```

Training machine must reach the teacher machine on ports `8001`, `8002`, and `8003`. Docker `--network host` is recommended.

## Teacher Machine

Start the container:

```bash
docker run --gpus all --network host \
  -v /matt/MOPD/ms-swift:/workspace/ms-swift \
  -v /models:/models \
  -it your_image bash
```

Inside the container:

```bash
cd /workspace/ms-swift
export PYTHONPATH=/workspace/ms-swift
```

Start three hidden-state teacher servers:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/train/rlhf/gkd/mopd_hidden_server.py \
  --model /models/Qwen3.5-0.8B \
  --host 0.0.0.0 \
  --port 8001 \
  --torch_dtype bf16
```

```bash
CUDA_VISIBLE_DEVICES=0 python examples/train/rlhf/gkd/mopd_hidden_server.py \
  --model /models/Qwen3.5-0.8B \
  --host 0.0.0.0 \
  --port 8002 \
  --torch_dtype bf16
```

```bash
CUDA_VISIBLE_DEVICES=0 python examples/train/rlhf/gkd/mopd_hidden_server.py \
  --model /models/Qwen3.5-0.8B \
  --host 0.0.0.0 \
  --port 8003 \
  --torch_dtype bf16
```

Expected:

```text
Uvicorn running on http://0.0.0.0:8001
Uvicorn running on http://0.0.0.0:8002
Uvicorn running on http://0.0.0.0:8003
```

## Training Machine

Start the container:

```bash
docker run --gpus all --network host \
  -v /matt/MOPD/ms-swift:/workspace/ms-swift \
  -v /models:/models \
  -v /data:/data \
  -it your_image bash
```

Inside the container:

```bash
cd /workspace/ms-swift
export PYTHONPATH=/workspace/ms-swift
export TEACHER_IP=<teacher_machine_ip>
```

Verify teacher connectivity:

```bash
python - <<'PY'
import io
import os
import requests
import torch

teacher_ip = os.environ['TEACHER_IP']
payload = {
    'input_ids': [[1, 2, 3, 4]],
    'attention_mask': [[1, 1, 1, 1]],
    'dtype': 'bf16',
}

for port in [8001, 8002, 8003]:
    url = f'http://{teacher_ip}:{port}/v1/mopd/hidden_states'
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    obj = torch.load(io.BytesIO(r.content), map_location='cpu')
    print(port, tuple(obj['hidden_states'].shape), obj['hidden_states'].dtype, obj['seq_lens'].tolist())
PY
```

Expected:

```text
8001 (1, 4, 1024) torch.bfloat16 [4]
8002 (1, 4, 1024) torch.bfloat16 [4]
8003 (1, 4, 1024) torch.bfloat16 [4]
```

Create a smoke dataset:

```bash
cat > /data/mopd_smoke.jsonl <<'EOF'
{"messages":[{"role":"user","content":"Solve 1+1."},{"role":"assistant","content":"2"}],"task":"reasoning"}
{"messages":[{"role":"user","content":"Write a Python add function."},{"role":"assistant","content":"def add(a, b):\n    return a + b"}],"task":"codegen"}
{"messages":[{"role":"user","content":"Plan how to inspect files and edit main.py."},{"role":"assistant","content":"Inspect the files, identify the target change, edit main.py, then run a focused check."}],"task":"agent"}
EOF
```

Run the 2-step smoke:

```bash
CUDA_VISIBLE_DEVICES=0 \
TRL_EXPERIMENTAL_SILENCE=1 \
PYTHONPATH=/workspace/ms-swift \
python swift/cli/rlhf.py \
  --rlhf_type gkd \
  --model /models/Qwen3.5-0.8B \
  --dataset /data/mopd_smoke.jsonl \
  --split_dataset_ratio 0 \
  --dataset_num_proc 1 \
  --dataloader_num_workers 0 \
  --dataset_shuffle false \
  --tuner_type lora \
  --mopd_enable true \
  --mopd_teacher_servers reasoning=http://${TEACHER_IP}:8001,codegen=http://${TEACHER_IP}:8002,agent=http://${TEACHER_IP}:8003 \
  --mopd_teacher_heads reasoning=/models/Qwen3.5-0.8B,codegen=/models/Qwen3.5-0.8B,agent=/models/Qwen3.5-0.8B \
  --mopd_teacher_weights 'reasoning=reasoning:0.75,codegen:0.05,agent:0.20;codegen=codegen:0.75,reasoning:0.05,agent:0.20;agent=agent:0.75,reasoning:0.05,codegen:0.20' \
  --mopd_task_column task \
  --mopd_default_task codegen \
  --mopd_hidden_dtype bf16 \
  --mopd_loss_chunk_size 8 \
  --beta 1.0 \
  --sft_alpha 0.0 \
  --lmbda 0.0 \
  --max_length 128 \
  --max_completion_length 16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_steps 2 \
  --logging_steps 1 \
  --save_strategy no \
  --report_to none \
  --output_dir /data/output/mopd_2gpu_smoke
```

Expected training-side signals:

```text
Using MOPD hidden-state teachers: ['agent', 'codegen', 'reasoning']
Loaded MOPD lm head `reasoning`
Loaded MOPD lm head `codegen`
Loaded MOPD lm head `agent`
global_step/max_steps: 2/2
```

Expected teacher-side signal:

```text
POST /v1/mopd/hidden_states HTTP/1.1 200 OK
```

## Negative Connectivity Check

To prove the run is really using remote MOPD teachers, change ports to unused ports and run `max_steps=1`:

```bash
--mopd_teacher_servers reasoning=http://${TEACHER_IP}:65501,codegen=http://${TEACHER_IP}:65502,agent=http://${TEACHER_IP}:65503
```

It should fail with connection refused. If it still succeeds, check that `swift.__file__` points to `/workspace/ms-swift/swift/__init__.py`.

## Larger Code Teacher

After the 0.8B smoke passes, replace only the `codegen` teacher:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/train/rlhf/gkd/mopd_hidden_server.py \
  --model /models/Qwen3.6-27B \
  --host 0.0.0.0 \
  --port 8002 \
  --torch_dtype bf16
```

Then point training to:

```bash
--mopd_teacher_heads reasoning=/models/Qwen3.5-0.8B,codegen=/models/Qwen3.6-27B,agent=/models/Qwen3.5-0.8B
```

Constraints:

- Teacher and student vocabularies must match.
- Teacher hidden size must match that teacher's LM head input dimension.
- Teacher LM head vocab size must match the student vocab size.
- A 27B bf16 model usually does not fit on a single 4090. Use a smaller code teacher, quantized/offloaded serving, or more GPUs.
