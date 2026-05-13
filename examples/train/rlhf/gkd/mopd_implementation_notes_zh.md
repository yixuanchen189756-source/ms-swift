# MOPD 实现说明

本文档说明当前分支在 ms-swift GKD/OPD 训练链路中加入的 MOPD 能力，便于 review 这次改动的设计、接口和验证方式。

## 背景与目标

ms-swift 原本的 OPD/GKD 路径主要支持单 teacher：训练时从一个 teacher 模型或一个 teacher server 获取 teacher 分布，然后让 student 匹配这个分布。

这次改动的目标是实现 DeepSeek V4/MAD-OPD 风格的多 teacher 蒸馏：

- 每个样本可以由多个 teacher 同时参与。
- 不同任务类型使用不同 teacher 权重，例如代码任务可以让代码 teacher 权重大，数学/推理和 Agent teacher 权重较小。
- teacher 侧可以部署在其他物理机上，训练机不需要加载完整 teacher 模型。
- 为减少网络带宽，远端 teacher 不传全词表 logits，只传最后一层 hidden states；训练机本地挂载对应 teacher 的 `lm_head`，再还原全词表 log-prob 分布。

## 新增能力

### 1. 多 teacher MOPD

新增 `--mopd_enable true` 后，GKD trainer 会走 MOPD 分支，而不是原来的单 teacher 分支。

训练时需要配置三类 teacher：

- `reasoning`：数学/推理 teacher。
- `codegen`：代码生成 teacher。
- `agent`：Agent 能力 teacher。

teacher 的名字不是写死的，但当前示例和 smoke 按这三个方向组织。实际训练可以扩展更多 teacher，只要 server、head、weight profile 中的 key 一致即可。

### 2. 按任务 profile 加权融合

实现方式是所有相关 teacher 同时参与，然后对 teacher log-prob 分布做加权融合。假设一个代码样本使用：

```text
codegen=codegen:0.75,reasoning:0.05,agent:0.20
```

则 trainer 会：

1. 向 `codegen`、`reasoning`、`agent` 三个 teacher server 请求该样本的 hidden states。
2. 使用本地挂载的三个 teacher `lm_head` 分别得到全词表 teacher log-probs。
3. 在 log 概率空间按权重融合：

```text
mixed_teacher_log_probs = logsumexp(log(weight_i) + teacher_i_log_probs)
```

4. 用原 GKD/OPD 的 KL/JSD 目标让 student 匹配混合 teacher 分布。

默认 profile 设计为：

```text
reasoning=reasoning:0.75,codegen:0.05,agent:0.20
codegen=codegen:0.75,reasoning:0.05,agent:0.20
agent=agent:0.75,reasoning:0.05,codegen:0.20
```

含义是每类任务都有一个主 teacher，同时保留其他 teacher 的弱监督信号。

### 3. 保留单 teacher 覆盖能力

数据集中如果有 `teacher_id` 字段，trainer 会把该样本退化为单 teacher 蒸馏：

```json
{"messages": [...], "task": "reasoning", "teacher_id": "codegen"}
```

这个样本会直接使用：

```text
codegen: 1.0
```

这样可以兼容已有“样本指定 teacher”的数据，也方便 debug 某个 teacher 的效果。

### 4. hidden-state 传输和本地 lm_head 挂载

如果远端 teacher 直接传全词表 logits，通信量约为：

```text
batch_size * seq_len * vocab_size * dtype_size
```

以 150K vocab、bf16 为例，每个 token 约 300KB，跨机器传输成本很高。

当前实现改为：

```text
远端 teacher: input_ids -> final hidden states
训练机: hidden states -> teacher lm_head -> full-vocab logits/log-probs
```

通信量变为：

```text
batch_size * seq_len * hidden_size * dtype_size
```

hidden size 通常远小于 vocab size，可以显著降低 teacher server 到训练机之间的带宽压力。

### 5. 支持 teacher hidden size 不同

每个 teacher 可以有自己的 hidden size。trainer 内部按 teacher 分组处理 hidden states，并使用该 teacher 对应的 `lm_head`。

限制是：

- teacher hidden size 必须匹配该 teacher `lm_head` 的输入维度。
- teacher `lm_head` 输出 vocab size 必须和 student logits vocab size 一致。
- 如果 vocab 不一致，当前 MOPD 分支会报错，避免静默蒸馏到错误 token 空间。

## 新增参数

主要参数如下：

```bash
--mopd_enable true
--mopd_teacher_servers reasoning=http://host1:8001,codegen=http://host2:8002,agent=http://host3:8003
--mopd_teacher_heads reasoning=/path/reasoning_model,codegen=/path/code_model,agent=/path/agent_model
--mopd_teacher_weights 'reasoning=reasoning:0.75,codegen:0.05,agent:0.20;codegen=codegen:0.75,reasoning:0.05,agent:0.20;agent=agent:0.75,reasoning:0.05,codegen:0.20'
--mopd_task_column task
--mopd_default_task codegen
--mopd_teacher_id_column teacher_id
--mopd_hidden_dtype bf16
--mopd_request_timeout 300
--mopd_loss_chunk_size 512
```

参数含义：

- `mopd_enable`：开启 MOPD。
- `mopd_teacher_servers`：teacher 名字到远端 hidden-state server 的映射。
- `mopd_teacher_heads`：teacher 名字到本地 lm_head 来源模型路径的映射。
- `mopd_teacher_weights`：任务类型到 teacher 权重分布的映射。
- `mopd_task_column`：数据集中表示任务类型的列，默认 `task`。
- `mopd_default_task`：样本缺少 task 时使用的默认 profile，默认 `codegen`。
- `mopd_teacher_id_column`：样本级 teacher 覆盖列，默认 `teacher_id`。
- `mopd_hidden_dtype`：请求 teacher hidden states 的 dtype，可选 `bf16`、`fp16`、`fp32`。
- `mopd_request_timeout`：请求 teacher server 的超时时间。
- `mopd_loss_chunk_size`：本地从 hidden states 计算 full-vocab log-probs 时的 token chunk 大小，用于控制显存峰值。

## 数据格式

训练数据仍然使用 ms-swift messages 格式，只额外保留 `task` 字段：

```json
{"messages":[{"role":"user","content":"Solve 1+1."},{"role":"assistant","content":"2"}],"task":"reasoning"}
{"messages":[{"role":"user","content":"Write a Python add function."},{"role":"assistant","content":"def add(a, b):\n    return a + b"}],"task":"codegen"}
{"messages":[{"role":"user","content":"Plan how to inspect files and edit main.py."},{"role":"assistant","content":"Inspect files, edit main.py, then run a check."}],"task":"agent"}
```

当前代码已调整数据预处理逻辑，确保 `task` 这类额外字段不会在 messages preprocess 后丢失。

## 远端 teacher server

新增示例 server：

```text
examples/train/rlhf/gkd/mopd_hidden_server.py
```

它提供接口：

```text
POST /v1/mopd/hidden_states
```

请求字段：

```json
{
  "input_ids": [[1, 2, 3, 4]],
  "attention_mask": [[1, 1, 1, 1]],
  "dtype": "bf16"
}
```

响应内容是 torch 序列化后的对象，包含：

- `hidden_states`：最后一层 hidden states。
- `seq_lens`：每个样本的有效序列长度。

训练端会自动把响应解析为 tensor，并按 teacher 权重 profile 组织为 MOPD loss 所需的输入。

## 训练流程

推荐先使用小模型跑通完整链路，例如三个 teacher 都用 `Qwen/Qwen3.5-0.8B`：

```bash
python swift/cli/rlhf.py \
  --rlhf_type gkd \
  --model /models/Qwen3.5-0.8B \
  --dataset /data/mopd_smoke.jsonl \
  --mopd_enable true \
  --mopd_teacher_servers reasoning=http://${TEACHER_IP}:8001,codegen=http://${TEACHER_IP}:8002,agent=http://${TEACHER_IP}:8003 \
  --mopd_teacher_heads reasoning=/models/Qwen3.5-0.8B,codegen=/models/Qwen3.5-0.8B,agent=/models/Qwen3.5-0.8B \
  --mopd_teacher_weights 'reasoning=reasoning:0.75,codegen:0.05,agent:0.20;codegen=codegen:0.75,reasoning:0.05,agent:0.20;agent=agent:0.75,reasoning:0.05,codegen:0.20' \
  --mopd_task_column task \
  --mopd_default_task codegen \
  --max_steps 2 \
  --save_strategy no \
  --report_to none
```

链路跑通后，可以只替换代码 teacher，例如：

```bash
codegen=/models/Qwen3.6-27B
```

并让 `codegen` teacher server 使用更大的代码模型。

完整两机部署步骤见：

```text
examples/train/rlhf/gkd/mopd_2gpu_remote_teacher_runbook.md
```

## 代码改动概览

核心改动集中在：

- `swift/rlhf_trainers/gkd_trainer.py`：MOPD teacher 请求、hidden-state 解析、本地 lm_head 加载、teacher 加权融合、MOPD loss。
- `swift/arguments/rlhf_args.py` 和 `swift/rlhf_trainers/arguments.py`：新增 MOPD CLI 参数和参数校验。
- `swift/dataset/preprocessor/core.py`：保留样本级 `task` 等额外字段。
- `examples/train/rlhf/gkd/mopd_hidden_server.py`：远端 hidden-state teacher server。
- `examples/train/rlhf/gkd/mopd.sh`：MOPD 训练示例。
- `tests/train/test_mopd.py`：MOPD 参数解析、lm_head 加载、hidden-state 响应解析、单 teacher loss、多 teacher 加权融合 loss 等单元测试。

## 已验证内容

当前分支已经做过以下 smoke/静态验证：

- `git diff --check`
- 相关 Python 文件 `py_compile`
- `tests/train/test_mopd.py` 中测试函数的直接调用
- 使用 `Qwen/Qwen3.5-0.8B` 跑通本地 MOPD smoke
- 三个远端 teacher server 均收到 `POST /v1/mopd/hidden_states`
- 错误端口的负向 smoke 会按预期连接失败，证明训练确实依赖远端 MOPD teacher

本地真实 smoke 输出目录：

```text
/matt/MOPD/output/mopd_smoke_qwen35_08b_real/v0-20260512-123224
```

该 smoke 完成了 2 step 训练，说明 hidden-state teacher server、本地 lm_head、task 权重融合和 loss 回传链路可以跑通。

## 当前限制和注意事项

- 训练机必须能访问所有 teacher server 的端口。
- teacher server 和训练机需要使用同一套 tokenizer/token id 语义。
- `mopd_teacher_heads` 必须和远端 teacher 模型匹配，不能随意混用。
- 大 teacher 例如 `Qwen3.6-27B` 是否能在 RTX 4090 上运行，取决于量化、dtype、上下文长度和推理框架；建议先用 0.8B smoke 跑通网络和代码链路。
- 该实现目前聚焦 decoder-only text 训练链路，未扩展 OPSD teacher prompt 和多模态 teacher hidden-state 传输。
- 如果多个 teacher 部署在同一张 4090 上，需要控制并发和 batch size，避免三个 teacher server 同时抢显存。
