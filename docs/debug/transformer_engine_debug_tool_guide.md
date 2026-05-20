# Transformer Engine Debug Tool 使用指南

本文整理自 `docs/debug` 下的参考文档，以及 `transformer_engine/debug` 中的实现代码。目标是给使用者一份可以直接落地的 Markdown 版说明：如何接入调试工具、如何编写配置、各个 debug feature 适合解决什么问题，以及在分布式训练中需要注意什么。

## 1. 工具定位

Transformer Engine 的 precision debug tool 基于 NVIDIA 的 `nvdlfw_inspect` 包工作，目前主要支持 PyTorch。它会在 Transformer Engine 层的 GEMM 和量化路径中插入 hook，使用户可以按 layer、GEMM、tensor 粒度启用调试功能。

常见用途包括：

- 记录 activation、weight、gradient、output、wgrad、dgrad 等 tensor 的统计值。
- 对指定 GEMM 或整层禁用量化，强制使用高精度执行。
- 对特定 tensor 使用 per-tensor current scaling 或 fake quantization。
- 监控 FP8、MXFP8、FP8 block scaling、NVFP4 等量化格式的 underflow、overflow、MSE 和 scale 信息。
- 根据量化质量动态切换 GEMM 到高精度路径。
- 将 tensor dump 到磁盘，便于离线排查。

## 2. 快速接入流程

使用 debug tool 通常需要四步：

1. 编写一个 YAML 配置文件，声明哪些 layer 启用哪些 feature。
2. 在训练脚本中导入并初始化 `nvdlfw_inspect.api`。
3. 创建 TE layer 时尽量传入 `name=...`，方便在配置文件中稳定匹配 layer。
4. 每个训练 step 结束后调用一次 `debug_api.step()`。

最小接入示例：

```python
import nvdlfw_inspect.api as debug_api

debug_api.initialize(
    config_file="./config.yaml",
    feature_dirs=["/path/to/TransformerEngine/transformer_engine/debug/features"],
    log_dir="./log",
    default_logging_enabled=True,
)

# training loop
for step in range(num_steps):
    loss = train_one_step()
    debug_api.step()
```

如果需要 TensorBoard 输出，可以在初始化时传入 `tb_writer`：

```python
from torch.utils.tensorboard import SummaryWriter
import nvdlfw_inspect.api as debug_api

tb_writer = SummaryWriter("./tensorboard_dir/run1")

debug_api.initialize(
    config_file="./config.yaml",
    feature_dirs=["/path/to/TransformerEngine/transformer_engine/debug/features"],
    log_dir="./log",
    tb_writer=tb_writer,
)
```

## 3. YAML 配置结构

配置文件由一个或多个 section 组成。每个 section 负责选择一组 layer，并为这些 layer 启用一个或多个 feature。

```yaml
section_name:
  enabled: True
  layers:
    layer_types: [fc1, fc2]
  transformer_engine:
    LogTensorStats:
      enabled: True
      stats: [max, min, mean, std]
      tensors: [activation, weight]
      freq: 10
      start_step: 0
```

每个 section 的核心字段：

- `enabled`: 是否启用整个 section。
- `layers`: layer 选择规则。
- `transformer_engine`: Transformer Engine namespace 下的 feature 配置。

### 3.1 Layer 选择

推荐在创建 TE layer 时显式设置 `name=...`。配置中可以用两种方式选择 layer：

```yaml
# 使用正则匹配完整 layer 名称
section_by_regex:
  enabled: True
  layers:
    layer_name_regex_pattern: ".*(fc1|self_attention).*"
  transformer_engine:
    DisableQuantizationGEMM:
      enabled: True
      gemms: [wgrad]

# 使用子串匹配 layer 名称
section_by_type:
  enabled: True
  layers:
    layer_types: [layernorm_mlp, linear_qkv]
  transformer_engine:
    LogTensorStats:
      enabled: True
      stats: [cur_amax]
      tensors: [activation]
```

如果没有显式传入 `name`，可以调用 `debug_api.infer_and_assign_layer_names(model)` 自动推断，也可以依赖默认的 `Layer_n` 命名。对于 `TransformerLayer`，常见可匹配子层包括：

- `self_attention.layernorm_qkv`、`self_attn.linear_qkv`、`self_attn.proj`
- `inter_attn.*`
- `layernorm_mlp.fc1`
- `layernorm_mlp.fc2`

`GroupedLinear` 的底层 GEMM 名称通常是 `layer_name.gemm_n`。

### 3.2 GEMM 和 Tensor 选择

支持的 GEMM 名称：

- `fprop`
- `dgrad`
- `wgrad`

常见 tensor 名称：

- `activation`
- `weight`
- `gradient`
- `output`
- `wgrad`
- `dgrad`

简单列表写法：

```yaml
SomeFeature:
  enabled: True
  gemms: [fprop, dgrad]
  tensors: [activation, weight]
```

按 GEMM 或 tensor 设置不同参数时，可以使用结构化写法：

```yaml
SomeFeature:
  enabled: True
  gemms_struct:
    - gemm: fprop
      tensors:
        - activation
        - weight
      freq: 10
    - gemm: wgrad
      tensors_struct:
        - tensor: activation
          freq: 5
        - tensor: gradient
          freq: 20
```

## 4. 内置 Feature 速查

### 4.1 LogTensorStats

`LogTensorStats` 用于记录高精度 tensor 的基础统计值。支持 micro-batching；如果多个 forward/backward 对应一次 `debug_api.step()`，除 weight 外的 tensor 统计会被累计。

支持的统计项包括：

- `min`
- `max`
- `mean`
- `std`
- `l1_norm`
- `l2_norm`
- `cur_amax`
- `dynamic_range`
- `max_blockwise_dynamic_range`

示例：

```yaml
log_tensor_stats:
  enabled: True
  layers:
    layer_types: [layernorm_linear]
  transformer_engine:
    LogTensorStats:
      enabled: True
      stats:
        - max
        - min
        - mean
        - std
        - dynamic_range
        - max_blockwise_dynamic_range:
            block_size: 32
            dims: 1
      tensors: [activation, gradient, weight]
      freq: 10
      start_step: 10
```

### 4.2 LogFp8TensorStats

`LogFp8TensorStats` 用于记录量化后 tensor 的 FP8 相关统计。它可以记录当前训练 recipe 的统计，也可以额外模拟其他 recipe 的统计，但额外模拟会带来开销。

支持的 recipe 名称包括：

- `fp8_delayed_scaling`
- `fp8_current_scaling`
- `mxfp8`
- `fp8_block_scaling`

常见统计项包括：

- `underflows%`
- `overflows%`
- `scale_inv_min`
- `scale_inv_max`
- `mse`

统计名可以写成 `<recipe>_<stat>`，例如 `mxfp8_mse`；对于 `mxfp8` 和 `fp8_block_scaling`，还可以使用 `_columnwise` 后缀。

### 4.3 LogNvfp4TensorStats

`LogNvfp4TensorStats` 用于记录 NVFP4 量化 tensor 的统计。支持：

- `underflows%`
- `mse`

示例：

```yaml
log_nvfp4_stats:
  enabled: True
  layers:
    layer_types: [layernorm_linear]
  transformer_engine:
    LogNvfp4TensorStats:
      enabled: True
      tensors_struct:
        - tensor: activation
          stats: [underflows%, mse]
          freq: 1
        - tensor: gradient
          stats: [underflows%, mse]
          freq: 5
          start_step: 0
          end_step: 80
```

### 4.4 DisableQuantizationGEMM 和 DisableQuantizationLayer

`DisableQuantizationGEMM` 会让指定 GEMM 强制走高精度路径，适用于 FP8、NVFP4 等 TE 支持的量化格式。

```yaml
disable_wgrad:
  enabled: True
  layers:
    layer_types: [fc1]
  transformer_engine:
    DisableQuantizationGEMM:
      enabled: True
      gemms: [wgrad]
```

`DisableQuantizationLayer` 会禁用整层所有量化 GEMM：

```yaml
disable_layer_quantization:
  enabled: True
  layers:
    layer_types: [layernorm_mlp]
  transformer_engine:
    DisableQuantizationLayer:
      enabled: True
```

`DisableFP8GEMM` 和 `DisableFP8Layer` 仍然保留，但已经是兼容旧配置的 deprecated feature；新配置建议使用 `DisableQuantizationGEMM` 和 `DisableQuantizationLayer`。

### 4.5 AutoswitchGemm

`AutoswitchGemm` 会监控指定 tensor 的量化质量，并在指标超过阈值时，按 `(layer_name, gemm)` 临时切换到高精度 GEMM。判断逻辑是 OR：同一个 GEMM 中任一被监控 tensor 触发阈值，就会切换该 GEMM。

默认监控 tensor 可以从 GEMM 推断：

- `fprop` -> `activation`, `weight`
- `dgrad` -> `gradient`, `weight`
- `wgrad` -> `activation`, `gradient`

关键参数：

- `underflow_threshold_pct`: underflow 百分比阈值，默认 `5.0`。
- `mse_threshold`: 量化 MSE 阈值，默认 `1e-4`。
- `freq`: 采样间隔。
- `start_step` / `end_step` / `start_end_list`: 采样窗口。
- `allow_fp8_model_params_dequantized_weight`: 当模型参数以 FP8 保存时，是否允许 `fprop` / `dgrad` 使用临时反量化 weight 切到高精度。

示例：

```yaml
autoswitch_gemm:
  enabled: True
  layers:
    layer_types: [linear_qkv, linear_proj, linear_fc1, linear_fc2]
  transformer_engine:
    LogTensorStats:
      enabled: True
      stats: [max, min, mean, std, dynamic_range, cur_amax]
      tensors: [activation, gradient, weight]
      freq: 10
      start_step: 10
    AutoswitchGemm:
      enabled: True
      gemms: [fprop, dgrad, wgrad]
      tensors: [activation, weight, gradient]
      underflow_threshold_pct: 5
      mse_threshold: 0.1
      allow_fp8_model_params_dequantized_weight: True
      freq: 10
      start_step: 10
```

如果第 `n` 个采样 step 触发阈值，该 `(layer, gemm)` 会保持高精度直到 `n + freq - 1`。下一个采样 step 会刷新决策。

启用后会额外生成日志目录：

```text
<log_dir>/nvdlfw_inspect_autoswitchgemm_logs/nvdlfw_inspect_globalrank-<rank>.log
```

常见指标包括：

- `<layer>_<gemm>_<tensor>_underflow_pct`
- `<layer>_<gemm>_<tensor>_mse`
- `<layer>_<gemm>_quantized_enabled`
- `<layer>_<gemm>_disable_until_iter`
- `<layer>_<gemm>_switch_blocked_fp8_model_params`
- `<layer>_<gemm>_fp8_model_params_dequantized_fallback`
- `<layer>_<gemm>_final_decision`

使用 CUDA Graph 时，采样窗口和高精度窗口需要 eager 执行；量化窗口可以继续走 CUDA Graph，前提是训练框架支持对应路由。

### 4.6 DumpTensors

`DumpTensors` 会用 `torch.save()` 将 tensor 保存到磁盘。支持保存量化前的高精度 tensor，也支持保存量化后的 tensor。

输出目录：

```text
<log_dir>/tensor_dumps/rank_<rank>/iter_<iteration>/<layer>_<tensor>.pt
```

示例：

```yaml
dump_tensors:
  enabled: True
  layers:
    layer_name_regex_pattern: ".*(fc1|self_attention).*"
  transformer_engine:
    DumpTensors:
      enabled: True
      tensors_struct:
        - tensor: activation
          high_precision_tensor: True
          quantized_tensor: True
          freq: 100
        - tensor: weight
          high_precision_tensor: True
          quantized_tensor: False
          freq: 500
```

### 4.7 FakeQuant

`FakeQuant` 会先将指定 tensor fake quantize 到 FP8/MXFP8，再反量化回高精度，并强制 GEMM 走高精度路径。它适合评估量化误差对计算结果的影响。

支持格式：

- `FP8E4M3`
- `FP8E5M2`
- `MXFP8E4M3`
- `MXFP8E5M2`

```yaml
fake_quant:
  enabled: True
  layers:
    layer_types: [transformer_layer.layernorm_mlp.fc1]
  transformer_engine:
    FakeQuant:
      enabled: True
      quant_format: FP8E5M2
      gemms_struct:
        - gemm: fprop
          tensors: [activation, weight]
        - gemm: dgrad
          tensors: [gradient]
```

### 4.8 PerTensorScaling

`PerTensorScaling` 会对指定 tensor 使用 per-tensor current scaling。它只能在 `DelayedScaling` recipe 的 autocast 中使用。

```yaml
per_tensor_scaling:
  enabled: True
  layers:
    layer_types: [transformer_layer.self_attn.layernorm_q]
  transformer_engine:
    PerTensorScaling:
      enabled: True
      gemms: [dgrad]
      tensors: [weight, activation]
```

## 5. 日志输出

启用 `default_logging_enabled=True` 后，常见日志包括：

```text
<log_dir>/nvdlfw_inspect_logs/nvdlfw_inspect_globalrank-<rank>.log
<log_dir>/nvdlfw_inspect_statistics_logs/nvdlfw_inspect_globalrank-<rank>.log
```

其中：

- `nvdlfw_inspect_logs` 记录配置加载、layer/GEMM 路径、feature 调用等调试信息。
- `nvdlfw_inspect_statistics_logs` 记录 tensor 统计值。
- `nvdlfw_inspect_autoswitchgemm_logs` 记录 `AutoswitchGemm` 的 per-rank 指标。
- `tensor_dumps` 保存 `DumpTensors` 输出的 `.pt` 文件。

## 6. 分布式训练注意事项

在多 GPU 训练中，每个 rank 都需要调用 `debug_api.initialize(...)`，并使用同一份配置文件。

如果要记录 tensor 统计，建议显式设置统计 reduction group：

```python
import nvdlfw_inspect.api as debug_api

debug_api.set_tensor_reduction_group(group)
```

weight tensor 默认会在 tensor parallel group 内做统计归约。如果希望每个 rank 记录本地 weight shard，可以关闭该行为：

```python
from transformer_engine.debug import set_weight_tensor_tp_group_reduce

set_weight_tensor_tp_group_reduce(False)
```

分布式行为要点：

- `DisableQuantizationGEMM` 和 `DisableQuantizationLayer` 与单卡行为基本一致。
- `PerTensorScaling` 和 `FakeQuant` 会在每个 rank 上独立计算 scaling factor，因此 GPU 数量可能影响结果。
- logging feature 会先在本地计算统计，再在 `debug_api.step()` 时聚合并写入日志。
- micro-batching 场景下，activation、gradient 等 tensor 的统计会跨 microbatch 累计，weight 统计通常保持不变。
- TensorBoard 只会在传入 `tb_writer` 的 rank 上写入；pipeline parallel 场景通常需要在每个 pipeline group 中选择一个 rank 写 TensorBoard。

## 7. 常见排查建议

- 如果日志中没有目标 layer，先确认 layer 是否显式设置了 `name`，或者使用了正确的 `layer_types` / `layer_name_regex_pattern`。
- 如果统计日志为空，确认 feature section 和 feature 自身的 `enabled` 都是 `True`，并且训练 loop 每步调用了 `debug_api.step()`。
- 如果开销过大，优先增大 `freq`，缩小 `start_step` / `end_step` / `start_end_list` 采样窗口，或只选择少量 layer 和 tensor。
- 如果使用 `AutoswitchGemm`，建议让 companion inspection feature 使用相同的 `freq` 和采样窗口，避免采样时序不一致。
- 如果使用 FP8 model params 并希望 `AutoswitchGemm` 能切换 `fprop` / `dgrad`，需要设置 `allow_fp8_model_params_dequantized_weight: True`。
- 如果只想验证某个 GEMM 的量化影响，优先从 `DisableQuantizationGEMM` 或 `FakeQuant` 开始，配置范围尽量小。

## 8. 推荐配置模板

下面模板适合先快速观察一组 attention / MLP linear 的数值范围，再用 `AutoswitchGemm` 根据 underflow 和 MSE 自动切换高精度：

```yaml
debug_attention_mlp_linears:
  enabled: True
  layers:
    layer_types: [linear_qkv, linear_proj, linear_fc1, linear_fc2]
  transformer_engine:
    LogTensorStats:
      enabled: True
      stats: [max, min, mean, std, dynamic_range, cur_amax]
      tensors: [activation, gradient, weight]
      freq: 10
      start_step: 10

    AutoswitchGemm:
      enabled: True
      gemms: [fprop, dgrad, wgrad]
      tensors: [activation, weight, gradient]
      underflow_threshold_pct: 5
      mse_threshold: 0.1
      allow_fp8_model_params_dequantized_weight: True
      freq: 10
      start_step: 10
```

Megatron-LM 这类环境可以通过环境变量指定配置和日志目录：

```bash
export ENABLE_NVDFW_INSPECT=1
export NVDFW_CONFIG_FILE=/path/to/config.yaml
export NVDFW_LOG_DIR=/path/to/output/nvdlfw_logs
```
