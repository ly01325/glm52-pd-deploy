# GLM 5.2 高性能部署最佳实践

> 适用范围：GLM-5.2-NVFP4、8 GPU/实例、TP8 + EP、vLLM 0.28、Dynamo 1.5.1。  
> 已验证组合：Prefill 使用 B12X PCIe DMA，Decode 使用 FlashInfer PCIe IPC，P/D 之间通过 NIXL/UCX 传输 KV Cache。  
> 本文中的 IP、网卡名、吞吐和并发数据来自当前测试环境，迁移到其他机器前必须重新完成拓扑、单池容量和端到端验证。

## 1. 结论先行

推荐部署形态如下：

```text
Client
  |
  v
Dynamo Frontend / Router
  |                         control plane: etcd + NATS
  +--> Prefill Worker(s)    B12X PCIe DMA / TP8 + EP
           |
           | NIXL + UCX: KV Cache transfer
           v
       Decode Worker(s)     FlashInfer PCIe IPC / TP8 + EP
```

部署时必须遵循五条原则：

1. **先验证硬件拓扑和 NCCL 基线，再测试模型。** 不要直接复制其他机器的 `NCCL_P2P_LEVEL`、HCA 或 SHM 配置。
2. **P、D 必须独立压测。** 使用相同模型参数、请求分布、SLO 和运行时版本，分别得到单个 P/D 实例的可持续请求吞吐。
3. **按容量而不是 GPU 数量确定配比。** 理论比例为 `N_D / N_P ≈ C_P / C_D`，然后验证相邻整数配置。
4. **P/D 采用不同通信后端。** 当前环境中 Prefill 适合 B12X，Decode 适合 FlashInfer PCIe IPC；EP 通信仍可能回退到 PYNCCL，必须从日志确认。
5. **Decode 强制使用 PIECEWISE CUDA Graph。** 当前 GLM 5.2 + P/D 分离场景中，FULL graph 会破坏外部 KV 的等待/状态恢复，导致 MTP acceptance 严重下降。

当前测试环境的容量结论：

| 指标 | 单 Prefill，B12X | 单 Decode，FlashInfer IPC | 理论 D/P 比例 |
|---|---:|---:|---:|
| Gate A | 0.7603 req/s | 0.9413 req/s | 0.8076 |
| Gate B | 0.7526 req/s | 0.9413 req/s | 0.7996 |

因此当前 workload 的容量匹配点约为 **1P:0.8D**，大规模可从 **5P:4D** 起步；小规模使用 **1P1D** 更稳妥，但 Decode 约有 20% 容量余量，系统主要受 Prefill 限制。

## 2. 上线流程总览

建议按以下顺序执行，每一步未通过都不要进入下一步：

1. 固化软件、模型、容器镜像、驱动和固件版本。
2. 采集 GPU、PCIe、NUMA、NIC/HCA 和 UCX 拓扑。
3. 建立默认 NCCL 基线，并验证显式 NCCL 参数确实更优。
4. 单独压测 Prefill，得到 `C_P`。
5. 单独压测 Decode，得到 `C_D`。
6. 计算 P/D 理论比例，选择相邻整数候选。
7. 启动 etcd、NATS、Dynamo Frontend。
8. 启动 Prefill，确认注册和健康后再启动 Decode。
9. 从日志验证实际后端、NIXL、CUDA Graph 和 MTP 状态。
10. 对候选 P/D 比例执行端到端阶梯压测。
11. 留出容量余量，完成长稳、故障和回滚验证。

## 3. 前置条件与配置冻结

### 3.1 必须记录的版本

每次测试和生产变更都应保存以下信息：

```bash
nvidia-smi
nvidia-smi -q
python3 -m pip freeze
docker image inspect <runtime-image>
sha256sum <model-files-or-manifest>
```

至少冻结：

- GPU 型号、数量、VBIOS、驱动、CUDA 和 NCCL 版本。
- vLLM、Dynamo、FlashInfer、B12X、NIXL 和 UCX 版本。
- 模型目录、权重版本、量化格式和 tokenizer。
- TP/EP、KV Cache dtype、MTP token 数、CUDA Graph 模式。
- Prefix Cache、最大上下文、最大批处理 token、显存利用率。
- 请求输入/输出长度分布、并发、到达模型、随机种子和 SLO。

### 3.2 当前验证配置

| 项目 | 配置 |
|---|---|
| 模型 | `/models/nvidia/GLM-5.2-NVFP4` |
| 并行 | TP8 + Expert Parallel |
| KV Cache | `fp8_e4m3` |
| 推测解码 | MTP3 |
| 最大模型长度 | 66,564 |
| 最大批处理 token | 16,384 |
| GPU memory utilization | 0.92 |
| P/D KV 传输 | NIXL Connector + UCX |
| Prefill TP AllReduce | B12X PCIe DMA |
| Decode TP AllReduce | FlashInfer PCIe IPC |
| Decode CUDA Graph | PIECEWISE |

## 4. NCCL 与硬件拓扑验证

### 4.1 为什么先验证 NCCL

B12X 和 FlashInfer 只优化符合条件的通信；未覆盖的消息、EP group 或异常路径仍可能落到 PYNCCL/NCCL。NCCL 配置错误会表现为吞吐下降、抖动、错误的 Socket fallback，甚至被误判为模型或 P/D 架构问题。

### 4.2 采集物理拓扑

在每台 P/D 节点执行：

```bash
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
lspci -tv
numactl -H
ibdev2netdev
ibstat
ucx_info -d
```

重点确认：

- 8 张 GPU 是否处于预期 PCIe Root Complex/CPU NUMA 节点下。
- GPU 间 P2P read/write 是否可用，是否存在跨 PHB/SYS 路径。
- HCA/NIC 与 GPU 的 NUMA 和 PCIe 距离是否合理。
- `mlx5_0:1` 等 UCX 设备是否真实存在、Active 且映射到正确业务网口。
- 容器内看到的设备、网卡和 NUMA 信息是否与宿主机一致。

### 4.3 建立默认 NCCL 基线

先尽量不设置强制调优变量，仅开启诊断：

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,TUNING
export NCCL_TOPO_DUMP_FILE=/tmp/nccl-topo.xml
```

运行 `nccl-tests` 的 AllReduce，覆盖模型实际消息范围，而不是只看最大带宽：

```bash
./build/all_reduce_perf -b 8 -e 512M -f 2 -g 8
```

建议额外聚焦：

- Decode 常见的小到中等消息区间。
- FlashInfer PCIe IPC 候选区间，例如 4–16 MiB。
- B12X 候选的大消息区间，例如 64–256 MiB。
- 当前 B12X 配置的实际 dispatch 范围 8–256 MiB。

这些范围只是当前机器上的经验值，最终 crossover 必须以目标拓扑实测为准。

### 4.4 NCCL 参数验证原则

将默认结果与显式配置逐项 A/B，不要一次修改多个变量。重点参数包括：

```bash
NCCL_NET=IB
NCCL_IB_DISABLE=0
NCCL_P2P_LEVEL=PHB       # 或 SYS；必须根据拓扑验证
NCCL_SHM_DISABLE=1       # 只有实测更优时才使用
NCCL_NET_DISABLE_INTRA=1 # 只有目标运行时语义和实测均确认时使用
NCCL_PROTO=LL            # 对小消息可能有利，需实测
```

验收条件：

- 日志使用了目标 HCA/网口，没有意外回退 Socket。
- GPU P2P 路径符合 `nvidia-smi topo -m` 的预期。
- 各消息大小的延迟和带宽曲线合理，无断崖式异常。
- 连续多轮结果稳定，建议主要区间波动不超过约 3%。
- 无 NCCL warning、timeout、CUDA error、链路重试或进程退出。
- 在真实 GLM 5.2 workload 下，PYNCCL reference 能稳定通过质量和性能门槛。

> 注意：当前 Prefill B12X 脚本使用 `NCCL_P2P_LEVEL=PHB`，当前 Decode fallback 使用 `NCCL_P2P_LEVEL=SYS`、`NCCL_PROTO=LL`。这是当前硬件上的验证结果，不是通用默认值。

## 5. 如何单独测试 Prefill 和 Decode

### 5.1 先定义统一的业务门槛

单测与端到端测试必须使用同一套 SLO。当前实验使用：

| 门槛 | TTFT | 用户侧输出速度 |
|---|---|---|
| Gate A | p95 ≤ 10 s | p99 > 30 tok/s |
| Gate B | p50 < 5 s | p50 > 55 tok/s，p99 > 25 tok/s |

当前长上下文 workload：

- Shared prefix：58,944 tokens。
- Suffix：6,592 tokens。
- 总输入：65,536 tokens。
- 输出：1,024 tokens。
- 目标 Prefix Cache reuse：约 89.94%。
- `temperature=0`、`ignore_eos=true`、固定 seed。

如果生产流量存在多种长度，应按流量占比分桶测试，而不是只使用平均长度。

### 5.2 Prefill 单池测试

目标是得到一个 Prefill 实例在满足 TTFT 门槛时的最大可持续请求率 `C_P`。

推荐方法：

1. 使用与生产完全一致的模型、TP/EP、B12X、KV dtype、MTP 和 Prefix Cache 配置。
2. 让请求经过真实 tokenizer、调度器和 Prefix Cache 路径。
3. 最理想的是把生成的 KV 通过 NIXL 发给一个轻量接收端；若单测省略 KV 传输，应单独测量传输成本并对 `C_P` 打折。
4. 从低并发开始阶梯增加 concurrency/arrival rate。
5. 每个点至少 1 次预热，正式点建议 100 请求；测试点之间重启或清空 KV，避免前一轮缓存污染。
6. 取满足 SLO 的最高稳定请求率，而不是最大 token/s 峰值。

当前实测：

| Gate | 最大通过点 | `C_P` | 边界说明 |
|---|---:|---:|---|
| A | C6 | 0.760252 req/s | C6 TTFT p95 8.455 s；C7 为 10.985 s，失败 |
| B | C3 | 0.752638 req/s | C3 TTFT p50 4.134 s；C4 为 5.500 s，失败 |

不要使用 C9 的 50,355.94 total tok/s 峰值作为容量，因为它不满足目标延迟门槛。

### 5.3 Decode 单池测试

目标是得到一个 Decode 实例在满足输出速度门槛时的最大可持续请求率 `C_D`。

推荐方法：

1. 预先填充完整的 65,536-token KV，避免 Decode 单测混入 Prefill 计算。
2. 使用与生产相同的 NIXL KV 形态、KV dtype、MTP3、FlashInfer IPC 和 PIECEWISE CUDA Graph。
3. 固定输出 1,024 tokens，并确保所有请求完成。
4. 阶梯增加并发；观察用户侧 tok/s、inter-token latency、GPU 利用率和 MTP acceptance。
5. 排除开始和结束阶段的非稳态样本。当前测试对 100 个 completion 去掉前 16 和后 16，使用中间 68 个计算 steady req/s。

当前 FlashInfer PCIe IPC Decode 在 Gate A/B 下均可达到：

```text
C_D = 0.941321 req/s，最大通过并发 C16
```

### 5.4 质量检查不能省略

性能通过但输出错误不能上线。至少检查：

- 请求成功率为 100%。
- 无空输出、异常结束、重复 token 或长度异常。
- MTP acceptance 与 IFB/NCCL reference 同量级。
- 当前最终配置 MTP acceptance 为 95.64%，IFB reference 为 94.54%。
- 无 traceback、engine death、CUDA/NCCL fatal 或 OOM。

## 6. 如何计算 P/D 配比

### 6.1 容量模型

设：

- `C_P`：单个 Prefill 实例在 SLO 内的稳定请求率。
- `C_D`：单个 Decode 实例在 SLO 内的稳定请求率。
- `N_P`：Prefill 实例数。
- `N_D`：Decode 实例数。

系统理论容量为：

```text
lambda_max = min(N_P * C_P, N_D * C_D)
```

容量匹配比例为：

```text
N_D / N_P ≈ C_P / C_D
```

代入当前数据：

```text
Gate A: 0.760252 / 0.941321 = 0.807644
Gate B: 0.752638 / 0.941321 = 0.799555
```

因此理论上约为 `1P:0.8D`，可映射为：

- 小规模：`1P1D`，简单且有 Decode 余量。
- 中等规模：对比 `4P3D`、`5P4D` 和 `1:1`。
- 生产规模：按实测容量向上取整，并给瓶颈池预留 15%–30% 余量。

### 6.2 为什么理论值不能直接上线

理论模型没有完整包含：

- Frontend 路由和排队成本。
- Prefill 到 Decode 的 KV 传输与同步。
- Cache hit ratio 波动和 prefix 分布变化。
- 请求长度长尾、突发到达和租户不均衡。
- P/D worker 故障、重试和滚动升级余量。

因此必须对理论值附近的整数网格做端到端测试。判断瓶颈时：

- Prefill 队列持续增长、Decode 有余量：增加 P，不要盲目增加 D。
- Decode 活跃序列饱和、输出速度跌破门槛：增加 D。
- 两边利用率都不高但 TTFT 高：检查 KV 传输、路由、cache miss 和网络。

当前 all-NCCL 1P1D 中增加到 1P2D 后吞吐仅约 0.368 req/s，未得到对应提升，说明瓶颈仍在单个 Prefill。这是“按容量扩容而不是按节点对称扩容”的直接证据。

## 7. 基于 Dynamo 部署异构 P/D

### 7.1 控制面

准备 etcd、NATS 和 Dynamo Frontend。示例（请替换为你自己的控制面地址）：

```bash
export ETCD_ENDPOINTS=http://<CONTROL_HOST>:2379
export NATS_SERVER=nats://<CONTROL_HOST>:4222
export DYN_DISCOVERY_BACKEND=etcd

python3 -m dynamo.frontend --http-port 8000
```

生产中应补充进程守护、健康检查、日志轮转和鉴权；不要把示例 IP 当成固定配置。

### 7.2 Prefill：B12X PCIe DMA

关键环境变量：

```bash
export VLLM_ALLREDUCE_USE_B12X_PCIE_DMA=1
export VLLM_B12X_DMA_MIN_BYTES=8388608
export VLLM_B12X_DMA_MAX_BYTES=268435456
export VLLM_B12X_DMA_FP8=ring

export VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_NCCL_SYMM_MEM=0

export VLLM_NIXL_SIDE_CHANNEL_HOST=<PREFILL_KV_IP>
export VLLM_NIXL_SIDE_CHANNEL_PORT=20097
export UCX_NET_DEVICES=<PREFILL_UCX_DEVICE>
export DYN_SYSTEM_PORT=8082
```

启动参数模板：

```bash
python3 -m dynamo.vllm \
  --model /models/nvidia/GLM-5.2-NVFP4 \
  --served-model-name glm52 \
  --disaggregation-mode prefill \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --max-model-len 66564 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.92 \
  --no-enable-flashinfer-autotune \
  --spec-method mtp \
  --spec-tokens 3
```

Prefill 验收日志应包含：

```text
B12x PCIe DMA all-reduce enabled
Using ['B12X_PCIE_DMA', 'PYNCCL'] ... group 'tp:0'
```

同时确认 EP group 的实际后端。当前环境 EP 使用 PYNCCL，因此 NCCL 基线仍然重要。

### 7.3 Decode：FlashInfer PCIe IPC

关键环境变量：

```bash
export VLLM_ALLREDUCE_USE_B12X_PCIE_DMA=0
export VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_NCCL_SYMM_MEM=0

export VLLM_NIXL_SIDE_CHANNEL_HOST=<DECODE_KV_IP>
export VLLM_NIXL_SIDE_CHANNEL_PORT=20098
export UCX_NET_DEVICES=<DECODE_UCX_DEVICE>
export DYN_SYSTEM_PORT=8083
```

启动参数模板：

```bash
python3 -m dynamo.vllm \
  --model /models/nvidia/GLM-5.2-NVFP4 \
  --served-model-name glm52 \
  --disaggregation-mode decode \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --max-model-len 66564 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.92 \
  --spec-method mtp \
  --spec-tokens 3 \
  --no-enable-flashinfer-autotune \
  -cc.cudagraph_mode=PIECEWISE
```

Decode 验收日志应包含：

```text
Using ['FLASHINFER_PCIE_IPC', 'PYNCCL'] ... group 'tp:0'
cudagraph_mode ... PIECEWISE
```

`-cc.cudagraph_mode=PIECEWISE` 是当前组合的硬性要求。历史测试中 FULL graph 下 acceptance 仅约 14%–15%；PIECEWISE 恢复到约 95%。根因是 FULL graph 绕过了外部 KV 等待/状态恢复相关路径，并非 FlashInfer 算术本身错误。

### 7.4 启动顺序

严格按以下顺序：

1. 启动 etcd 和 NATS，确认端口和健康状态。
2. 启动 Dynamo Frontend。
3. 启动 Prefill worker。
4. 确认 Prefill 已加载模型、完成 warmup、注册到 Dynamo 且 Ready。
5. 再启动 Decode worker。
6. 确认 Decode 完成注册，发送一条小流量 smoke request。
7. 验证请求确实经历 Prefill、NIXL KV 传输和 Decode，而不是异常 fallback。

不要并行冷启动 P/D。当前问题定位经验表明，先等待 Prefill 注册完成再启动 Decode 更可靠。

### 7.5 Prefix Cache 开关

vLLM 0.28 中若要明确关闭 Prefix Cache，应传：

```bash
--no-enable-prefix-caching
```

不要假设省略 `--enable-prefix-caching` 就等价于关闭。测试报告必须记录实际 EngineConfig 中的 `enable_prefix_caching`。

## 8. 端到端验证与上线门槛

### 8.1 测试矩阵

对每个候选 P/D 比例至少执行：

- Cache cold 与目标 cache reuse 两种场景。
- 从低并发到过载点的完整阶梯。
- 固定长度与生产长度分布。
- 100 请求正式点，100% completion。
- 每个正式点前重启 P/D 或可靠清空 KV。
- 至少一轮长稳测试和一轮 worker 重启测试。

### 8.2 当前端到端参考结果

| 部署 | Gate A | Gate B | 结论 |
|---|---:|---:|---|
| v0.28 IFB + NCCL | 0.195212 req/s | 0.192559 req/s | 基线 |
| v0.28 1P1D，全 NCCL | 0.336496 req/s | 0.306463 req/s | P/D 分离已有收益 |
| v0.28 1P1D，B12X P + FI D + PIECEWISE | 0.585720 req/s | 0.647078 req/s | 当前推荐组合 |

相对 IFB：

- Gate A：约 `3.00x`，提升约 `200%`。
- Gate B：约 `3.36x`，提升约 `236%`。

端到端容量低于单池理论上限是正常现象，因为真实路径增加了 KV 传输、路由、排队和同步成本。容量规划应优先使用端到端实测结果，单池模型用于解释瓶颈和规划扩容方向。

### 8.3 上线检查清单

- [ ] 软件、容器和模型版本已冻结。
- [ ] GPU/PCIe/NUMA/NIC/UCX 拓扑已归档。
- [ ] NCCL 默认和显式配置完成 A/B 测试。
- [ ] 无意外 Socket fallback，HCA 和 UCX 设备正确。
- [ ] Prefill 单池容量 `C_P` 已按 SLO 确定。
- [ ] Decode 单池容量 `C_D` 已按 SLO 确定。
- [ ] 已计算理论比例并测试相邻整数配置。
- [ ] Prefill 日志确认 TP 使用 B12X。
- [ ] Decode 日志确认 TP 使用 FlashInfer PCIe IPC。
- [ ] Decode EngineConfig 确认 PIECEWISE。
- [ ] EP group 后端和 NCCL fallback 已确认。
- [ ] NIXL side-channel IP、端口和 UCX 设备正确。
- [ ] MTP acceptance、成功率和输出质量通过。
- [ ] 端到端 Gate A/B、长稳和故障恢复通过。
- [ ] 生产容量留有 15%–30% 余量。

## 9. 常见误区与排障

| 现象 | 常见原因 | 优先动作 |
|---|---|---|
| 增加 D 但吞吐不提升 | Prefill 已饱和 | 查看 P 队列和 TTFT，增加 P 或优化 Prefill |
| MTP acceptance 降到十几个百分点 | Decode 使用 FULL graph | 切换 `PIECEWISE`，重启并复测 |
| 配了 FI/B12X 但性能没变化 | 未命中 dispatch 范围或环境变量未生效 | 从日志确认实际 backend 和消息大小 |
| TP 加速了但整体仍慢 | EP 仍由 PYNCCL 承担 | 检查 `ep:0` 日志和 NCCL 基线 |
| TTFT 抖动、GPU 利用率不高 | KV 传输、路由或网络问题 | 检查 NIXL/UCX、队列、cache hit 和 HCA |
| 单池预测明显高于 E2E | 单测省略 KV/路由成本 | 在 P 单测加入 KV receiver，或使用折扣系数 |
| 更换机器后性能回退 | 直接复制 NCCL/UCX 参数 | 重新执行拓扑和 nccl-tests 验证 |
| 关闭 Prefix Cache 后仍命中 | 只删除 enable 参数 | 显式使用 `--no-enable-prefix-caching` |

## 10. 性能调优决策顺序

为避免把多个变量混在一起，建议按以下顺序调优：

1. **正确性**：输出、MTP acceptance、KV 完整性、100% completion。
2. **硬件通信**：PCIe/P2P/NUMA/HCA/UCX/NCCL。
3. **运行态后端**：确认 B12X、FlashInfer 和 PYNCCL 的实际 dispatch。
4. **单池容量**：分别测 `C_P`、`C_D`。
5. **P/D 比例**：计算理论比，测试相邻整数网格。
6. **调度与缓存**：Prefix Cache、长度分桶、路由和排队。
7. **生产余量**：突发、长尾、故障和滚动升级。

每轮实验只修改一个核心变量，并保存启动命令、环境变量、完整日志和原始请求结果。没有运行态日志证据的“已启用优化”不能作为结论。

## 11. 证据索引

本文数据来自作者的内部测试环境（总报告、根因报告、正式 NCCL/P-D 报告、测试索引及 Prefill/Decode 启动脚本）。公开版本不包含这些原始数据与内部路径；启动脚本请参考 `templates/`。
