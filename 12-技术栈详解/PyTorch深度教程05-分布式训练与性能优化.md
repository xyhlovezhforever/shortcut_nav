# PyTorch深度教程（五）：分布式训练与性能优化

> **前置要求**：完成前四篇教程
> **核心目标**：掌握大规模训练与极致性能优化

---

## 第一部分：分布式训练基础

### 1.1 并行计算模式

#### 1.1.1 数据并行（Data Parallelism）

```python
"""
数据并行原理：

1. 模型复制到多个GPU
2. 将batch数据分割到各GPU
3. 每个GPU独立前向传播
4. 收集梯度并平均
5. 同步更新参数

数学表达：
g = (1/N) Σᵢ ∇L(θ, Dᵢ)
θ ← θ - α·g
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

class DataParallelismBasics:
    """数据并行基础"""

    def simple_data_parallel(self):
        """
        简单数据并行（nn.DataParallel）

        单进程多线程，适合单机多卡
        """
        model = MyModel()

        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个GPU")
            model = nn.DataParallel(model)

        model = model.cuda()

        # 训练循环
        for inputs, targets in dataloader:
            inputs = inputs.cuda()
            targets = targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 局限性：
        # 1. 单进程瓶颈（GIL）
        # 2. GPU 0负载不均
        # 3. 不支持多机

    def distributed_data_parallel(self):
        """
        分布式数据并行（DistributedDataParallel）

        多进程，支持多机多卡
        性能优于DataParallel
        """

        # 初始化进程组
        def setup(rank, world_size):
            """
            rank: 当前进程的排名
            world_size: 总进程数
            """
            import os
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'

            # 初始化进程组
            dist.init_process_group(
                backend='nccl',  # NVIDIA GPU用nccl，CPU用gloo
                rank=rank,
                world_size=world_size
            )

        def cleanup():
            dist.destroy_process_group()

        def train_ddp(rank, world_size):
            setup(rank, world_size)

            # 每个进程使用不同的GPU
            torch.cuda.set_device(rank)

            # 创建模型并移到GPU
            model = MyModel().to(rank)

            # 包装为DDP
            ddp_model = DDP(model, device_ids=[rank])

            # 分布式采样器
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=32,
                sampler=train_sampler
            )

            optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)

            # 训练循环
            for epoch in range(100):
                train_sampler.set_epoch(epoch)  # 重要！打乱数据

                for inputs, targets in train_loader:
                    inputs = inputs.to(rank)
                    targets = targets.to(rank)

                    outputs = ddp_model(inputs)
                    loss = criterion(outputs, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            cleanup()

        # 启动多进程
        import torch.multiprocessing as mp
        world_size = torch.cuda.device_count()
        mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)

    def gradient_synchronization(self):
        """
        梯度同步机制

        All-Reduce操作：
        - Ring-AllReduce: 环形通信
        - Tree-AllReduce: 树形通信
        """

        # 手动实现All-Reduce概念
        def all_reduce_sum(tensor, group=None):
            """
            所有进程的tensor求和并广播结果

            步骤：
            1. 每个进程发送自己的tensor
            2. 求和
            3. 将结果发送回所有进程
            """
            if dist.is_initialized():
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
            return tensor

        # DDP自动处理梯度同步
        # 在backward()时自动调用all-reduce

        # 手动梯度同步（理解用）
        def manual_gradient_sync(model):
            """手动同步梯度"""
            world_size = dist.get_world_size()

            for param in model.parameters():
                if param.grad is not None:
                    # All-reduce梯度
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    # 平均
                    param.grad.data /= world_size
```

#### 1.1.2 模型并行（Model Parallelism）

```python
class ModelParallelism:
    """模型并行"""

    def pipeline_parallelism(self):
        """
        流水线并行（Pipeline Parallelism）

        将模型分割到多个GPU
        适用于大模型无法放入单个GPU

        示例：4层网络分到2个GPU
        GPU0: Layer1, Layer2
        GPU1: Layer3, Layer4
        """

        class PipelineParallelModel(nn.Module):
            def __init__(self):
                super().__init__()
                # 前两层在GPU 0
                self.layer1 = nn.Linear(1000, 1000).to('cuda:0')
                self.layer2 = nn.Linear(1000, 1000).to('cuda:0')

                # 后两层在GPU 1
                self.layer3 = nn.Linear(1000, 1000).to('cuda:1')
                self.layer4 = nn.Linear(1000, 10).to('cuda:1')

            def forward(self, x):
                # GPU 0
                x = x.to('cuda:0')
                x = torch.relu(self.layer1(x))
                x = torch.relu(self.layer2(x))

                # 传输到GPU 1
                x = x.to('cuda:1')
                x = torch.relu(self.layer3(x))
                x = self.layer4(x)

                return x

        # 问题：GPU空闲时间（气泡）
        # 解决：微批次流水线

    def microbatch_pipelining(self):
        """
        微批次流水线

        将大batch分成多个micro-batch
        流水线执行，减少气泡
        """

        class GPipePipeline:
            """
            GPipe风格的流水线并行

            将batch分成chunks
            前向传播所有chunks
            反向传播所有chunks
            """
            def __init__(self, model_stages, num_chunks=4):
                self.stages = model_stages
                self.num_chunks = num_chunks

            def forward_backward(self, inputs, targets):
                chunk_size = inputs.shape[0] // self.num_chunks
                chunks = inputs.split(chunk_size)
                target_chunks = targets.split(chunk_size)

                # 前向传播所有chunks
                outputs = []
                intermediates = []

                for chunk in chunks:
                    x = chunk
                    stage_outputs = []

                    for stage in self.stages:
                        x = stage(x)
                        stage_outputs.append(x)

                    outputs.append(x)
                    intermediates.append(stage_outputs)

                # 计算损失
                losses = []
                for output, target in zip(outputs, target_chunks):
                    loss = criterion(output, target)
                    losses.append(loss)

                # 反向传播
                for loss in losses:
                    loss.backward()

                return sum(losses) / len(losses)

    def tensor_parallelism(self):
        """
        张量并行（Tensor Parallelism）

        将单个层的参数分割到多个GPU
        适用于超大层（如Transformer的FFN）

        例：将线性层分割为列并行
        Y = XW = X[W₁ W₂] = [XW₁ XW₂]
        """

        class ColumnParallelLinear(nn.Module):
            """
            列并行线性层

            输出维度被分割
            """
            def __init__(self, in_features, out_features, world_size):
                super().__init__()
                self.world_size = world_size
                self.rank = dist.get_rank()

                # 每个GPU持有部分列
                self.out_features_per_partition = out_features // world_size

                self.weight = nn.Parameter(
                    torch.randn(self.out_features_per_partition, in_features)
                )
                self.bias = nn.Parameter(
                    torch.zeros(self.out_features_per_partition)
                )

            def forward(self, x):
                # 本地计算
                output = torch.matmul(x, self.weight.t()) + self.bias
                return output

        class RowParallelLinear(nn.Module):
            """
            行并行线性层

            输入维度被分割
            需要All-Reduce输出
            """
            def __init__(self, in_features, out_features, world_size):
                super().__init__()
                self.world_size = world_size
                self.in_features_per_partition = in_features // world_size

                self.weight = nn.Parameter(
                    torch.randn(out_features, self.in_features_per_partition)
                )

            def forward(self, x):
                # 本地计算
                output = torch.matmul(x, self.weight.t())

                # All-Reduce合并结果
                dist.all_reduce(output, op=dist.ReduceOp.SUM)

                return output
```

### 1.2 通信优化

```python
class CommunicationOptimization:
    """通信优化技术"""

    def gradient_compression(self):
        """
        梯度压缩

        减少通信量
        """

        # 1. Top-K梯度
        class TopKCompression:
            def __init__(self, k_ratio=0.01):
                self.k_ratio = k_ratio

            def compress(self, tensor):
                """只传输top-k个梯度"""
                numel = tensor.numel()
                k = max(1, int(numel * self.k_ratio))

                # 选择top-k
                values, indices = torch.topk(tensor.abs().view(-1), k)
                values = tensor.view(-1)[indices]

                return values, indices

            def decompress(self, values, indices, shape):
                """重建稀疏梯度"""
                tensor = torch.zeros(shape).view(-1)
                tensor[indices] = values
                return tensor.view(shape)

        # 2. 量化
        class GradientQuantization:
            def __init__(self, num_bits=8):
                self.num_bits = num_bits
                self.num_levels = 2 ** num_bits

            def quantize(self, tensor):
                """量化为int8"""
                # 计算缩放因子
                min_val = tensor.min()
                max_val = tensor.max()
                scale = (max_val - min_val) / (self.num_levels - 1)

                # 量化
                quantized = ((tensor - min_val) / scale).round().to(torch.int8)

                return quantized, scale, min_val

            def dequantize(self, quantized, scale, min_val):
                """反量化"""
                return quantized.float() * scale + min_val

        # 3. 随机化舍入
        def stochastic_rounding(tensor, num_bits=16):
            """
            随机化舍入保持期望

            E[round(x)] = x
            """
            scale = 2 ** (num_bits - 1)
            scaled = tensor * scale

            # 随机舍入
            floor = scaled.floor()
            frac = scaled - floor
            random_add = (torch.rand_like(frac) < frac).float()

            rounded = floor + random_add
            return rounded / scale

    def overlapping_computation_communication(self):
        """
        计算与通信重叠

        在计算后向传播时同步前面层的梯度
        """

        # DDP自动实现
        # 通过register_backward_hook实现

        # 手动实现概念
        class OverlappedDDP(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

                # 注册hooks
                for param in model.parameters():
                    if param.requires_grad:
                        param.register_hook(self._grad_hook)

                self.grad_queue = []

            def _grad_hook(self, grad):
                """
                梯度计算完成时调用

                立即开始通信，不等待所有梯度
                """
                # 异步All-Reduce
                handle = dist.all_reduce(
                    grad, op=dist.ReduceOp.SUM, async_op=True
                )
                self.grad_queue.append(handle)

                return grad

            def synchronize_gradients(self):
                """等待所有通信完成"""
                for handle in self.grad_queue:
                    handle.wait()
                self.grad_queue.clear()

    def hierarchical_allreduce(self):
        """
        层次化All-Reduce

        多机训练时：
        1. 机内All-Reduce（NVLink，快）
        2. 机间All-Reduce（网络，慢）
        """

        def hierarchical_all_reduce(tensor):
            """
            两层All-Reduce

            假设：
            - node_group: 同一节点的进程组
            - inter_node_group: 跨节点的进程组
            """
            # 1. 节点内All-Reduce
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=node_group)

            # 2. 每个节点选一个代表
            if is_node_master:
                # 3. 跨节点All-Reduce
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=inter_node_group)

            # 4. 节点内广播
            dist.broadcast(tensor, src=node_master_rank, group=node_group)

            return tensor
```

---

## 第二部分：内存优化

### 2.1 激活值检查点

```python
class ActivationCheckpointing:
    """激活值检查点（梯度检查点）"""

    def basic_checkpointing(self):
        """
        基础检查点技术

        思想：
        - 前向：不保存中间激活值
        - 反向：重新计算中间激活值

        权衡：
        - 内存：O(√n) 而非 O(n)
        - 时间：增加~30%
        """

        from torch.utils.checkpoint import checkpoint

        class CheckpointedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(1000, 1000) for _ in range(100)
                ])

            def forward(self, x):
                # 每10层使用一个checkpoint
                for i, layer in enumerate(self.layers):
                    if i % 10 == 0:
                        x = checkpoint(self._forward_segment, x, i)
                    else:
                        x = torch.relu(layer(x))
                return x

            def _forward_segment(self, x, start_idx):
                """checkpoint包装的前向片段"""
                end_idx = min(start_idx + 10, len(self.layers))
                for i in range(start_idx, end_idx):
                    x = torch.relu(self.layers[i](x))
                return x

    def selective_checkpointing(self):
        """
        选择性检查点

        只对内存占用大的层使用检查点
        """

        def should_checkpoint(layer):
            """决定是否对某层使用检查点"""
            # 策略1：大层使用检查点
            num_params = sum(p.numel() for p in layer.parameters())
            return num_params > 1_000_000

            # 策略2：靠后的层使用检查点
            # 因为前面的层在反向传播时已经释放

        class SmartCheckpointModel(nn.Module):
            def forward(self, x):
                for layer in self.layers:
                    if should_checkpoint(layer):
                        x = checkpoint(layer, x)
                    else:
                        x = layer(x)
                return x

    def checkpoint_with_rng_state(self):
        """
        保存随机数状态的检查点

        重计算时需要相同的dropout等随机操作
        """

        def checkpoint_with_rng(function, *args):
            """保存RNG状态的检查点"""
            # 保存RNG状态
            cpu_rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

            # 前向传播
            with torch.no_grad():
                outputs = function(*args)

            # 自定义反向传播
            def backward_hook(grad):
                # 恢复RNG状态
                torch.set_rng_state(cpu_rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state(cuda_rng_state)

                # 重新计算
                with torch.enable_grad():
                    outputs = function(*args)
                    outputs.backward(grad)

            outputs.register_hook(backward_hook)
            return outputs
```

### 2.2 内存池与分配策略

```python
class MemoryManagement:
    """内存管理技术"""

    def memory_profiling(self):
        """
        内存profiling

        分析内存使用
        """

        # 1. 基础内存查询
        def print_memory_usage():
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"已分配: {allocated:.2f} GB")
                print(f"已保留: {reserved:.2f} GB")

        # 2. 内存快照
        def memory_snapshot():
            """记录当前内存状态"""
            if torch.cuda.is_available():
                snapshot = torch.cuda.memory_snapshot()
                # 分析快照...
                return snapshot

        # 3. 使用torch.profiler
        from torch.profiler import profile, ProfilerActivity

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True
        ) as prof:
            # 训练代码
            model(inputs)

        # 打印内存使用
        print(prof.key_averages().table(
            sort_by="cuda_memory_usage", row_limit=10
        ))

    def memory_efficient_operations(self):
        """
        内存高效操作
        """

        # 1. 原地操作
        def inplace_operations():
            x = torch.randn(1000, 1000)

            # 避免：创建新张量
            y = x + 1

            # 推荐：原地修改
            x.add_(1)

        # 2. 及时删除不需要的张量
        def delete_unused_tensors():
            x = torch.randn(1000, 1000)
            y = expensive_computation(x)

            # 不再需要x
            del x
            torch.cuda.empty_cache()  # 释放缓存

        # 3. 使用更小的数据类型
        def use_smaller_dtypes():
            # FP32 -> FP16/BF16
            model = model.half()

            # INT8量化
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )

        # 4. 梯度累积
        def gradient_accumulation(accumulation_steps=4):
            """
            模拟大batch，实际使用小batch
            """
            optimizer.zero_grad()

            for i, (inputs, targets) in enumerate(dataloader):
                outputs = model(inputs)
                loss = criterion(outputs, targets) / accumulation_steps

                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

    def zero_redundancy_optimizer(self):
        """
        零冗余优化器（ZeRO）

        DeepSpeed的核心技术
        分割优化器状态、梯度、参数
        """

        # 概念实现
        class ZeROOptimizer:
            """
            ZeRO-1: 分割优化器状态
            ZeRO-2: 分割梯度
            ZeRO-3: 分割参数
            """

            def __init__(self, model, optimizer_class, **kwargs):
                self.model = model
                self.world_size = dist.get_world_size()
                self.rank = dist.get_rank()

                # 分割参数到不同rank
                self.partition_parameters()

                # 每个rank只为自己的参数创建优化器
                self.optimizer = optimizer_class(
                    self.local_params, **kwargs
                )

            def partition_parameters(self):
                """将参数分割到不同GPU"""
                params = list(self.model.parameters())
                num_params = len(params)
                params_per_rank = (num_params + self.world_size - 1) // self.world_size

                start_idx = self.rank * params_per_rank
                end_idx = min((self.rank + 1) * params_per_rank, num_params)

                self.local_params = params[start_idx:end_idx]

            def step(self):
                """更新参数"""
                # 1. All-Gather梯度
                for param in self.model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad)

                # 2. 本地优化器更新本地参数
                self.optimizer.step()

                # 3. All-Gather更新后的参数
                for param in self.local_params:
                    dist.all_gather(param.data)

        # 实际使用DeepSpeed
        """
        import deepspeed

        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config={
                "train_batch_size": 32,
                "zero_optimization": {
                    "stage": 3,  # ZeRO-3
                    "offload_optimizer": {
                        "device": "cpu"  # 卸载到CPU
                    }
                }
            }
        )
        """
```

---

## 第三部分：性能优化

### 3.1 算子优化

```python
class OperatorOptimization:
    """算子级优化"""

    def fused_operations(self):
        """
        算子融合

        将多个小算子合并为一个大算子
        减少内存访问
        """

        # 1. Fused Adam
        # 融合参数更新的多个操作

        # 标准Adam（多次内存访问）
        def standard_adam_step(param, grad, m, v, lr, beta1, beta2, eps):
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            param = param - lr * m / (torch.sqrt(v) + eps)
            return param, m, v

        # Fused Adam（单次内存访问）
        # 使用apex或自定义CUDA kernel

        # 2. Fused LayerNorm + Linear
        class FusedLayerNormLinear(nn.Module):
            """融合LayerNorm和Linear"""
            def __init__(self, normalized_shape, out_features):
                super().__init__()
                self.ln = nn.LayerNorm(normalized_shape)
                self.linear = nn.Linear(normalized_shape, out_features)

            def forward(self, x):
                # 融合实现会更快
                # 需要自定义CUDA kernel
                x = self.ln(x)
                x = self.linear(x)
                return x

    def custom_cuda_kernels(self):
        """
        自定义CUDA kernels

        极致性能优化
        """

        # 使用PyTorch C++/CUDA扩展
        from torch.utils.cpp_extension import load

        # 加载自定义CUDA kernel
        """
        custom_ops = load(
            name="custom_ops",
            sources=["custom_ops.cpp", "custom_ops_cuda.cu"],
            verbose=True
        )
        """

        # CUDA kernel示例（概念）
        """
        // custom_ops_cuda.cu
        __global__ void fused_add_relu_kernel(
            float* output,
            const float* input,
            const float* bias,
            int n
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                float val = input[idx] + bias[idx];
                output[idx] = val > 0 ? val : 0;  // ReLU
            }
        }
        """

    def torch_compile(self):
        """
        torch.compile (PyTorch 2.0+)

        自动算子融合与优化
        """

        # 简单使用
        model = MyModel()
        compiled_model = torch.compile(model)

        # 高级配置
        compiled_model = torch.compile(
            model,
            mode="reduce-overhead",  # 或 "default", "max-autotune"
            fullgraph=True,          # 尝试编译整个图
            dynamic=False            # 静态形状假设
        )

        # mode选项：
        # - "default": 平衡性能和编译时间
        # - "reduce-overhead": 减少Python开销
        # - "max-autotune": 最大化性能（编译慢）

    def torch_jit(self):
        """
        TorchScript JIT

        将模型编译为静态图
        """

        # 1. Tracing
        model = MyModel()
        example_input = torch.randn(1, 3, 224, 224)

        traced_model = torch.jit.trace(model, example_input)

        # 保存
        traced_model.save("model_traced.pt")

        # 2. Scripting
        scripted_model = torch.jit.script(model)

        # Scripting支持控制流
        @torch.jit.script
        def scripted_function(x):
            if x.sum() > 0:
                return x * 2
            else:
                return x + 1

        # 3. 混合使用
        class HybridModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.jit.trace(
                    ResNet(), torch.randn(1, 3, 224, 224)
                )
                self.head = nn.Linear(1000, 10)

            def forward(self, x):
                x = self.backbone(x)
                x = self.head(x)
                return x
```

### 3.2 数据加载优化

```python
class DataLoadingOptimization:
    """数据加载优化"""

    def efficient_dataloader(self):
        """
        高效DataLoader配置
        """

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,

            # 关键参数
            num_workers=4,          # 多进程加载
            pin_memory=True,        # 锁页内存，加速传输
            prefetch_factor=2,      # 每个worker预取batch数
            persistent_workers=True # 保持worker进程（PyTorch 1.7+）
        )

        # num_workers调优：
        # - CPU密集：num_workers = CPU核数
        # - I/O密集：实验确定最佳值
        # - 过多会增加内存和开销

    def custom_collate_fn(self):
        """
        自定义collate函数

        批次数据的组装方式
        """

        def custom_collate(batch):
            """
            batch: list of (sample, target)
            """
            # 分离样本和目标
            samples, targets = zip(*batch)

            # 动态padding
            max_len = max(s.shape[0] for s in samples)
            padded_samples = []

            for s in samples:
                pad_len = max_len - s.shape[0]
                padded = torch.nn.functional.pad(s, (0, pad_len))
                padded_samples.append(padded)

            # 堆叠
            samples = torch.stack(padded_samples)
            targets = torch.tensor(targets)

            return samples, targets

        dataloader = DataLoader(
            dataset,
            batch_size=32,
            collate_fn=custom_collate
        )

    def data_prefetching(self):
        """
        数据预取

        GPU计算时，CPU预先加载下一批数据
        """

        class DataPrefetcher:
            """异步数据预取器"""
            def __init__(self, loader):
                self.loader = iter(loader)
                self.stream = torch.cuda.Stream()
                self.preload()

            def preload(self):
                try:
                    self.next_input, self.next_target = next(self.loader)
                except StopIteration:
                    self.next_input = None
                    self.next_target = None
                    return

                # 异步传输到GPU
                with torch.cuda.stream(self.stream):
                    self.next_input = self.next_input.cuda(non_blocking=True)
                    self.next_target = self.next_target.cuda(non_blocking=True)

            def __iter__(self):
                return self

            def __next__(self):
                torch.cuda.current_stream().wait_stream(self.stream)

                input = self.next_input
                target = self.next_target

                if input is None:
                    raise StopIteration

                input.record_stream(torch.cuda.current_stream())
                target.record_stream(torch.cuda.current_stream())

                self.preload()
                return input, target

        # 使用
        prefetcher = DataPrefetcher(dataloader)
        for inputs, targets in prefetcher:
            # 训练...
            pass

    def caching_and_mmap(self):
        """
        缓存与内存映射

        减少重复I/O
        """

        # 1. 内存映射数据集
        class MemmapDataset(torch.utils.data.Dataset):
            """使用mmap的大型数据集"""
            def __init__(self, data_file, shape, dtype=np.float32):
                self.data = np.memmap(
                    data_file,
                    dtype=dtype,
                    mode='r',
                    shape=shape
                )

            def __len__(self):
                return self.data.shape[0]

            def __getitem__(self, idx):
                return torch.from_numpy(self.data[idx].copy())

        # 2. 缓存预处理结果
        class CachedDataset(torch.utils.data.Dataset):
            """缓存数据集"""
            def __init__(self, base_dataset, cache_dir):
                self.base_dataset = base_dataset
                self.cache_dir = cache_dir
                os.makedirs(cache_dir, exist_ok=True)

            def __getitem__(self, idx):
                cache_path = os.path.join(self.cache_dir, f"{idx}.pt")

                # 检查缓存
                if os.path.exists(cache_path):
                    return torch.load(cache_path)

                # 计算并缓存
                data = self.base_dataset[idx]
                torch.save(data, cache_path)
                return data

            def __len__(self):
                return len(self.base_dataset)
```

### 3.3 推理优化

```python
class InferenceOptimization:
    """推理优化"""

    def model_quantization(self):
        """
        模型量化

        减少模型大小，加速推理
        """

        # 1. 动态量化（最简单）
        import torch.quantization

        model_fp32 = MyModel()

        # 量化Linear和LSTM层
        model_int8 = torch.quantization.quantize_dynamic(
            model_fp32,
            {nn.Linear, nn.LSTM},
            dtype=torch.qint8
        )

        # 2. 静态量化（更激进）
        # 需要校准数据

        # 准备：插入observer
        model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        model_prepared = torch.quantization.prepare(model_fp32)

        # 校准：运行代表性数据
        with torch.no_grad():
            for inputs in calibration_data:
                model_prepared(inputs)

        # 转换为量化模型
        model_int8 = torch.quantization.convert(model_prepared)

        # 3. 量化感知训练（QAT）
        # 训练时模拟量化

        model_fp32.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        model_prepared = torch.quantization.prepare_qat(model_fp32)

        # 训练
        for epoch in range(epochs):
            train(model_prepared)

        # 转换
        model_int8 = torch.quantization.convert(model_prepared)

    def model_pruning(self):
        """
        模型剪枝

        移除不重要的权重
        """

        import torch.nn.utils.prune as prune

        # 1. 非结构化剪枝
        model = MyModel()

        # 剪枝单个层
        prune.l1_unstructured(
            model.conv1,
            name='weight',
            amount=0.3  # 剪枝30%
        )

        # 剪枝多个层
        parameters_to_prune = [
            (model.conv1, 'weight'),
            (model.conv2, 'weight'),
            (model.fc1, 'weight'),
        ]

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.2
        )

        # 2. 结构化剪枝
        # 剪枝整个通道或神经元

        prune.ln_structured(
            model.conv1,
            name='weight',
            amount=0.5,
            n=2,  # L2范数
            dim=0  # 剪枝输出通道
        )

        # 3. 使剪枝永久化
        # 移除重参数化
        for module, name in parameters_to_prune:
            prune.remove(module, name)

    def knowledge_distillation(self):
        """
        知识蒸馏

        用大模型（教师）训练小模型（学生）
        """

        class DistillationLoss(nn.Module):
            def __init__(self, temperature=3.0, alpha=0.5):
                super().__init__()
                self.temperature = temperature
                self.alpha = alpha
                self.kl_div = nn.KLDivLoss(reduction='batchmean')

            def forward(self, student_logits, teacher_logits, targets):
                """
                蒸馏损失 = α * 软目标损失 + (1-α) * 硬目标损失
                """
                # 软目标损失（KL散度）
                student_soft = torch.log_softmax(
                    student_logits / self.temperature, dim=1
                )
                teacher_soft = torch.softmax(
                    teacher_logits / self.temperature, dim=1
                )

                soft_loss = self.kl_div(student_soft, teacher_soft) * \
                           (self.temperature ** 2)

                # 硬目标损失（交叉熵）
                hard_loss = nn.functional.cross_entropy(
                    student_logits, targets
                )

                # 组合
                return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        # 训练
        teacher_model.eval()
        student_model.train()

        distillation_loss = DistillationLoss(temperature=3.0, alpha=0.7)

        for inputs, targets in dataloader:
            # 教师预测
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)

            # 学生预测
            student_logits = student_model(inputs)

            # 蒸馏损失
            loss = distillation_loss(student_logits, teacher_logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def onnx_export(self):
        """
        导出为ONNX

        跨平台部署
        """

        model = MyModel()
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)

        # 导出
        torch.onnx.export(
            model,
            dummy_input,
            "model.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        # 使用ONNX Runtime推理
        """
        import onnxruntime as ort

        session = ort.InferenceSession("model.onnx")
        outputs = session.run(
            None,
            {"input": input_data.numpy()}
        )
        """
```

---

## 第四部分：大规模训练实战

### 4.1 完整训练流程

```python
class LargeScaleTrainingPipeline:
    """大规模训练完整流程"""

    def distributed_training_template(self):
        """
        分布式训练模板

        集成所有最佳实践
        """

        def setup(rank, world_size):
            """初始化分布式环境"""
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'

            dist.init_process_group(
                backend='nccl',
                rank=rank,
                world_size=world_size
            )

            torch.cuda.set_device(rank)

        def cleanup():
            dist.destroy_process_group()

        def train(rank, world_size, config):
            setup(rank, world_size)

            # 1. 创建模型
            model = create_model(config).to(rank)
            model = DDP(model, device_ids=[rank])

            # 2. 优化器
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config['lr'],
                weight_decay=config['weight_decay']
            )

            # 3. 学习率调度器
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config['lr'],
                steps_per_epoch=len(train_loader),
                epochs=config['epochs']
            )

            # 4. 混合精度
            scaler = torch.cuda.amp.GradScaler()

            # 5. 数据加载
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=config['batch_size'],
                sampler=train_sampler,
                num_workers=4,
                pin_memory=True
            )

            # 6. 训练循环
            for epoch in range(config['epochs']):
                train_sampler.set_epoch(epoch)
                model.train()

                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs = inputs.to(rank, non_blocking=True)
                    targets = targets.to(rank, non_blocking=True)

                    # 混合精度前向传播
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                    # 反向传播
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()

                    # 梯度裁剪
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0
                    )

                    # 更新
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    # 日志（仅rank 0）
                    if rank == 0 and batch_idx % 100 == 0:
                        print(f"Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")

                # 验证
                if epoch % config['val_frequency'] == 0:
                    val_loss = validate(model, val_loader, rank)
                    if rank == 0:
                        print(f"Validation Loss: {val_loss}")

                        # 保存检查点
                        save_checkpoint(model, optimizer, epoch, val_loss)

            cleanup()

        def validate(model, dataloader, rank):
            """验证函数"""
            model.eval()
            total_loss = 0

            with torch.no_grad():
                for inputs, targets in dataloader:
                    inputs = inputs.to(rank)
                    targets = targets.to(rank)

                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                    total_loss += loss.item()

            # All-reduce损失
            avg_loss = total_loss / len(dataloader)
            loss_tensor = torch.tensor(avg_loss, device=rank)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

            return loss_tensor.item()

        def save_checkpoint(model, optimizer, epoch, val_loss):
            """保存检查点"""
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }

            torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')

        # 启动
        world_size = torch.cuda.device_count()
        mp.spawn(train, args=(world_size, config), nprocs=world_size, join=True)
```

---

## 总结

本教程涵盖了PyTorch大规模训练的全部核心内容：

### 分布式训练
- 数据并行 vs 模型并行
- DDP实现与优化
- 通信优化技术

### 内存优化
- 激活值检查点
- ZeRO优化器
- 内存profiling

### 性能优化
- 算子融合
- 自定义CUDA kernels
- torch.compile

### 推理优化
- 模型量化
- 模型剪枝
- 知识蒸馏

### 实战经验
- 完整训练流程
- 最佳实践集成

---

## PyTorch十年经验总结

### 核心原则
1. **理解数学原理**：知其然知其所以然
2. **性能优先**：内存、计算、通信的权衡
3. **工程实践**：可维护、可扩展的代码
4. **持续学习**：关注最新研究与工具

### 学习路线图
1. 数学基础（线代、微积分、概率论）
2. PyTorch核心（张量、autograd、nn）
3. 模型设计（架构、组件、tricks）
4. 训练优化（优化器、学习率、正则化）
5. 大规模训练（分布式、内存、性能）

### 推荐资源
- 官方文档：pytorch.org
- 源码阅读：github.com/pytorch/pytorch
- 论文实现：paperswithcode.com
- 社区讨论：discuss.pytorch.org

---

## 结语

PyTorch的学习是一个持续深入的过程。这五篇教程涵盖了从数学原理到工程实践的方方面面，但真正的掌握需要大量的实践和思考。

记住：**理论指导实践，实践验证理论。**

祝你在PyTorch的学习道路上越走越远！
