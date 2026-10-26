from pathlib import Path
import shutil

root = Path("/usr/local/lib/python3.12/dist-packages/vllm")
backup = Path("/tmp/glm52-vllm-original")
backup.mkdir(parents=True, exist_ok=True)

files = {
    "envs.py": root / "envs.py",
    "cuda_communicator.py": root / "distributed/device_communicators/cuda_communicator.py",
    "parallel_state.py": root / "distributed/parallel_state.py",
    "kernel_warmup.py": root / "model_executor/warmup/kernel_warmup.py",
}
for name, path in files.items():
    target = backup / name
    if not target.exists():
        shutil.copy2(path, target)


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, got {count}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1))


envs = files["envs.py"]
replace_once(
    envs,
    "    VLLM_ALLREDUCE_USE_FLASHINFER: bool = False\n",
    "    VLLM_ALLREDUCE_USE_FLASHINFER: bool = False\n"
    "    VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC: bool = False\n"
    "    VLLM_ALLREDUCE_USE_B12X_PCIE_DMA: bool = False\n"
    "    VLLM_B12X_DMA_MIN_BYTES: int = 8 * 1024 * 1024\n"
    "    VLLM_B12X_DMA_MAX_BYTES: int = 256 * 1024 * 1024\n"
    "    VLLM_B12X_DMA_FP8: str = \"ring\"\n",
)
replace_once(
    envs,
    '    "VLLM_ALLREDUCE_USE_FLASHINFER": lambda: bool(\n'
    '        int(os.getenv("VLLM_ALLREDUCE_USE_FLASHINFER", "0"))\n'
    "    ),\n",
    '    "VLLM_ALLREDUCE_USE_FLASHINFER": lambda: bool(\n'
    '        int(os.getenv("VLLM_ALLREDUCE_USE_FLASHINFER", "0"))\n'
    "    ),\n"
    '    "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC": lambda: bool(\n'
    '        int(os.getenv("VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC", "0"))\n'
    "    ),\n"
    '    "VLLM_ALLREDUCE_USE_B12X_PCIE_DMA": lambda: bool(\n'
    '        int(os.getenv("VLLM_ALLREDUCE_USE_B12X_PCIE_DMA", "0"))\n'
    "    ),\n"
    '    "VLLM_B12X_DMA_MIN_BYTES": lambda: int(\n'
    '        os.getenv("VLLM_B12X_DMA_MIN_BYTES", str(8 * 1024 * 1024))\n'
    "    ),\n"
    '    "VLLM_B12X_DMA_MAX_BYTES": lambda: int(\n'
    '        os.getenv("VLLM_B12X_DMA_MAX_BYTES", str(256 * 1024 * 1024))\n'
    "    ),\n"
    '    "VLLM_B12X_DMA_FP8": lambda: os.getenv("VLLM_B12X_DMA_FP8", "ring"),\n',
)

comm = files["cuda_communicator.py"]
replace_once(comm, '        if "tp" not in unique_name:\n', '        if unique_name.split(":")[0] != "tp":\n')
replace_once(
    comm,
    "            use_flashinfer_allreduce = False\n            use_aiter_allreduce = False\n",
    "            use_flashinfer_allreduce = False\n"
    "            use_flashinfer_pcie_ipc_allreduce = False\n"
    "            use_b12x_pcie_dma_allreduce = False\n"
    "            use_aiter_allreduce = False\n",
)
replace_once(
    comm,
    "            use_flashinfer_allreduce = envs.VLLM_ALLREDUCE_USE_FLASHINFER\n",
    "            use_flashinfer_allreduce = envs.VLLM_ALLREDUCE_USE_FLASHINFER\n"
    "            use_flashinfer_pcie_ipc_allreduce = (\n"
    "                envs.VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC\n"
    "                and not envs.VLLM_BATCH_INVARIANT\n"
    "            )\n"
    "            use_b12x_pcie_dma_allreduce = (\n"
    "                envs.VLLM_ALLREDUCE_USE_B12X_PCIE_DMA\n"
    "                and not envs.VLLM_BATCH_INVARIANT\n"
    "            )\n",
)
replace_once(
    comm,
    "        self.use_flashinfer_allreduce = use_flashinfer_allreduce\n",
    "        self.use_flashinfer_allreduce = use_flashinfer_allreduce\n"
    "        self.use_flashinfer_pcie_ipc_allreduce = use_flashinfer_pcie_ipc_allreduce\n"
    "        self.use_b12x_pcie_dma_allreduce = use_b12x_pcie_dma_allreduce\n",
)
replace_once(
    comm,
    "        # lazy import to avoid documentation build error\n",
    "        # lazy import to avoid documentation build error\n"
    "        from vllm.distributed.device_communicators.b12x_pcie_dma import (\n"
    "            B12xPcieDmaAllReduce,\n"
    "        )\n",
)
replace_once(
    comm,
    "        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator\n",
    "        from vllm.distributed.device_communicators.flashinfer_pcie_ipc_all_reduce import (\n"
    "            FlashInferPcieIpcAllReduce,\n"
    "        )\n"
    "        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator\n",
)
replace_once(
    comm,
    "        self.fi_ar_comm: FlashInferAllReduce | None = None\n",
    "        self.fi_ar_comm: FlashInferAllReduce | None = None\n"
    "        self.fi_pcie_ipc_ar_comm: FlashInferPcieIpcAllReduce | None = None\n"
    "        self.b12x_pcie_dma_ar_comm: B12xPcieDmaAllReduce | None = None\n",
)
replace_once(
    comm,
    "        if self.use_flashinfer_allreduce and self.world_size > 1:\n",
    "        if (\n"
    "            self.use_b12x_pcie_dma_allreduce\n"
    "            and self.world_size > 1\n"
    "            and self.device_group is not None\n"
    "        ):\n"
    "            self.b12x_pcie_dma_ar_comm = B12xPcieDmaAllReduce(\n"
    "                group=self.device_group, device=self.device\n"
    "            )\n\n"
    "        if self.use_flashinfer_allreduce and self.world_size > 1:\n",
)
replace_once(
    comm,
    "        if self.use_aiter_allreduce and self.world_size > 1:\n",
    "        if (\n"
    "            self.use_flashinfer_pcie_ipc_allreduce\n"
    "            and self.world_size > 1\n"
    "            and self.device_group is not None\n"
    "        ):\n"
    "            self.fi_pcie_ipc_ar_comm = FlashInferPcieIpcAllReduce(\n"
    "                group=self.device_group,\n"
    "                tune_group=self.cpu_group,\n"
    "                device=self.device,\n"
    "            )\n\n"
    "        if self.use_aiter_allreduce and self.world_size > 1:\n",
)
replace_once(
    comm,
    '        all_potential_ar_backends = [\n            "NCCL_SYMM_MEM",\n',
    '        all_potential_ar_backends = [\n'
    '            "B12X_PCIE_DMA",\n'
    '            "FLASHINFER_PCIE_IPC",\n'
    '            "NCCL_SYMM_MEM",\n',
)
replace_once(
    comm,
    "        enabled_ar_backends: list[str] = []\n",
    "        enabled_ar_backends: list[str] = []\n"
    "        if (\n"
    "            self.b12x_pcie_dma_ar_comm is not None\n"
    "            and not self.b12x_pcie_dma_ar_comm.disabled\n"
    "        ):\n"
    '            enabled_ar_backends.append("B12X_PCIE_DMA")\n'
    "        if (\n"
    "            self.fi_pcie_ipc_ar_comm is not None\n"
    "            and not self.fi_pcie_ipc_ar_comm.disabled\n"
    "        ):\n"
    '            enabled_ar_backends.append("FLASHINFER_PCIE_IPC")\n',
)
replace_once(
    comm,
    "        fi_ar_comm = self.fi_ar_comm\n",
    "        b12x_pcie_dma_ar_comm = self.b12x_pcie_dma_ar_comm\n"
    "        if b12x_pcie_dma_ar_comm is not None:\n"
    "            out = b12x_pcie_dma_ar_comm.all_reduce(input_)\n"
    "            if out is not None:\n"
    "                return out\n"
    "        fi_pcie_ipc_ar_comm = self.fi_pcie_ipc_ar_comm\n"
    "        if fi_pcie_ipc_ar_comm is not None and fi_pcie_ipc_ar_comm.should_use(input_):\n"
    "            return fi_pcie_ipc_ar_comm.all_reduce(input_)\n"
    "        fi_ar_comm = self.fi_ar_comm\n",
)
replace_once(
    comm,
    "        if self.all2all_manager is not None:\n",
    "        if self.fi_pcie_ipc_ar_comm is not None:\n"
    "            self.fi_pcie_ipc_ar_comm.destroy()\n"
    "            self.fi_pcie_ipc_ar_comm = None\n"
    "        if self.b12x_pcie_dma_ar_comm is not None:\n"
    "            self.b12x_pcie_dma_ar_comm.close()\n"
    "            self.b12x_pcie_dma_ar_comm = None\n"
    "        if self.all2all_manager is not None:\n",
)

parallel = files["parallel_state.py"]
replace_once(
    parallel,
    "        maybe_ca_context = nullcontext()\n        maybe_aiter_context = nullcontext()\n",
    "        maybe_ca_context = nullcontext()\n"
    "        maybe_fi_pcie_ipc_context = nullcontext()\n"
    "        maybe_aiter_context = nullcontext()\n",
)
replace_once(
    parallel,
    "            if ca_comm is not None:\n"
    "                maybe_ca_context = ca_comm.capture()  # type: ignore\n\n"
    "            from vllm._aiter_ops import rocm_aiter_ops\n",
    "            if ca_comm is not None:\n"
    "                maybe_ca_context = ca_comm.capture()  # type: ignore\n"
    "            if isinstance(self.device_communicator, CudaCommunicator):\n"
    "                fi_pcie_ipc_ar_comm = self.device_communicator.fi_pcie_ipc_ar_comm\n"
    "                if fi_pcie_ipc_ar_comm is not None:\n"
    "                    maybe_fi_pcie_ipc_context = fi_pcie_ipc_ar_comm.capture()\n\n"
    "            from vllm._aiter_ops import rocm_aiter_ops\n",
)
replace_once(
    parallel,
    "        with torch.cuda.stream(stream), maybe_ca_context, maybe_aiter_context:\n"
    "            yield graph_capture_context\n",
    "        with (\n"
    "            torch.cuda.stream(stream),\n"
    "            maybe_ca_context,\n"
    "            maybe_fi_pcie_ipc_context,\n"
    "            maybe_aiter_context,\n"
    "        ):\n"
    "            yield graph_capture_context\n",
)

warmup = files["kernel_warmup.py"]
replace_once(
    warmup,
    "    minimax_m3_msa_warmup(worker)\n\n"
    "    enable_flashinfer_autotune = (\n",
    "    minimax_m3_msa_warmup(worker)\n\n"
    "    if envs.VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC:\n"
    "        from vllm.distributed.device_communicators import (\n"
    "            flashinfer_pcie_ipc_all_reduce,\n"
    "        )\n\n"
    "        flashinfer_pcie_ipc_all_reduce.warmup_flashinfer_pcie_ipc_allreduce(worker)\n\n"
    "    enable_flashinfer_autotune = (\n",
)

print("patched", *(str(p) for p in files.values()), sep="\n")
