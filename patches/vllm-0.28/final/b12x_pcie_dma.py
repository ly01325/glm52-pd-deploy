import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_SUPPORTED_WORLD_SIZES = (2, 4, 8)


class B12xPcieDmaAllReduce:
    def __init__(self, group: ProcessGroup, device: torch.device):
        self.disabled = True
        self.group = group
        self.device = device
        self.world_size = dist.get_world_size(group)
        self._runtime = None
        self._logged_shapes: set[tuple[tuple[int, ...], torch.dtype]] = set()

        if self.world_size not in _SUPPORTED_WORLD_SIZES or device.type != "cuda":
            return

        try:
            from b12x.comm.pcie import DmaAllReduce

            self._runtime = DmaAllReduce(
                exchange_group=group,
                device=device,
                max_bytes=envs.VLLM_B12X_DMA_MAX_BYTES,
                fp8=envs.VLLM_B12X_DMA_FP8 or None,
            )
            self._runtime.min_bytes = envs.VLLM_B12X_DMA_MIN_BYTES
        except Exception:
            logger.exception("B12x PCIe DMA all-reduce initialization failed")
            self.close()
            return

        self.disabled = False
        logger.info(
            "B12x PCIe DMA all-reduce enabled (world=%d, wire=%s, bytes=%d..%d)",
            self.world_size,
            self._runtime.wire_mode,
            self._runtime.min_bytes,
            self._runtime.max_bytes,
        )

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor | None:
        runtime = self._runtime
        if self.disabled or runtime is None or not runtime.should_allreduce(inp):
            return None

        try:
            out = runtime.all_reduce(inp)
            key = (tuple(inp.shape), inp.dtype)
            if key not in self._logged_shapes:
                self._logged_shapes.add(key)
                logger.info(
                    "B12x PCIe DMA dispatch: shape=%s dtype=%s bytes=%d",
                    tuple(inp.shape),
                    inp.dtype,
                    inp.numel() * inp.element_size(),
                )
            return out
        except Exception:
            logger.exception("B12x PCIe DMA all-reduce failed; disabling backend")
            self.disabled = True
            return None

    def close(self):
        runtime = self._runtime
        self._runtime = None
        if runtime is not None:
            runtime.close()
        self._logged_shapes.clear()
