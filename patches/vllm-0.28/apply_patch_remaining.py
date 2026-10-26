from pathlib import Path
r=Path("/usr/local/lib/python3.12/dist-packages/vllm")
def R(p,a,z):
 s=p.read_text(); n=s.count(a)
 if n!=1: raise RuntimeError(f"{p}: match={n}: {a[:100]!r}")
 p.write_text(s.replace(a,z,1))
p=r/"distributed/device_communicators/cuda_communicator.py"
a="""        if self.fi_ar_comm is not None:
            self.fi_ar_comm.destroy()
            self.fi_ar_comm = None
        if self.all2all_manager is not None:
"""
z="""        if self.fi_ar_comm is not None:
            self.fi_ar_comm.destroy()
            self.fi_ar_comm = None
        if self.fi_pcie_ipc_ar_comm is not None:
            self.fi_pcie_ipc_ar_comm.destroy()
            self.fi_pcie_ipc_ar_comm = None
        if self.b12x_pcie_dma_ar_comm is not None:
            self.b12x_pcie_dma_ar_comm.close()
            self.b12x_pcie_dma_ar_comm = None
        if self.all2all_manager is not None:
"""
R(p,a,z)
p=r/"distributed/parallel_state.py"
R(p,"        maybe_ca_context = nullcontext()\n        maybe_aiter_context = nullcontext()\n","        maybe_ca_context = nullcontext()\n        maybe_fi_pcie_ipc_context = nullcontext()\n        maybe_aiter_context = nullcontext()\n")
R(p,"            if ca_comm is not None:\n                maybe_ca_context = ca_comm.capture()  # type: ignore\n\n            from vllm._aiter_ops import rocm_aiter_ops\n","            if ca_comm is not None:\n                maybe_ca_context = ca_comm.capture()  # type: ignore\n            if isinstance(self.device_communicator, CudaCommunicator):\n                fi_pcie_ipc_ar_comm = self.device_communicator.fi_pcie_ipc_ar_comm\n                if fi_pcie_ipc_ar_comm is not None:\n                    maybe_fi_pcie_ipc_context = fi_pcie_ipc_ar_comm.capture()\n\n            from vllm._aiter_ops import rocm_aiter_ops\n")
R(p,"        with torch.cuda.stream(stream), maybe_ca_context, maybe_aiter_context:\n            yield graph_capture_context\n","        with (torch.cuda.stream(stream), maybe_ca_context, maybe_fi_pcie_ipc_context, maybe_aiter_context):\n            yield graph_capture_context\n")
p=r/"model_executor/warmup/kernel_warmup.py"
R(p,"    minimax_m3_msa_warmup(worker)\n\n    enable_flashinfer_autotune = (\n","    minimax_m3_msa_warmup(worker)\n\n    if envs.VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC:\n        from vllm.distributed.device_communicators import flashinfer_pcie_ipc_all_reduce\n        flashinfer_pcie_ipc_all_reduce.warmup_flashinfer_pcie_ipc_allreduce(worker)\n\n    enable_flashinfer_autotune = (\n")
print("remaining patch complete")
