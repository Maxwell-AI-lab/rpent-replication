from pathlib import Path
p = Path("/data/rpent/rpent/rpent/robots/components/sam3_server.py")
t = p.read_text()
orig = t

# 1. module-level NPU block: respect SAM3_DEVICE=cpu
t = t.replace(
    "    if not torch.cuda.is_available():\n        from torch_npu.contrib import transfer_to_npu  # noqa: F401",
    "    if not torch.cuda.is_available() and os.environ.get(\"SAM3_DEVICE\", \"\").lower() != \"cpu\":\n        from torch_npu.contrib import transfer_to_npu  # noqa: F401",
)

# 2. gate + tf32 + set_device
t = t.replace(
    """        if not torch.cuda.is_available():
            raise RuntimeError("local SAM3 requires a CUDA-capable GPU")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_device(0)
""",
    """        _cpu_mode = os.environ.get("SAM3_DEVICE", "").lower() == "cpu"
        _dev = "cpu" if _cpu_mode else "cuda"
        if not _cpu_mode:
            if not torch.cuda.is_available():
                raise RuntimeError("local SAM3 requires a CUDA-capable GPU")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.cuda.set_device(0)
""",
)

# 3. build model device + processor + self._device
t = t.replace('device="cuda",\n                checkpoint_path=checkpoint_path,',
              'device=_dev,\n                checkpoint_path=checkpoint_path,')
t = t.replace('self._processor = Sam3Processor(model, device="cuda", confidence_threshold=0.0)',
              'self._processor = Sam3Processor(model, device=_dev, confidence_threshold=0.0)')
t = t.replace('self._device = "cuda"', 'self._device = _dev')

assert t != orig, "no replacement applied"
p.write_text(t)
import ast; ast.parse(t)
print("patched + syntax ok")
