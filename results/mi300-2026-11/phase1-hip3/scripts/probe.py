import torch
p = torch.cuda.get_device_properties(0)
print("device_count", torch.cuda.device_count(), p.name, p.gcnArchName, "CUs", p.multi_processor_count)
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx942
print("is_rocm", current_platform.is_rocm(), "on_gfx942", on_gfx942())
