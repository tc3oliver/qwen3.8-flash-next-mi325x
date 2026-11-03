import ctypes, subprocess, sys, os, torch, collections
subprocess.run(["hipcc", "-O2", "-fPIC", "-shared", "--offload-arch=gfx942", "-o", "/tmp/arrive.so", "/s/arrive.hip"], check=True)
lib = ctypes.CDLL("/tmp/arrive.so")
rows, n = 8, 1 << 20
x = torch.randn(n, device="cuda"); rank = torch.zeros(rows * 8, dtype=torch.int32, device="cuda"); xcc = torch.zeros(rows, dtype=torch.int32, device="cuda")
pat = collections.Counter(); xs = collections.Counter()
for _ in range(300):
    lib.launch(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(rank.data_ptr()), ctypes.c_void_p(xcc.data_ptr()), n, rows)
    torch.accelerator.synchronize()
    r = rank.view(rows, 8)[2].tolist(); pat[tuple(sorted(range(8), key=lambda w: r[w]))] += 1
    xs[tuple(xcc.tolist())] += 1
print(f"ARRIVE pid={os.getpid()} block2 wave-arrival orders: {pat.most_common(3)} distinct={len(pat)}")
print(f"ARRIVE pid={os.getpid()} xcc-id-per-block: {xs.most_common(3)} distinct={len(xs)}")
