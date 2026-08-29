"""Measure ResNet-18 / CIFAR-10 training throughput on this CPU.

Times forward+backward+step on synthetic 32x32 batches, so the number
isolates compute from dataloading/disk. Extrapolates to epoch and
full-run cost for the ForgetCheck experiment matrix.
"""
import time
import torch
import torch.nn as nn
import torchvision

torch.set_num_threads(12)  # physical cores, not SMT threads

def make_resnet18_cifar():
    m = torchvision.models.resnet18(weights=None, num_classes=10)
    # standard CIFAR stem: 3x3 conv, no maxpool
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m

def bench(batch_size, n_iters=12, warmup=3):
    model = make_resnet18_cifar()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(batch_size, 3, 32, 32)
    y = torch.randint(0, 10, (batch_size,))

    model.train()
    for _ in range(warmup):
        opt.zero_grad(set_to_none=True)
        crit(model(x), y).backward()
        opt.step()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        opt.zero_grad(set_to_none=True)
        crit(model(x), y).backward()
        opt.step()
    dt = time.perf_counter() - t0
    return dt / n_iters

TRAIN_N = 50_000
print(f"threads={torch.get_num_threads()}  torch={torch.__version__}")
print(f"{'batch':>6} {'s/step':>9} {'img/s':>10} {'s/epoch':>9} {'min/epoch':>10}")
results = {}
for bs in (128, 256):
    s = bench(bs)
    ips = bs / s
    epoch_s = TRAIN_N / ips
    results[bs] = (s, ips, epoch_s)
    print(f"{bs:>6} {s:>9.3f} {ips:>10.1f} {epoch_s:>9.1f} {epoch_s/60:>10.2f}")

# extrapolate with the best configuration
best_bs = min(results, key=lambda b: results[b][2])
epoch_s = results[best_bs][2]
print(f"\n-- extrapolation at batch={best_bs} --")
for epochs in (30, 50, 100):
    per_model_h = epoch_s * epochs / 3600
    print(f"  {epochs:>3} epochs -> {per_model_h:6.2f} h per full training run")

print("\n-- ForgetCheck matrix (full trainings only) --")
# 3 originals + ~21 oracles + 12 RMIA references
n_full = 3 + 21 + 12
for epochs in (30, 50):
    per_model_h = epoch_s * epochs / 3600
    print(f"  {epochs:>3} epochs x {n_full} runs = {per_model_h*n_full:7.1f} GPU-equivalent CPU-hours")
