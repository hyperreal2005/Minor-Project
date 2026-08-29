"""CIFAR ResNet-18 and the feature taps the representation audit reads.

Architecture choice: torchvision's ``resnet18`` with the ImageNet stem replaced by a 3x3
stride-1 convolution and the max-pool removed. This is the adaptation the CIFAR unlearning
literature uses — RUM, the vision-transformer benchmark, and the NeurIPS competition starter kit
all train this variant — so our numbers sit on the same scale as the published ones. It is *not*
the He et al. CIFAR ResNet family (ResNet-20/32/56), which is a different architecture despite
the similar name.

The ImageNet stem downsamples 224x224 by 4x before the first residual stage. Applied to a 32x32
input that leaves an 8x8 feature map going into ``layer1``, which throws away most of the spatial
information CIFAR has. Hence the standard patch.

Feature taps for the representation audit are the four residual stages. They are read through
forward hooks and **global-average-pooled over spatial dimensions before leaving this module** — raw ``layer1`` output for 3000 probes is about
786 MB per model, which makes storing it for ~300 models untenable (implementation plan §5). GAP
discards spatial structure; that is the accepted convention for CKA on convnets and a limitation
the paper must state.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Final, Iterator, Mapping, Sequence

import torch
import torch.nn as nn
import torchvision

__all__ = [
    "make_resnet18",
    "FEATURE_LAYERS",
    "ALL_FEATURE_LAYERS",
    "extract_features",
    "feature_dims",
]

#: The taps the representation audit reads, in depth order. These names are also the
#: ``probe_set`` values registered for representation metrics in ``configs/metrics.yaml``.
#:
#: ``penultimate`` is deliberately **not** here. In ResNet-18 ``avgpool`` immediately follows
#: ``layer4`` and is an ``AdaptiveAvgPool2d((1, 1))`` — exactly the global average pool this
#: module already applies to every tap — so ``penultimate`` is numerically identical to
#: ``layer4`` (there is a test asserting this). Storing both would add 512 floats per probe for
#: zero information: about 35% of the project's activation storage, roughly 600 MB across the
#: full matrix, all of it duplicate. It remains requestable by name for architectures where the
#: two would differ.
FEATURE_LAYERS: Final[tuple[str, ...]] = ("layer1", "layer2", "layer3", "layer4")

#: Every tap that can be requested, including the redundant one.
ALL_FEATURE_LAYERS: Final[tuple[str, ...]] = FEATURE_LAYERS + ("penultimate",)

#: Channel counts after global-average-pooling, for ResNet-18's BasicBlock stages.
_DIMS: Final[dict[str, int]] = {
    "layer1": 64,
    "layer2": 128,
    "layer3": 256,
    "layer4": 512,
    "penultimate": 512,
}


def feature_dims(layers: Sequence[str] = FEATURE_LAYERS) -> dict[str, int]:
    """Pooled feature width per tap. The four default taps total 960 floats per probe."""
    return {name: _DIMS[name] for name in layers}


def make_resnet18(num_classes: int = 10, *, seed: int | None = None) -> nn.Module:
    """Build a CIFAR-adapted ResNet-18 with freshly initialised weights.

    Args:
        num_classes: 10 for CIFAR-10, 100 for CIFAR-100.
        seed: if given, seeds torch immediately before construction so that initialisation is
            reproducible. Paired M0/oracle runs rely on this: sharing ``train_seed`` must mean
            sharing an initialisation, or the two models differ for a reason that has nothing to
            do with the forget set.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    if seed is not None:
        # conv1 was replaced after the seeded construction above, so re-initialise it under the
        # same seed rather than leaving it drawn from whatever state the RNG reached.
        torch.manual_seed(seed)
        nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")

    return model


@contextmanager
def _taps(model: nn.Module, layers: Sequence[str]) -> Iterator[dict[str, torch.Tensor]]:
    """Attach GAP forward hooks to the named stages, and always remove them."""
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def hook(name: str):
        def fn(_module, _inputs, output):
            # Pool here, not later: the whole point is that the un-pooled tensor never leaves.
            captured[name] = output.mean(dim=(2, 3)).detach()

        return fn

    for name in layers:
        if name == "penultimate":
            continue  # taken from avgpool below, which is already spatially reduced
        module = getattr(model, name, None)
        if module is None:
            raise AttributeError(f"model has no stage named {name!r}")
        handles.append(module.register_forward_hook(hook(name)))

    if "penultimate" in layers:
        def pen_hook(_module, _inputs, output):
            captured["penultimate"] = torch.flatten(output, 1).detach()

        handles.append(model.avgpool.register_forward_hook(pen_hook))

    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader,
    *,
    layers: Sequence[str] = FEATURE_LAYERS,
    device: str | torch.device = "cpu",
    max_batches: int | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Run ``loader`` through ``model`` and collect pooled activations.

    Returns:
        ``({layer: (n_probe, dim) float32}, logits)``. Logits come back too because the
        behavioural audit needs them over the same probe set, and running the probes twice is
        both wasteful and a chance for the two passes to disagree.

    The loader must be deterministic and unshuffled — activations are only comparable across
    models if row *i* is the same example in every one of them. :func:`forgetcheck.data.cifar.
    make_loader` enforces this for evaluation loaders.
    """
    unknown = set(layers) - set(ALL_FEATURE_LAYERS)
    if unknown:
        raise ValueError(
            f"unknown feature layers {sorted(unknown)}; known: {ALL_FEATURE_LAYERS}"
        )

    model = model.to(device).eval()
    chunks: dict[str, list[torch.Tensor]] = {name: [] for name in layers}
    logit_chunks: list[torch.Tensor] = []

    with _taps(model, layers) as captured:
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            logits = model(x.to(device))
            logit_chunks.append(logits.detach().float().cpu())
            for name in layers:
                if name not in captured:
                    raise RuntimeError(f"tap {name!r} produced no output; hook not firing")
                chunks[name].append(captured[name].float().cpu())
            captured.clear()

    if not logit_chunks:
        raise ValueError("loader yielded no batches")

    feats = {name: torch.cat(parts, dim=0) for name, parts in chunks.items()}
    return feats, torch.cat(logit_chunks, dim=0)


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad or not trainable_only
    )
