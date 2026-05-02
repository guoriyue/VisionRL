"""Preference-pair datasets for offline DPO training.

Currently supports the Pick-a-Pic v2 schema (HuggingFace
``yuvalkirstain/pickapic_v2``):
  * ``jpg_0``, ``jpg_1`` — image bytes
  * ``label_0`` ∈ {0, 0.5, 1} — 1 ⇔ jpg_0 is the winner
  * ``caption`` — text prompt

The collate function returns batches with the convention used by the
reference Diffusion-DPO repo: ``pixel_values`` has shape ``[B, 6, H, W]``
where channels 0:3 are the *winner* image and channels 3:6 are the
*loser*. Trainers split this into ``[2B, 3, H, W]`` before VAE-encoding.
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class PreferenceBatch:
    """A collated batch of preference pairs.

    ``pixel_values`` channel layout (along dim=1): winner-then-loser.
    Splitting helper: ``batch.split_winner_loser()`` returns two
    ``[B, 3, H, W]`` tensors.
    """

    pixel_values: torch.Tensor   # [B, 6, H, W]
    captions: list[str]

    def split_winner_loser(self) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.pixel_values.shape[1] // 2
        return self.pixel_values[:, :c], self.pixel_values[:, c:]

    def stacked_winner_then_loser(self) -> torch.Tensor:
        """Return ``[2B, 3, H, W]`` — winner block first, loser second.

        This is the layout consumed by ``diffusion_dpo_loss``.
        """
        winner, loser = self.split_winner_loser()
        return torch.cat([winner, loser], dim=0)


class PickAPicPreferenceDataset(Dataset):
    """Wraps a HuggingFace ``datasets.Dataset`` of Pick-a-Pic samples.

    Filters out 0.5/0.5 split decisions on construction.
    Resizes + center-crops to ``resolution`` (square).
    """

    def __init__(
        self,
        hf_dataset: Any,
        resolution: int = 512,
        random_crop: bool = False,
        no_hflip: bool = False,
    ) -> None:
        from torchvision import transforms

        # Strip indecisive labels (0.5 means tie)
        keep_idx = [
            i for i, lbl in enumerate(hf_dataset["label_0"]) if lbl in (0, 1)
        ]
        if len(keep_idx) < len(hf_dataset):
            hf_dataset = hf_dataset.select(keep_idx)
        self._ds = hf_dataset

        ops: list[Any] = [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        ]
        ops.append(
            transforms.RandomCrop(resolution) if random_crop
            else transforms.CenterCrop(resolution)
        )
        if not no_hflip:
            ops.append(transforms.RandomHorizontalFlip())
        ops += [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # → [-1, 1]
        ]
        self._tx = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from PIL import Image

        row = self._ds[idx]
        # Winner is whichever of jpg_0 / jpg_1 has label==1
        if row["label_0"] == 1:
            w_bytes, l_bytes = row["jpg_0"], row["jpg_1"]
        else:
            w_bytes, l_bytes = row["jpg_1"], row["jpg_0"]
        winner = self._tx(Image.open(io.BytesIO(w_bytes)).convert("RGB"))
        loser = self._tx(Image.open(io.BytesIO(l_bytes)).convert("RGB"))
        # Stack on channel dim — winner-then-loser convention
        pix = torch.cat([winner, loser], dim=0)  # [6, H, W]
        return {"pixel_values": pix, "caption": row["caption"]}


def collate_preference(examples: Iterable[dict[str, Any]]) -> PreferenceBatch:
    """Collate a list of __getitem__ outputs into a ``PreferenceBatch``."""
    items = list(examples)
    pixel_values = torch.stack([e["pixel_values"] for e in items])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    captions = [e["caption"] for e in items]
    return PreferenceBatch(pixel_values=pixel_values, captions=captions)


def load_pickapic(
    split: str = "train",
    cache_dir: str | None = None,
    streaming: bool = False,
    max_samples: int | None = None,
    resolution: int = 512,
    random_crop: bool = False,
    no_hflip: bool = False,
    dataset_name: str = "yuvalkirstain/pickapic_v2",
) -> PickAPicPreferenceDataset:
    """One-liner loader: returns a ready-to-iterate PyTorch Dataset.

    Requires ``datasets`` and ``torchvision``. The first call streams
    Pick-a-Pic v2 (~250 GB) — use ``cache_dir`` and consider
    ``yuvalkirstain/pickapic_v2_no_images`` for metadata-only debugging.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir, streaming=streaming)
    if max_samples is not None and not streaming:
        ds = ds.select(range(min(max_samples, len(ds))))
    return PickAPicPreferenceDataset(
        ds, resolution=resolution, random_crop=random_crop, no_hflip=no_hflip,
    )
