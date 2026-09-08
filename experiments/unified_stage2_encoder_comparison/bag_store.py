from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PackedBagStore:
    """将小规模患者队列一次打包并常驻于执行设备。"""

    patient_ids: tuple[str, ...]
    feature_mode: str
    central: torch.Tensor | None
    whorl: torch.Tensor | None
    pair_features: torch.Tensor | None
    morphology: torch.Tensor
    labels: torch.Tensor
    pair_mask: torch.Tensor
    index: dict[str, int]

    @classmethod
    def from_bags(cls, bags: dict[str, dict], device: torch.device) -> "PackedBagStore":
        patient_ids = tuple(sorted(bags))
        if not patient_ids:
            raise ValueError("cannot pack an empty patient cohort")
        pair_mode = "pair_features" in bags[patient_ids[0]]
        feature_key = "pair_features" if pair_mode else "central"
        if any(("pair_features" in bags[pid]) != pair_mode for pid in patient_ids):
            raise ValueError("cannot mix native-pair and region bags")
        raw_dim = int(bags[patient_ids[0]][feature_key].shape[1])
        max_pairs = max(int(bags[pid][feature_key].shape[0]) for pid in patient_ids)
        n = len(patient_ids)
        pair_features = (
            torch.zeros(n, max_pairs, raw_dim, dtype=torch.float32)
            if pair_mode else None
        )
        central = (
            None if pair_mode else torch.zeros(n, max_pairs, raw_dim, dtype=torch.float32)
        )
        whorl = None if pair_mode else torch.zeros_like(central)
        morphology = torch.zeros(n, 21, dtype=torch.float32)
        labels = torch.empty(n, dtype=torch.long)
        pair_mask = torch.zeros(n, max_pairs, dtype=torch.bool)
        for row, pid in enumerate(patient_ids):
            bag = bags[pid]
            features = bag[feature_key]
            count = int(features.shape[0])
            if features.shape[1] != raw_dim or not 1 <= count <= max_pairs:
                raise ValueError(f"invalid bag shape for {pid}: {tuple(features.shape)}")
            if pair_mode:
                pair_features[row, :count] = features.float()
            else:
                if bag["central"].shape != bag["whorl"].shape:
                    raise ValueError(f"central/whorl mismatch for {pid}")
                central[row, :count] = bag["central"].float()
                whorl[row, :count] = bag["whorl"].float()
            morphology[row] = bag["morphology"].float()
            labels[row] = int(bag["label"])
            pair_mask[row, :count] = True
        return cls(
            patient_ids=patient_ids,
            feature_mode="native_pair" if pair_mode else "region",
            central=central.to(device) if central is not None else None,
            whorl=whorl.to(device) if whorl is not None else None,
            pair_features=pair_features.to(device) if pair_features is not None else None,
            morphology=morphology.to(device),
            labels=labels.to(device),
            pair_mask=pair_mask.to(device),
            index={pid: row for row, pid in enumerate(patient_ids)},
        )

    @property
    def device(self) -> torch.device:
        features = self.pair_features if self.pair_features is not None else self.central
        if features is None:
            raise RuntimeError("packed bag store has no features")
        return features.device

    def label(self, patient_id: str) -> int:
        return int(self.labels[self.index[patient_id]].item())

    def batch(
        self, patient_ids: list[str] | tuple[str, ...]
    ) -> tuple[torch.Tensor | None, ...]:
        rows = torch.tensor([self.index[pid] for pid in patient_ids], device=self.device)
        primary = self.pair_features if self.feature_mode == "native_pair" else self.central
        secondary = None if self.feature_mode == "native_pair" else self.whorl
        if primary is None:
            raise RuntimeError("packed bag store primary features are missing")
        return (
            primary.index_select(0, rows),
            secondary.index_select(0, rows) if secondary is not None else None,
            self.morphology.index_select(0, rows),
            self.labels.index_select(0, rows),
            self.pair_mask.index_select(0, rows),
        )
