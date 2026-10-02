"""
Seeded, reproducible Open-Set Person Re-ID Protocol.
Splits evaluation data into:
- Enrolled identities (gallery samples + genuine positive probes)
- Unenrolled identities (impostor negative probes, never in gallery)
"""

from __future__ import annotations
import json
import random
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, asdict
from .dataset import ReIDSample

@dataclass
class OpenSetSplit:
    """Contains sample lists for open-set evaluation."""
    enrolled_ids: List[int]
    unenrolled_ids: List[int]
    gallery_samples: List[ReIDSample]
    genuine_probe_samples: List[ReIDSample]
    impostor_probe_samples: List[ReIDSample]

    def summary(self) -> Dict[str, int]:
        return {
            "num_enrolled_ids": len(self.enrolled_ids),
            "num_unenrolled_ids": len(self.unenrolled_ids),
            "num_gallery_samples": len(self.gallery_samples),
            "num_genuine_probes": len(self.genuine_probe_samples),
            "num_impostor_probes": len(self.impostor_probe_samples),
            "total_probes": len(self.genuine_probe_samples) + len(self.impostor_probe_samples),
        }

    def to_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "enrolled_ids": self.enrolled_ids,
            "unenrolled_ids": self.unenrolled_ids,
            "gallery_samples": [asdict(s) for s in self.gallery_samples],
            "genuine_probe_samples": [asdict(s) for s in self.genuine_probe_samples],
            "impostor_probe_samples": [asdict(s) for s in self.impostor_probe_samples],
        }


class OpenSetProtocol:
    """
    Standardized open-set partitioner for re-id benchmarks.
    Guarantees strict separation between enrolled gallery identities
    and open-set impostors.
    """

    def __init__(self, enrolled_ratio: float = 0.5, gallery_samples_per_id: int = 2, seed: int = 42):
        self.enrolled_ratio = enrolled_ratio
        self.gallery_samples_per_id = gallery_samples_per_id
        self.seed = seed

    def create_split(self, samples: List[ReIDSample]) -> OpenSetSplit:
        """Partitions samples into enrolled gallery, genuine probes, and impostor probes."""
        rng = random.Random(self.seed)

        # Group samples by identity
        id_to_samples: Dict[int, List[ReIDSample]] = {}
        for s in samples:
            id_to_samples.setdefault(s.identity_id, []).append(s)

        all_ids = sorted(list(id_to_samples.keys()))
        rng.shuffle(all_ids)

        num_enrolled = max(1, int(len(all_ids) * self.enrolled_ratio))
        enrolled_ids = sorted(all_ids[:num_enrolled])
        unenrolled_ids = sorted(all_ids[num_enrolled:])

        gallery_samples: List[ReIDSample] = []
        genuine_probes: List[ReIDSample] = []
        impostor_probes: List[ReIDSample] = []

        # Enrolled identities: split into gallery and genuine queries
        for pid in enrolled_ids:
            p_samples = list(id_to_samples[pid])
            rng.shuffle(p_samples)

            k = min(self.gallery_samples_per_id, len(p_samples) - 1)
            k = max(1, k)
            gallery_samples.extend(p_samples[:k])
            genuine_probes.extend(p_samples[k:])

        # Unenrolled identities: all become impostor queries
        for pid in unenrolled_ids:
            impostor_probes.extend(id_to_samples[pid])

        return OpenSetSplit(
            enrolled_ids=enrolled_ids,
            unenrolled_ids=unenrolled_ids,
            gallery_samples=gallery_samples,
            genuine_probe_samples=genuine_probes,
            impostor_probe_samples=impostor_probes,
        )
