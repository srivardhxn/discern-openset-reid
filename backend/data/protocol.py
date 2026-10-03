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
from dataclasses import dataclass, asdict, field
from .dataset import ReIDSample

@dataclass
class OpenSetSplit:
    """Contains sample lists for open-set evaluation with disjoint validation and test splits."""
    enrolled_ids: List[int]
    unenrolled_ids: List[int]
    gallery_samples: List[ReIDSample]
    genuine_probe_samples: List[ReIDSample]  # Default/test genuine probes
    impostor_probe_samples: List[ReIDSample] # Default/test impostor probes
    val_genuine_probes: List[ReIDSample] = field(default_factory=list)
    val_impostor_probes: List[ReIDSample] = field(default_factory=list)
    test_genuine_probes: List[ReIDSample] = field(default_factory=list)
    test_impostor_probes: List[ReIDSample] = field(default_factory=list)

    def summary(self) -> Dict[str, int]:
        return {
            "num_enrolled_ids": len(self.enrolled_ids),
            "num_unenrolled_ids": len(self.unenrolled_ids),
            "num_gallery_samples": len(self.gallery_samples),
            "num_val_genuine_probes": len(self.val_genuine_probes),
            "num_val_impostor_probes": len(self.val_impostor_probes),
            "num_test_genuine_probes": len(self.test_genuine_probes),
            "num_test_impostor_probes": len(self.test_impostor_probes),
            "num_genuine_probes": len(self.genuine_probe_samples),
            "num_impostor_probes": len(self.impostor_probe_samples),
            "total_probes": len(self.test_genuine_probes) + len(self.test_impostor_probes) + len(self.val_genuine_probes) + len(self.val_impostor_probes),
        }

    def to_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "enrolled_ids": self.enrolled_ids,
            "unenrolled_ids": self.unenrolled_ids,
            "gallery_samples": [asdict(s) for s in self.gallery_samples],
            "val_genuine_probes": [asdict(s) for s in self.val_genuine_probes],
            "val_impostor_probes": [asdict(s) for s in self.val_impostor_probes],
            "test_genuine_probes": [asdict(s) for s in self.test_genuine_probes],
            "test_impostor_probes": [asdict(s) for s in self.test_impostor_probes],
            "genuine_probe_samples": [asdict(s) for s in self.genuine_probe_samples],
            "impostor_probe_samples": [asdict(s) for s in self.impostor_probe_samples],
        }


class OpenSetProtocol:
    """
    Standardized open-set partitioner for re-id benchmarks.
    Guarantees strict separation between enrolled gallery identities
    and open-set impostors, and isolates validation probes from test probes.
    """

    def __init__(
        self,
        enrolled_ratio: float = 0.5,
        gallery_samples_per_id: int = 2,
        val_ratio: float = 0.30,
        seed: int = 42,
    ):
        self.enrolled_ratio = enrolled_ratio
        self.gallery_samples_per_id = gallery_samples_per_id
        self.val_ratio = val_ratio
        self.seed = seed

    def create_split(self, samples: List[ReIDSample]) -> OpenSetSplit:
        """Partitions samples into enrolled gallery, validation probes, and test probes."""
        rng = random.Random(self.seed)

        # Group samples by identity
        id_to_samples: Dict[int, List[ReIDSample]] = {}
        for s in samples:
            id_to_samples.setdefault(s.identity_id, []).append(s)

        all_ids = sorted(list(id_to_samples.keys()))
        # Anchor demo probe identities for API scenario stability:
        # ID 26: enrolled staff
        # IDs 4 and 30: unenrolled open-set impostors
        pinned_enrolled = [26] if 26 in all_ids else []
        pinned_unenrolled = [pid for pid in [4, 30] if pid in all_ids]
        pool = [pid for pid in all_ids if pid not in pinned_enrolled and pid not in pinned_unenrolled]
        rng.shuffle(pool)

        num_needed_enrolled = max(1, int(len(all_ids) * self.enrolled_ratio)) - len(pinned_enrolled)
        enrolled_ids = sorted(pinned_enrolled + pool[:num_needed_enrolled])
        unenrolled_ids = sorted(pinned_unenrolled + pool[num_needed_enrolled:])

        gallery_samples: List[ReIDSample] = []
        val_genuine_probes: List[ReIDSample] = []
        test_genuine_probes: List[ReIDSample] = []
        val_impostor_probes: List[ReIDSample] = []
        test_impostor_probes: List[ReIDSample] = []

        # Enrolled identities: split into gallery, validation genuine, and test genuine
        for pid in enrolled_ids:
            p_samples = list(id_to_samples[pid])
            rng.shuffle(p_samples)

            # Gallery exemplars (at least 1, up to gallery_samples_per_id)
            k = min(self.gallery_samples_per_id, max(1, len(p_samples) - 2))
            gallery_samples.extend(p_samples[:k])
            remaining = p_samples[k:]

            if len(remaining) == 0:
                continue
            elif len(remaining) == 1:
                # Put in test genuine
                test_genuine_probes.extend(remaining)
            else:
                # Put 1 in val genuine, remainder in test genuine
                val_genuine_probes.append(remaining[0])
                test_genuine_probes.extend(remaining[1:])

        # Unenrolled identities: split into validation impostors and test impostors
        for pid in unenrolled_ids:
            p_samples = list(id_to_samples[pid])
            rng.shuffle(p_samples)

            if len(p_samples) <= 2:
                test_impostor_probes.extend(p_samples)
            else:
                # 1 or 2 for validation, rest for test
                n_val = max(1, int(len(p_samples) * self.val_ratio))
                val_impostor_probes.extend(p_samples[:n_val])
                test_impostor_probes.extend(p_samples[n_val:])

        return OpenSetSplit(
            enrolled_ids=enrolled_ids,
            unenrolled_ids=unenrolled_ids,
            gallery_samples=gallery_samples,
            genuine_probe_samples=test_genuine_probes,
            impostor_probe_samples=test_impostor_probes,
            val_genuine_probes=val_genuine_probes,
            val_impostor_probes=val_impostor_probes,
            test_genuine_probes=test_genuine_probes,
            test_impostor_probes=test_impostor_probes,
        )
