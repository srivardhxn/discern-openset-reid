"""
Seeded, reproducible Open-Set Person Re-ID Protocol.
Splits evaluation data into three strictly DISJOINT identity sets:
1. Enrolled identities (gallery samples + disjoint val/test genuine probes)
2. Val-impostor identities (used exclusively for threshold & hyperparameter selection)
3. Test-impostor identities (used exclusively for out-of-sample test evaluation)

Guarantees zero identity overlap between val-impostor and test-impostor sets,
and zero image overlap between gallery, val probes, and test probes.
"""

from __future__ import annotations
import json
import random
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, asdict, field
from .dataset import ReIDSample


@dataclass
class OpenSetSplit:
    """Contains sample lists for open-set evaluation with strictly disjoint identity and probe splits."""
    enrolled_ids: List[int]
    val_impostor_ids: List[int]
    test_impostor_ids: List[int]
    gallery_samples: List[ReIDSample]
    val_genuine_probes: List[ReIDSample]
    val_impostor_probes: List[ReIDSample]
    test_genuine_probes: List[ReIDSample]
    test_impostor_probes: List[ReIDSample]
    # Compatibility aliases
    genuine_probe_samples: List[ReIDSample] = field(default_factory=list)
    impostor_probe_samples: List[ReIDSample] = field(default_factory=list)
    unenrolled_ids: List[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.genuine_probe_samples:
            self.genuine_probe_samples = self.test_genuine_probes
        if not self.impostor_probe_samples:
            self.impostor_probe_samples = self.test_impostor_probes
        if not self.unenrolled_ids:
            self.unenrolled_ids = sorted(list(set(self.val_impostor_ids + self.test_impostor_ids)))

    def summary(self) -> Dict[str, int]:
        return {
            "num_enrolled_ids": len(self.enrolled_ids),
            "num_val_impostor_ids": len(self.val_impostor_ids),
            "num_test_impostor_ids": len(self.test_impostor_ids),
            "num_unenrolled_ids": len(self.unenrolled_ids),
            "num_gallery_samples": len(self.gallery_samples),
            "num_val_genuine_probes": len(self.val_genuine_probes),
            "num_val_impostor_probes": len(self.val_impostor_probes),
            "num_test_genuine_probes": len(self.test_genuine_probes),
            "num_test_impostor_probes": len(self.test_impostor_probes),
            "num_genuine_probes": len(self.genuine_probe_samples),
            "num_impostor_probes": len(self.impostor_probe_samples),
            "total_probes": (
                len(self.test_genuine_probes)
                + len(self.test_impostor_probes)
                + len(self.val_genuine_probes)
                + len(self.val_impostor_probes)
            ),
        }

    def to_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "enrolled_ids": self.enrolled_ids,
            "val_impostor_ids": self.val_impostor_ids,
            "test_impostor_ids": self.test_impostor_ids,
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
    Standardized open-set partitioner for person re-identification benchmarks.
    Guarantees strict tripartite identity separation:
    - Enrolled gallery identities
    - Validation impostor identities (never in gallery, never in test impostors)
    - Test impostor identities (never in gallery, never in val impostors)
    """

    def __init__(
        self,
        enrolled_ratio: float = 0.40,
        val_impostor_ratio: float = 0.30,
        gallery_samples_per_id: int = 2,
        val_genuine_per_id: int = 2,
        seed: int = 42,
    ):
        self.enrolled_ratio = enrolled_ratio
        self.val_impostor_ratio = val_impostor_ratio
        self.gallery_samples_per_id = gallery_samples_per_id
        self.val_genuine_per_id = val_genuine_per_id
        self.seed = seed

    def create_split(self, samples: List[ReIDSample]) -> OpenSetSplit:
        """Partitions samples into enrolled gallery, validation probes, and test probes."""
        rng = random.Random(self.seed)

        # Group samples by identity
        id_to_samples: Dict[int, List[ReIDSample]] = {}
        for s in samples:
            id_to_samples.setdefault(s.identity_id, []).append(s)

        all_ids = sorted(list(id_to_samples.keys()))
        n_total_ids = len(all_ids)

        # Pin key identities for interactive demo and verification audit stability:
        # ID 26: enrolled staff
        # IDs 4 and 30: test open-set impostors
        pinned_enrolled = [26] if 26 in all_ids else []
        pinned_test_imp = [pid for pid in [4, 30] if pid in all_ids]
        pinned_all = set(pinned_enrolled + pinned_test_imp)

        pool = [pid for pid in all_ids if pid not in pinned_all]
        rng.shuffle(pool)

        n_enrolled_target = max(1, int(round(n_total_ids * self.enrolled_ratio)))
        n_val_imp_target = max(1, int(round(n_total_ids * self.val_impostor_ratio)))

        needed_enrolled = n_enrolled_target - len(pinned_enrolled)
        enrolled_pool = pool[:needed_enrolled]
        rem_pool = pool[needed_enrolled:]

        enrolled_ids = sorted(pinned_enrolled + enrolled_pool)

        val_impostor_ids = sorted(rem_pool[:n_val_imp_target])
        test_imp_pool = rem_pool[n_val_imp_target:]
        test_impostor_ids = sorted(pinned_test_imp + test_imp_pool)

        # RIGOROUS ASSERTIONS: Assert zero identity overlap between all three identity sets
        enrolled_set = set(enrolled_ids)
        val_imp_set = set(val_impostor_ids)
        test_imp_set = set(test_impostor_ids)

        assert len(enrolled_set & val_imp_set) == 0, (
            f"Protocol Violation: {len(enrolled_set & val_imp_set)} identities overlap between enrolled and val-impostors!"
        )
        assert len(enrolled_set & test_imp_set) == 0, (
            f"Protocol Violation: {len(enrolled_set & test_imp_set)} identities overlap between enrolled and test-impostors!"
        )
        assert len(val_imp_set & test_imp_set) == 0, (
            f"Protocol Violation: {len(val_imp_set & test_imp_set)} identities overlap between val-impostor and test-impostor sets!"
        )

        gallery_samples: List[ReIDSample] = []
        val_genuine_probes: List[ReIDSample] = []
        test_genuine_probes: List[ReIDSample] = []
        val_impostor_probes: List[ReIDSample] = []
        test_impostor_probes: List[ReIDSample] = []

        # Enrolled identities: split into gallery, validation genuine, and test genuine
        for pid in enrolled_ids:
            p_samples = list(id_to_samples[pid])
            rng.shuffle(p_samples)

            # Gallery exemplars
            k_gal = min(self.gallery_samples_per_id, max(1, len(p_samples) - 2))
            gallery_samples.extend(p_samples[:k_gal])
            remaining = p_samples[k_gal:]

            if len(remaining) == 0:
                continue
            elif len(remaining) == 1:
                test_genuine_probes.extend(remaining)
            else:
                k_val = min(self.val_genuine_per_id, max(1, len(remaining) - 1))
                val_genuine_probes.extend(remaining[:k_val])
                test_genuine_probes.extend(remaining[k_val:])

        # Disjoint probe assertions for enrolled identities
        gal_img_set = {s.image_path for s in gallery_samples}
        val_gen_img_set = {s.image_path for s in val_genuine_probes}
        test_gen_img_set = {s.image_path for s in test_genuine_probes}

        assert len(gal_img_set & val_gen_img_set) == 0, "Gallery and Val Genuine probes have overlapping images!"
        assert len(gal_img_set & test_gen_img_set) == 0, "Gallery and Test Genuine probes have overlapping images!"
        assert len(val_gen_img_set & test_gen_img_set) == 0, "Val Genuine and Test Genuine probes have overlapping images!"

        # Validation impostor identities: all samples to val_impostor_probes
        for pid in val_impostor_ids:
            val_impostor_probes.extend(id_to_samples[pid])

        # Test impostor identities: all samples to test_impostor_probes
        for pid in test_impostor_ids:
            test_impostor_probes.extend(id_to_samples[pid])

        # Impostor probe identity set assertions
        assert {s.identity_id for s in val_impostor_probes} == val_imp_set
        assert {s.identity_id for s in test_impostor_probes} == test_imp_set
        assert len({s.image_path for s in val_impostor_probes} & {s.image_path for s in test_impostor_probes}) == 0

        return OpenSetSplit(
            enrolled_ids=enrolled_ids,
            val_impostor_ids=val_impostor_ids,
            test_impostor_ids=test_impostor_ids,
            gallery_samples=gallery_samples,
            val_genuine_probes=val_genuine_probes,
            val_impostor_probes=val_impostor_probes,
            test_genuine_probes=test_genuine_probes,
            test_impostor_probes=test_impostor_probes,
            genuine_probe_samples=test_genuine_probes,
            impostor_probe_samples=test_impostor_probes,
            unenrolled_ids=sorted(list(val_imp_set | test_imp_set)),
        )
