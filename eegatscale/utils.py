from typing import Tuple

import numpy as np
import torch
from jsonargparse._namespace import Namespace


def _make_span_from_seeds(seeds: np.ndarray, span: int, total: None | int = None) -> np.ndarray:
    inds = list()
    for seed in seeds:
        for i in range(seed, seed + span):
            if total is not None and i >= total:
                break
            elif i not in inds:
                inds.append(int(i))
    return np.array(inds)


def _make_mask(shape: Tuple[int, int], prob: float, total: int, span: int, allow_no_inds: bool = False) -> torch.Tensor:
    mask = torch.zeros(shape, requires_grad=False, dtype=torch.bool)

    for i in range(shape[0]):
        mask_seeds = list()
        while not allow_no_inds and len(mask_seeds) == 0 and prob > 0:
            mask_seeds = np.nonzero(np.random.rand(total) < prob)[0]

        mask[i, _make_span_from_seeds(mask_seeds, span, total=total)] = True

    return mask


def flatten_namespace(namespace: Namespace) -> dict:
    """ Flatting a nested namespace into a dict. Will also convert None to strings."""
    d = dict(namespace)
    new_d = {}
    for k, v in d.items():
        if isinstance(v, (list, tuple)):
            for i, elem in enumerate(v):
                if isinstance(elem, Namespace):
                    d_elem = dict(elem)
                    for k2, v2 in d_elem.items():
                        new_d[f"{k}.{i}.{k2}"] = v2 if v2 is not None else "None"
                else:
                    new_d[f"{k}.{i}"] = elem if elem is not None else "None"
        else:
            new_d[k] = v if v is not None else "None"
    return new_d
