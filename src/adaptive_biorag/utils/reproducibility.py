from __future__ import annotations

import os
import platform
import random
import sys


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:  # optional
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def environment_info() -> dict:
    """Only reports what can actually be observed; unavailable values are None."""
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "numpy": None, "torch": None, "cuda_available": None, "gpu": None}
    try:
        import numpy as np
        info["numpy"] = np.__version__
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def peak_rss_mb() -> float | None:
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(v / 1024.0 if sys.platform != "darwin" else v / (1024.0 * 1024.0), 1)
    except Exception:
        return None
