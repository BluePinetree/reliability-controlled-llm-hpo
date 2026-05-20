from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="environment/runtime_report.json")
    args = parser.parse_args()

    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "created_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }
    try:
        import torch

        report.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except Exception as exc:
        report["torch_error"] = str(exc)
    try:
        import torchvision

        report["torchvision"] = torchvision.__version__
    except Exception as exc:
        report["torchvision_error"] = str(exc)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(out)


if __name__ == "__main__":
    main()
