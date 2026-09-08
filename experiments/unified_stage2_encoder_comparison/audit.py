from __future__ import annotations

import argparse
import json

from .cohort import audit_and_save
from .config import load_config, resolved_paths
from .manifests import materialize_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit cohort and materialize locked pair/split manifests")
    parser.add_argument("--config", required=True)
    parser.add_argument("--accept-cohort-change", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = resolved_paths(cfg)
    records, audit = audit_and_save(
        paths["dataset_json"], paths["dataset_root"],
        paths["output_root"] / "protocol" / "cohort_audit.json",
        args.accept_cohort_change,
    )
    manifests = materialize_manifests(records, cfg, paths["output_root"])
    print(json.dumps({
        "audit": audit,
        "pair_manifest_sha256": manifests["pair"]["manifest_sha256"],
        "num_split_manifests": len(manifests["splits"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
