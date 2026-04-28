from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from vfm_gs.scorers.vfm_cache import validate_cache


def main(argv=None):
    parser = ArgumentParser(description="Validate a VFM feature cache manifest and entries.")
    parser.add_argument("--cache_dir", "-c", required=True, type=str)
    parser.add_argument("--backend", default=None, type=str)
    parser.add_argument("--source_path", "-s", default=None, type=str)
    parser.add_argument("--images", "-i", default=None, type=str)
    parser.add_argument("--skip_checksum", action="store_true")
    args = parser.parse_args(argv)

    errors, warnings, manifest = validate_cache(
        args.cache_dir,
        backend=args.backend,
        source_path=args.source_path,
        images=args.images,
        check_checksum=not args.skip_checksum,
    )

    for warning in warnings:
        print("[WARN] {}".format(warning))
    if errors:
        for error in errors:
            print("[ERROR] {}".format(error))
        raise SystemExit(1)

    print(
        "Validated {} {} cache entries in {}".format(
            len(manifest.get("entries", {})),
            manifest.get("backend", "<unknown>"),
            Path(args.cache_dir).resolve(),
        )
    )


if __name__ == "__main__":
    main()
