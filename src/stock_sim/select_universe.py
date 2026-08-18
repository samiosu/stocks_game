"""CLI entry point for sector-wise stock selection."""

from __future__ import annotations

import argparse

from .config import config_path, load_config, runtime_as_of, setup_logging
from .metadata import YahooMetadataResolver
from .universe import select_universe


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--as-of", default=None, help="As-of date, YYYY-MM-DD")
    parser.add_argument("--master", default=None, help="JPX/manual stock master CSV")
    parser.add_argument("--no-yahoo-metadata", action="store_true")
    args = parser.parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    as_of = runtime_as_of(config, args.as_of)
    resolver = None
    if not args.no_yahoo_metadata:
        fetch = config.get("fetch", {})
        resolver = YahooMetadataResolver(
            retries=int(fetch.get("retries", 3)),
            timeout_seconds=int(fetch.get("timeout_seconds", 30)),
            cache_path=config_path(config, "metadata_cache"),
        )
    select_universe(
        args.master or config_path(config, "stock_master"),
        as_of=as_of,
        start_date=config.get("data", {}).get("start_date", "2015-01-01"),
        sector_mapping_path=config_path(config, "sector_mapping"),
        sector_overrides_path=config_path(config, "sector_overrides"),
        listing_overrides_path=config_path(config, "listing_overrides"),
        output_path=config_path(config, "selected_universe"),
        rejection_path=config_path(config, "selected_universe").parent / "metadata" / "universe_rejections.csv",
        metadata_resolver=resolver,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
