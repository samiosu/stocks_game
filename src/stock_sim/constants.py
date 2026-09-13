"""Stable IDs and names shared by every pipeline stage."""

SECTOR_DEFINITIONS = (
    ("energy", "エネルギー"),
    ("materials", "素材"),
    ("industrials", "資本財・サービス"),
    ("consumer_discretionary", "一般消費財・サービス"),
    ("consumer_staples", "生活必需品"),
    ("health_care", "ヘルスケア"),
    ("financials", "金融"),
    ("information_technology", "情報技術"),
    ("communication_services", "コミュニケーション・サービス"),
    ("utilities", "公益事業"),
    ("real_estate", "不動産"),
)

SECTOR_IDS = tuple(sector_id for sector_id, _ in SECTOR_DEFINITIONS)
SECTOR_NAMES = dict(SECTOR_DEFINITIONS)

DEFAULT_EVENT_TYPES = (
    "ai_investment",
    "rates_up",
    "rates_down",
    "yen_weakness",
    "yen_strength",
    "semiconductor_shortage",
    "entertainment_hit",
    "recession",
    "market_shock",
)

STOCK_FEATURES = (
    "return_1d",
    "return_5d",
    "oc_change",
    "hl_range",
    "volume_log_change",
    "vol_5d",
    "vol_20d",
    "excess_topix",
    "excess_n225",
)

OHLCV_FIELDS = ("open", "high", "low", "close", "volume")


def sector_name(sector_id: str) -> str:
    """Return the canonical Japanese name for a sector ID."""

    return SECTOR_NAMES[sector_id]
