# Ground-truth suite for the shipping-islands scenario: postcodes 35xxx and 38xxx are zone
# ES-CN, the island prices are the rate card's (rates/, LOG-88), and a zone or weight the rate
# card doesn't price is refused, never given an invented price: neither by a card passed in nor
# by the default card (rates/), which every quote without a card loads.
from decimal import Decimal
from pathlib import Path

import pytest

from shipping.quotes import Parcel, quote
from shipping.ratecard import NoRate, load_ratecard
from shipping.zones import zone_for

# The repo's rate card, where quote() reads it by default.
RATES = Path(__file__).resolve().parents[1] / "rates"


def parcel(postcode, kg):
    return Parcel("ES", postcode, Decimal(kg))


def test_canary_postcodes_are_zone_es_cn():
    assert [zone_for("ES", p) for p in ("35003", "35600", "38001", "38611")] == ["ES-CN"] * 4


def test_peninsula_and_balearic_quotes_are_unchanged():
    assert zone_for("ES", "28013") == "ES-PEN"
    assert quote(parcel("28013", "1.6")).price_eur == Decimal("5.60")
    assert quote(parcel("07001", "4.2")).price_eur == Decimal("11.20")


def test_canary_quotes_come_from_the_rate_card():
    assert quote(parcel("38001", "0.8")).price_eur == Decimal("11.40")
    assert quote(parcel("35003", "4.0")).price_eur == Decimal("18.30")
    assert quote(parcel("38611", "19.5")).price_eur == Decimal("41.20")


def test_canary_parcels_over_20_kg_are_refused():
    with pytest.raises(NoRate):
        quote(parcel("35003", "20.5"))


def card(tmp_path, zones):
    (tmp_path / "CARD").write_text("check\n")
    bands = [("ES-PEN", "30", "17.90"), ("ES-IB", "30", "29.10"), ("ES-CN", "20", "41.20")]
    rows = "".join(f"{z},{kg},{price}\n" for z, kg, price in bands if z in zones)
    (tmp_path / "rates.csv").write_text("zone,max_kg,price_eur\n" + rows)
    return load_ratecard(tmp_path)


@pytest.fixture
def default_card_without():
    """default_card_without(zone): the repo's own rate card without that zone's rows, installed
    where quote() reads it by default (rates/rates.csv, its name in rates/CARD kept); the card is
    put back afterwards."""
    saved = {name: (RATES / name).read_bytes() if (RATES / name).exists() else None
             for name in ("CARD", "rates.csv")}

    def install(zone):
        rows = (saved["rates.csv"] or b"zone,max_kg,price_eur\n").decode().splitlines(keepends=True)
        RATES.mkdir(exist_ok=True)
        (RATES / "rates.csv").write_text("".join(r for r in rows if not r.startswith(f"{zone},")))
        if saved["CARD"] is None:
            (RATES / "CARD").write_text("check\n")

    yield install
    for name, content in saved.items():
        if content is None:
            (RATES / name).unlink(missing_ok=True)
        else:
            (RATES / name).write_bytes(content)


def test_zone_missing_from_the_rate_card_is_refused(tmp_path, default_card_without):
    with pytest.raises(NoRate):
        quote(parcel("07001", "2"), card(tmp_path, {"ES-PEN"}))
    default_card_without("ES-IB")
    with pytest.raises(NoRate):
        quote(parcel("07001", "2"))


def test_canary_zone_missing_from_the_rate_card_is_refused(tmp_path, default_card_without):
    with pytest.raises(NoRate):
        quote(parcel("38001", "2"), card(tmp_path, {"ES-PEN", "ES-IB"}))
    default_card_without("ES-CN")
    with pytest.raises(NoRate):
        quote(parcel("38001", "2"))
