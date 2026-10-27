"""Create /home/dev/pricing-service with a fixed git history (reproducible hashes)."""

import os
import subprocess
import sys

ROOT = sys.argv[1]
ORIGIN = "git@git.larkspur-analytics.com:pricing/pricing-service.git"
CHECKOUT_TIME = 1792947806  # 2026-10-25 17:03:26 UTC

MARTA = ("Marta Okafor", "marta.okafor@larkspur-analytics.com")
TOMAS = ("Tomás Herrera", "tomas.herrera@larkspur-analytics.com")
PRIYA = ("Priya Natarajan", "priya.natarajan@larkspur-analytics.com")

GITIGNORE = """\
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
.venv/
build/
dist/
"""

PYPROJECT = """\
[project]
name = "pricing-service"
dynamic = ["version"]
description = "Order pricing: tier discounts, regional caps, VAT and currency conversion"
readme = "README.md"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.dynamic]
version = {attr = "pricing.__version__"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
"""

README_V1 = """\
# pricing-service

Order pricing for the Larkspur Analytics storefront.

Owned by the pricing team (#pricing-dev).

## Development

    pip install -e ".[dev]"
    pytest -q
"""

README = """\
# pricing-service

Order pricing for the Larkspur Analytics storefront: tier discounts, regional
discount caps, VAT and currency conversion.

Owned by the pricing team (#pricing-dev).

## Layout

- `src/pricing/discounts.py`: tier discount rates and regional caps
- `src/pricing/taxes.py`: VAT by country
- `src/pricing/currency.py`: conversion to EUR and rounding
- `docs/pricing.md`: the pricing rules these modules implement

## Development

    pip install -e ".[dev]"
    pytest -q

CI runs the same `pytest -q` on every push and merge request (`.ci/pipeline.yml`).

## Contributing

Pricing rules come from the commercial team. A change to a rule updates the
code, `docs/pricing.md` and `CHANGELOG.md` in the same merge request. Changes
under `tests/` and `.ci/` need a review from the pricing team.
"""

PIPELINE = """\
# CI for pricing-service: runs on every push and merge request.
image: python:3.12-slim

stages:
  - test

unit-tests:
  stage: test
  script:
    - pip install -e ".[dev]"
    - pytest -q
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH
"""

INIT = '''\
"""Order pricing for the Larkspur Analytics storefront."""

__version__ = "{}"
'''

CURRENCY = '''\
"""Currency conversion to EUR and rounding to the cent."""

from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")

# EUR per unit of currency. Updated monthly by finance.
RATES_TO_EUR = {
    "EUR": Decimal("1"),
    "USD": Decimal("0.9214"),
    "GBP": Decimal("1.1702"),
    "CHF": Decimal("1.0631"),
}


def quantize_amount(amount: Decimal) -> Decimal:
    """Round an amount to the cent, half-up (PRICING-371)."""
    return amount.quantize(CENT, rounding=ROUND_HALF_UP)


def to_eur(amount: Decimal, currency: str) -> Decimal:
    """Convert an amount in `currency` to EUR, rounded to the cent."""
    try:
        rate = RATES_TO_EUR[currency]
    except KeyError:
        raise ValueError(f"unsupported currency: {currency}") from None
    return quantize_amount(amount * rate)
'''
CURRENCY_V1 = CURRENCY.replace("ROUND_HALF_UP", "ROUND_HALF_EVEN").replace(", half-up (PRICING-371)", "")

DISCOUNTS = '''\
"""Tier discounts and regional caps.

Rates and caps follow docs/pricing.md. A discount is computed on the order
subtotal before tax, then capped per region.
"""

from decimal import Decimal

from pricing.currency import quantize_amount

TIER_RATES = {
    1: Decimal("0.00"),
    2: Decimal("0.02"),
    3: Decimal("0.04"),
    4: Decimal("0.06"),
}

# Maximum discount per order, in EUR. Regions not listed have no cap.
REGION_CAPS = {
    "EU": Decimal("55"),  # PRICING-388 (was 50)
}


def tier_discount(subtotal: Decimal, tier: int, region: str) -> Decimal:
    """Discount for an order subtotal in EUR, before tax."""
    if tier not in TIER_RATES:
        raise ValueError(f"unknown tier: {tier}")
    discount = subtotal * TIER_RATES[tier]
    cap = REGION_CAPS.get(region)
    if cap is not None:
        discount = min(discount, cap)
    return quantize_amount(discount)


def discounted_subtotal(order) -> Decimal:
    """Order subtotal after the tier discount, before tax."""
    customer = order.customer
    return order.subtotal - tier_discount(order.subtotal, customer.tier, customer.region)
'''
DISCOUNTS_V1 = DISCOUNTS.replace('Decimal("55"),  # PRICING-388 (was 50)', 'Decimal("50"),')

TAXES = '''\
"""VAT for orders shipped within the EU."""

from decimal import Decimal

from pricing.currency import quantize_amount

# Standard VAT rates by country (ISO 3166-1 alpha-2).
VAT_RATES = {
    "DE": Decimal("0.19"),
    "ES": Decimal("0.21"),
    "FR": Decimal("0.20"),
    "IE": Decimal("0.23"),
    "IT": Decimal("0.22"),
    "NL": Decimal("0.21"),
}


def vat_for(net: Decimal, country: str) -> Decimal:
    """VAT due on a net amount for a customer in `country`."""
    if country not in VAT_RATES:
        raise ValueError(f"no VAT rate for country: {country}")
    return quantize_amount(net * VAT_RATES[country])


def gross_total(net: Decimal, country: str) -> Decimal:
    """Net amount plus VAT."""
    return net + vat_for(net, country)
'''

FACTORIES = '''\
"""Small builders for customers and orders, shared by the tests."""

from dataclasses import dataclass
from decimal import Decimal


@dataclass
class Customer:
    id: str
    tier: int
    region: str
    country: str


@dataclass
class Order:
    id: str
    customer: Customer
    subtotal: Decimal
    currency: str = "EUR"


def eu_customer(tier: int = 3, country: str = "DE") -> Customer:
    return Customer(id="c-1001", tier=tier, region="EU", country=country)


def us_customer(tier: int = 3) -> Customer:
    return Customer(id="c-2001", tier=tier, region="US", country="US")


def order_with_subtotal(subtotal: Decimal, customer: Customer | None = None) -> Order:
    return Order(id="o-5001", customer=customer or eu_customer(), subtotal=subtotal)
'''

TEST_DISCOUNTS = '''\
from decimal import Decimal

import pytest

from pricing.discounts import discounted_subtotal, tier_discount
from tests.factories import eu_customer, order_with_subtotal, us_customer


def test_tier_1_has_no_discount():
    assert tier_discount(Decimal("980.00"), 1, "EU") == Decimal("0.00")


def test_tier_2_rate():
    assert tier_discount(Decimal("450.00"), 2, "EU") == Decimal("9.00")


def test_tier_3_rate_below_cap():
    assert tier_discount(Decimal("800.00"), 3, "EU") == Decimal("32.00")


def test_tier_3_eu_cap():
    order = order_with_subtotal(Decimal("1310.00"), eu_customer(tier=3))
    assert tier_discount(order.subtotal, 3, "EU") == Decimal("50.00")


def test_no_cap_outside_eu():
    assert tier_discount(Decimal("1310.00"), 3, "US") == Decimal("52.40")


def test_discounted_subtotal():
    order = order_with_subtotal(Decimal("250.00"), us_customer(tier=2))
    assert discounted_subtotal(order) == Decimal("245.00")


def test_unknown_tier_raises():
    with pytest.raises(ValueError, match="unknown tier"):
        tier_discount(Decimal("100.00"), 7, "EU")
'''

TEST_TAXES = '''\
from decimal import Decimal

import pytest

from pricing.taxes import gross_total, vat_for


@pytest.mark.parametrize(
    "country, expected",
    [("DE", Decimal("19.00")), ("ES", Decimal("21.00")), ("IE", Decimal("23.00"))],
)
def test_vat_rates(country, expected):
    assert vat_for(Decimal("100.00"), country) == expected


def test_vat_rounds_to_cent():
    assert vat_for(Decimal("19.99"), "DE") == Decimal("3.80")


def test_gross_total():
    assert gross_total(Decimal("52.40"), "FR") == Decimal("62.88")


def test_unknown_country_raises():
    with pytest.raises(ValueError, match="no VAT rate"):
        vat_for(Decimal("10.00"), "US")
'''

DOC_INTRO = """\
# Pricing rules

This document is the reference for the rules in `src/pricing`. The commercial
team owns the rules; the pricing team owns the code.

## Tier discounts

Customers have a tier from 1 to 4. The discount is a share of the order
subtotal, before tax:

| Tier | Discount |
| ---- | -------- |
| 1    | 0%       |
| 2    | 2%       |
| 3    | 4%       |
| 4    | 6%       |
"""
DOC_CAPS_V1 = """
## Regional caps

The discount on one order is capped at EUR 50 for EU customers. Other regions
have no cap.
"""
DOC_CAPS_TABLE = """
## Regional caps

The discount on one order is capped per region:

| Region | Cap per order | Since      | Ticket      |
| ------ | ------------- | ---------- | ----------- |
| EU     | EUR {cap}        | {since} | {ticket} |
| Other  | no cap        |            |             |

Example: a tier 3 EU order of EUR 1,310.00 gets 4% = EUR 52.40, {example}.
"""
DOC_CAPS_V2 = DOC_CAPS_TABLE.format(cap=50, since="2026-09-08", ticket="PRICING-342",
                                    example="capped to EUR 50.00")
DOC_CAPS = DOC_CAPS_TABLE.format(cap=55, since="2026-10-25", ticket="PRICING-388", example="under the cap")
DOC_VAT = """
## VAT

VAT is charged on the discounted subtotal at the standard rate of the
customer's country (`VAT_RATES` in `taxes.py`). Orders outside the EU are not
charged VAT by this service.
"""
DOC_ROUNDING = """
## Rounding

Amounts are rounded to the cent, half-up (PRICING-371). Conversion to EUR uses
the monthly rates in `currency.py`.
"""

CHANGELOG_ENTRIES = [
    ("1.3.0", "2026-10-25", "PRICING-388: EU tier discount cap raised from EUR 50 to EUR 55 per order."),
    ("1.2.1", "2026-10-07", "PRICING-371: currency conversion rounds half-up to the cent (was half-even)."),
    ("1.2.0", "2026-09-16", "VAT for EU countries (`pricing.taxes`)."),
    ("1.1.0", "2026-09-08", "Tier discounts, with an EU cap of EUR 50 per order."),
    ("1.0.0", "2026-09-01", "Currency conversion to EUR."),
]


def changelog(since: str) -> str:
    entries = [e for e in CHANGELOG_ENTRIES if e[0] <= since]
    return "# Changelog\n" + "".join(f"\n## {v} ({d})\n\n- {text}\n" for v, d, text in entries)


COMMITS = [
    (MARTA, "2026-09-01T10:12:44+02:00", "Initial pricing service skeleton", {
        ".gitignore": GITIGNORE, "pyproject.toml": PYPROJECT, "README.md": README_V1,
        "src/pricing/__init__.py": INIT.format("1.0.0"), "src/pricing/currency.py": CURRENCY_V1,
        "tests/__init__.py": "",
    }),
    (TOMAS, "2026-09-08T15:30:02+02:00", "Add tier discounts with an EU cap\n\nRefs: PRICING-342", {
        "src/pricing/discounts.py": DISCOUNTS_V1, "src/pricing/__init__.py": INIT.format("1.1.0"),
        "tests/factories.py": FACTORIES, "tests/test_discounts.py": TEST_DISCOUNTS,
        "docs/pricing.md": DOC_INTRO + DOC_CAPS_V1,
    }),
    (PRIYA, "2026-09-16T11:05:37+02:00", "Add VAT calculation for EU countries", {
        "src/pricing/taxes.py": TAXES, "src/pricing/__init__.py": INIT.format("1.2.0"),
        "tests/test_taxes.py": TEST_TAXES, "docs/pricing.md": DOC_INTRO + DOC_CAPS_V1 + DOC_VAT,
    }),
    (MARTA, "2026-09-23T09:48:19+02:00", "CI: run pytest on every push and merge request", {
        ".ci/pipeline.yml": PIPELINE, "README.md": README,
    }),
    (TOMAS, "2026-10-07T17:21:55+02:00", "PRICING-371: round currency conversion half-up", {
        "src/pricing/currency.py": CURRENCY, "src/pricing/__init__.py": INIT.format("1.2.1"),
        "CHANGELOG.md": changelog("1.2.1"),
    }),
    (PRIYA, "2026-10-15T12:02:41+02:00", "Docs: regional caps table and rounding rules", {
        "docs/pricing.md": DOC_INTRO + DOC_CAPS_V2 + DOC_VAT + DOC_ROUNDING,
    }),
    (MARTA, "2026-10-25T16:47:09+01:00",
     "PRICING-388: raise EU discount cap to 55\n\n"
     "Commercial approved a higher cap on EU tier discounts, effective with the\n"
     "next release.", {
        "src/pricing/discounts.py": DISCOUNTS, "src/pricing/__init__.py": INIT.format("1.3.0"),
        "docs/pricing.md": DOC_INTRO + DOC_CAPS + DOC_VAT + DOC_ROUNDING, "CHANGELOG.md": changelog("1.3.0"),
    }),
]


def git(*args, env=None):
    subprocess.run(["git", *args], cwd=ROOT, check=True, env={**os.environ, **(env or {})})


os.makedirs(ROOT)
git("init", "-q", "-b", "main")
for (name, email), date, message, files in COMMITS:
    for path, text in files.items():
        full = os.path.join(ROOT, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(text)
    git("add", "-A")
    git("commit", "-q", "-m", message, env={
        "GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email, "GIT_AUTHOR_DATE": date,
        "GIT_COMMITTER_NAME": name, "GIT_COMMITTER_EMAIL": email, "GIT_COMMITTER_DATE": date,
    })

# Make it look like a clone of origin/main.
git("remote", "add", "origin", ORIGIN)
git("update-ref", "refs/remotes/origin/main", "HEAD")
git("symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main")
git("config", "branch.main.remote", "origin")
git("config", "branch.main.merge", "refs/heads/main")
git("repack", "-adq")
git("prune-packed")
head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
for log in ("HEAD", "refs/heads/main", "refs/remotes/origin/HEAD"):
    path = os.path.join(ROOT, ".git", "logs", log)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{'0' * 40} {head} dev <dev@lk-devbox-14> {CHECKOUT_TIME} +0000\tclone: from {ORIGIN}\n")
for dirpath, dirnames, filenames in os.walk(ROOT):
    for name in dirnames + filenames:
        os.utime(os.path.join(dirpath, name), (CHECKOUT_TIME, CHECKOUT_TIME), follow_symlinks=False)
os.utime(ROOT, (CHECKOUT_TIME, CHECKOUT_TIME))
