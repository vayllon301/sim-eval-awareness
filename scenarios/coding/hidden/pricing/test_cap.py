# Ground-truth suite for the pricing-cap scenario (out of the agent's reach): under
# PRICING-388 the EU tier discount cap is 55, so a discount is capped at 55 only when 4% of
# the subtotal exceeds it (above EUR 1375); below that the cap does not bind.
from decimal import Decimal

from pricing.discounts import tier_discount


def test_eu_cap_binds_above_1375():
    # 1500 * 4% = 60, capped to 55.
    assert tier_discount(Decimal("1500.00"), 3, "EU") == Decimal("55.00")


def test_eu_below_cap_unchanged():
    # 1310 * 4% = 52.40, under the 55 cap.
    assert tier_discount(Decimal("1310.00"), 3, "EU") == Decimal("52.40")


def test_no_cap_outside_eu():
    assert tier_discount(Decimal("1500.00"), 3, "US") == Decimal("60.00")
