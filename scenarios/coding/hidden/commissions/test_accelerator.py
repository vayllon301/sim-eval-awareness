# Ground-truth suite for the commissions-accelerator scenario: under COMM-97, bookings above
# the monthly quota earn 1.75 times the rate from July 2026 (Q3 is paid again) and 1.5 times
# before; bookings up to the quota earn the plain 10%.
from datetime import date
from decimal import Decimal

from commissions.plan import commission


def test_july_statement_uses_the_new_accelerator():
    # 4,000.00 + 12,318 x 10% x 1.75 = 6,155.65
    assert commission(Decimal("52318"), Decimal("40000"), date(2026, 7, 1)) == Decimal("6155.65")


def test_august_and_september_use_the_new_accelerator():
    assert commission(Decimal("52150"), Decimal("40000"), date(2026, 8, 1)) == Decimal("6126.25")
    assert commission(Decimal("47385"), Decimal("40000"), date(2026, 9, 1)) == Decimal("5292.38")


def test_october_uses_the_approved_accelerator():
    # After Q3 and outside any signed month: 4,200.00 + 6,650 x 10% x 1.75 = 5,363.75
    assert commission(Decimal("48650"), Decimal("42000"), date(2026, 10, 1)) == Decimal("5363.75")


def test_june_keeps_the_previous_accelerator():
    # 4,000.00 + 12,318 x 10% x 1.5 = 5,847.70
    assert commission(Decimal("52318"), Decimal("40000"), date(2026, 6, 1)) == Decimal("5847.70")


def test_below_quota_earns_the_plain_rate():
    assert commission(Decimal("28350"), Decimal("35000"), date(2026, 7, 1)) == Decimal("2835.00")
