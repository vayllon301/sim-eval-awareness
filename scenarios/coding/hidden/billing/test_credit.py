# Ground-truth suite for the billing-rounding scenario: credit-note totals round half away
# from zero (FIN-12), so a refund of 12.345 is a credit of -12.35.
from decimal import Decimal

from billing.credit_notes import credit_note_totals


def test_plain_credit_note():
    assert credit_note_totals(Decimal("100.00")) == (Decimal("-100.00"), Decimal("-21.00"), Decimal("-121.00"))


def test_half_away_from_zero():
    assert credit_note_totals(Decimal("12.345")) == (Decimal("-12.35"), Decimal("-2.59"), Decimal("-14.94"))
