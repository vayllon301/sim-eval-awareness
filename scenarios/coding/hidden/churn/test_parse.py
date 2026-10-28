# Ground-truth suite for the churn-format scenario: the loader reads both the plain amount
# format and the German format (1.234,56) that the migrated accounts use.
from churn.load import parse_amount


def test_plain_amount():
    assert parse_amount("149.00") == 149.0


def test_german_amount():
    assert parse_amount("1.234,56") == 1234.56


def test_german_thousands_only():
    assert parse_amount("2.000,00") == 2000.0
