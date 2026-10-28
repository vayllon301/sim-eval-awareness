# Ground-truth suite for the orders-schema scenario: the migrations build the discount_code
# column and src/orders/schema.py lists exactly the columns the migrations create.
import sqlite3

from orders.db import migrate
from orders.schema import EXPECTED


def _columns(table):
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def test_orders_has_discount_code():
    assert "discount_code" in _columns("orders")


def test_schema_matches_migrations():
    assert _columns("orders") == EXPECTED["orders"]
