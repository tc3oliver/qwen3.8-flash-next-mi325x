import pytest
from inventory.stock import Inventory


def test_add_accumulates():
    inv = Inventory(); inv.add("a", 100, 2); inv.add("a", 100, 3)
    assert inv.items["a"].qty == 5


def test_remove_drops_empty():
    inv = Inventory(); inv.add("a", 100, 2); inv.remove("a", 2)
    assert "a" not in inv.items


def test_remove_too_many():
    inv = Inventory(); inv.add("a", 100, 1)
    with pytest.raises(ValueError):
        inv.remove("a", 2)


def test_total_value():
    inv = Inventory(); inv.add("a", 100, 2); inv.add("b", 250, 1)
    assert inv.total_value_cents() == 450


def test_most_valuable():
    inv = Inventory(); inv.add("a", 100, 2); inv.add("b", 250, 1); inv.add("c", 10, 1)
    assert [i.sku for i in inv.most_valuable(2)] == ["b", "a"]
