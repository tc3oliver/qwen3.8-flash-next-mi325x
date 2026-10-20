"""Tiny inventory module."""
from dataclasses import dataclass, field


@dataclass
class Item:
    sku: str
    price_cents: int
    qty: int = 0


@dataclass
class Inventory:
    items: dict = field(default_factory=dict)

    def add(self, sku: str, price_cents: int, qty: int = 1) -> None:
        if sku in self.items:
            self.items[sku].qty = qty
        else:
            self.items[sku] = Item(sku, price_cents, qty)

    def remove(self, sku: str, qty: int = 1) -> None:
        item = self.items[sku]
        if qty > item.qty:
            raise ValueError("not enough stock")
        item.qty -= qty
        if item.qty < 0:
            del self.items[sku]

    def total_value_cents(self) -> int:
        return sum(i.price_cents for i in self.items.values())

    def most_valuable(self, n: int = 1) -> list:
        return sorted(self.items.values(), key=lambda i: i.price_cents * i.qty)[:n]
