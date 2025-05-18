from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class InventoryMethod:
    method: str                        # 'sQ' (re-order point) or 'RS'
    s: Optional[int] = None            # for 'sQ'
    Q: Optional[int] = None            # for 'sQ'
    R: Optional[int] = None            # for 'RS'
    S: Optional[int] = None            # for 'RS'
    initial_inventory: int = 0
    lead_time_low: int = 2             # inclusive  lower bound
    lead_time_high: int = 5            # inclusive  upper bound
