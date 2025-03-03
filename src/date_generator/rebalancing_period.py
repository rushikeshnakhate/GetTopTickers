from enum import Enum


class RebalancingPeriod(Enum):
    DAILY = "daily"
    HALF_MONTHLY = "half_monthly"
    MONTHLY = "monthly"
    YEARLY = "yearly"
