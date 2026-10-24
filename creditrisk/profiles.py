"""Illustrative applicant profiles (invented, not from the dataset) for the dashboard and smoke tests."""

PRESETS = {
    "Reliable payer": dict(limit_bal=200000, pay=[-1, -1, -1, -1, -1, -1],
                           bill=[12000, 9800, 15200, 8700, 11000, 9900], paid=[9800, 15200, 8700, 11000, 9900, 7000]),
    "Revolving, on time": dict(limit_bal=80000, pay=[0, 0, 0, 0, 0, 0],
                               bill=[52000, 50500, 49800, 48000, 46500, 45000],
                               paid=[2100, 2000, 2000, 1900, 1800, 1800]),
    "Recently late": dict(limit_bal=50000, pay=[2, 2, 0, 0, 0, 0],
                          bill=[48500, 47200, 45100, 30200, 29000, 28100], paid=[0, 1500, 1800, 1200, 1100, 1000]),
    "Maxed out, missed payments": dict(limit_bal=30000, pay=[3, 2, 2, 2, 0, 0],
                                       bill=[31500, 30800, 30100, 29500, 27000, 25000],
                                       paid=[0, 0, 0, 500, 1000, 1000]),
}


def to_applicant(preset: dict) -> dict:
    """Preset -> API / raw-schema record (index 1 = most recent month)."""
    row = {"limit_bal": preset["limit_bal"]}
    for m in range(6):
        row[f"pay_{m + 1}"] = preset["pay"][m]
        row[f"bill_amt{m + 1}"] = preset["bill"][m]
        row[f"pay_amt{m + 1}"] = preset["paid"][m]
    return row
