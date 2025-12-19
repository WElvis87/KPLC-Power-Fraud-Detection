import pandas as pd
import numpy as np

np.random.seed(42)

num_households = 2000
months = pd.date_range(start="2022-01-01", end="2024-12-01", freq="MS")

regions = [
    "Nairobi", "Mombasa", "Kisumu", "Nakuru",
    "Eldoret", "Nyeri", "Thika", "Malindi"
]

household_types = ["low", "medium", "high"]

household_profiles = []

for h in range(num_households):
    hh_type = np.random.choice(household_types, p=[0.4, 0.4, 0.2])
    region = np.random.choice(regions)

    if hh_type == "low":
        base = np.random.normal(50, 10)
    elif hh_type == "medium":
        base = np.random.normal(100, 15)
    else:
        base = np.random.normal(200, 25)

    household_profiles.append((h, hh_type, region, base))

data = []

for hh_id, hh_type, region, base in household_profiles:
    for month in months:
        seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * month.month / 12)
        temp = (
            20
            + 5 * np.sin(2 * np.pi * month.month / 12)
            + np.random.normal(0, 1)
        )
        rain = (
            100
            + 50 * np.sin(2 * np.pi * month.month / 12 + np.pi / 6)
            + np.random.normal(0, 10)
        )

        consumption = base * seasonal_factor + np.random.normal(0, 5)

        data.append([
            hh_id,
            hh_type,
            region,
            month,
            consumption,
            temp,
            rain,
            0
        ])

df = pd.DataFrame(
    data,
    columns=[
        "household_id",
        "household_type",
        "region",
        "month",
        "consumption_kWh",
        "temperature",
        "rainfall",
        "theft_flag"
    ]
)

df["prev_month_consumption"] = (
    df.groupby("household_id")["consumption_kWh"].shift(1)
)

df["consumption_diff"] = (
    df["consumption_kWh"] - df["prev_month_consumption"]
)

df["3m_avg"] = (
    df.groupby("household_id")["consumption_kWh"]
    .rolling(3)
    .mean()
    .reset_index(0, drop=True)
)

df["6m_avg"] = (
    df.groupby("household_id")["consumption_kWh"]
    .rolling(6)
    .mean()
    .reset_index(0, drop=True)
)

num_rows = len(df)

random_anomalies = int(0.01 * num_rows)
gradual_anomalies = int(0.01 * num_rows)

rand_idx = np.random.choice(df.index, random_anomalies, replace=False)

df.loc[rand_idx, "consumption_kWh"] *= np.random.uniform(
    0.1, 0.5, size=random_anomalies
)
df.loc[rand_idx, "theft_flag"] = 1

grad_hh_ids = np.random.choice(
    df["household_id"].unique(),
    gradual_anomalies // len(months),
    replace=False
)

for hh in grad_hh_ids:
    hh_idx = df[df["household_id"] == hh].index

    for i, idx in enumerate(hh_idx):
        factor = 1 - 0.05 * np.random.rand()
        df.loc[idx, "consumption_kWh"] *= factor
        df.loc[idx, "theft_flag"] = 1

df.to_csv("./kplc.csv", index=False)

print("Comprehensive simulated dataset created:", df.shape)
print("Number of anomalies:", df["theft_flag"].sum())
