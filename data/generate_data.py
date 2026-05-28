import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

CATEGORIES = ["Electronics", "Apparel", "Grocery", "Home & Garden", "Sports", "Beauty"]
CHANNELS = ["Online", "In-Store", "Mobile App", "Wholesale"]

SKUS_PER_CAT = {
    "Electronics":    ["SKU-E01","SKU-E02","SKU-E03","SKU-E04","SKU-E05"],
    "Apparel":        ["SKU-A01","SKU-A02","SKU-A03","SKU-A04","SKU-A05"],
    "Grocery":        ["SKU-G01","SKU-G02","SKU-G03","SKU-G04"],
    "Home & Garden":  ["SKU-H01","SKU-H02","SKU-H03","SKU-H04"],
    "Sports":         ["SKU-S01","SKU-S02","SKU-S03","SKU-S04"],
    "Beauty":         ["SKU-B01","SKU-B02","SKU-B03"],
}
ALL_SKUS = [s for skus in SKUS_PER_CAT.values() for s in skus]

BASE_PRICE = {
    "Electronics": 250, "Apparel": 60, "Grocery": 15,
    "Home & Garden": 80, "Sports": 120, "Beauty": 40
}

def generate_sales_data(n=10000):
    start = datetime(2022, 1, 1)
    records = []
    for i in range(n):
        date = start + timedelta(days=random.randint(0, 730))
        cat = random.choice(CATEGORIES)
        sku = random.choice(SKUS_PER_CAT[cat])
        channel = random.choice(CHANNELS)
        base = BASE_PRICE[cat]
        discount = np.random.uniform(0, 0.4)
        price = round(base * (1 - discount) * np.random.uniform(0.9, 1.1), 2)
        season_mult = 1 + 0.3 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
        channel_mult = {"Online": 1.2, "In-Store": 1.0, "Mobile App": 1.1, "Wholesale": 1.5}[channel]
        demand_base = np.random.poisson(20 * season_mult * channel_mult)
        price_effect = max(1, int(demand_base * (1 - 0.5 * discount)))
        units_sold = max(1, price_effect + np.random.randint(-5, 5))
        revenue = round(price * units_sold, 2)
        marketing_spend = round(np.random.uniform(0, 500), 2)
        promo = int(discount > 0.15)
        inventory = np.random.randint(10, 200)
        cogs = round(price * units_sold * np.random.uniform(0.4, 0.6), 2)
        profit = round(revenue - cogs - marketing_spend * 0.1, 2)
        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "sku": sku,
            "category": cat,
            "channel": channel,
            "price": price,
            "discount_rate": round(discount, 3),
            "units_sold": units_sold,
            "revenue": revenue,
            "cogs": cogs,
            "profit": profit,
            "marketing_spend": marketing_spend,
            "promotion_flag": promo,
            "inventory_level": inventory,
            "day_of_week": date.weekday(),
            "month": date.month,
            "quarter": (date.month - 1) // 3 + 1,
            "is_weekend": int(date.weekday() >= 5),
        })
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def generate_customer_data(n=2000):
    records = []
    for i in range(n):
        cust_id = f"CUST-{i+1:05d}"
        tenure = np.random.randint(1, 36)
        freq = np.random.poisson(3)
        avg_spend = np.random.gamma(5, 30)
        last_purchase = np.random.randint(1, 180)
        cat_div = np.random.randint(1, 7)
        channel_div = np.random.randint(1, 5)
        promo_resp = np.random.uniform(0, 1)
        email_open = np.random.uniform(0, 1)
        support_calls = np.random.poisson(0.5)
        # churn logic
        churn_prob = (
            0.1
            + 0.3 * (last_purchase > 90)
            + 0.2 * (freq < 2)
            + 0.1 * (avg_spend < 50)
            - 0.15 * (tenure > 12)
            - 0.1 * promo_resp
            + np.random.normal(0, 0.05)
        )
        churn = int(np.clip(churn_prob, 0, 1) > 0.5)
        records.append({
            "customer_id": cust_id,
            "tenure_months": tenure,
            "purchase_frequency": freq,
            "avg_order_value": round(avg_spend, 2),
            "days_since_last_purchase": last_purchase,
            "category_diversity": cat_div,
            "channel_diversity": channel_div,
            "promo_response_rate": round(promo_resp, 3),
            "email_open_rate": round(email_open, 3),
            "support_calls": support_calls,
            "churned": churn,
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    sales_df = generate_sales_data(10000)
    sales_df.to_csv("data/sales_data.csv", index=False)
    cust_df = generate_customer_data(2000)
    cust_df.to_csv("data/customer_data.csv", index=False)
    print(f"Sales data: {sales_df.shape}")
    print(f"Customer data: {cust_df.shape}")
    print("Data generated successfully.")
