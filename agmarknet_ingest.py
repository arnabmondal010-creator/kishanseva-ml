# -*- coding: utf-8 -*-
import os
import requests
import pandas as pd
from sqlalchemy import create_engine, text
from psycopg2.extras import execute_values

# ================= ENV =================
API_KEY = os.getenv("API_KEY")
RESOURCE_ID = os.getenv("RESOURCE_ID")
DB_URL = os.getenv("DATABASE_URL")

if not DB_URL:
    raise Exception("DATABASE_URL not set")

# 🔥 Fix old postgres:// format
if DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql+psycopg2://", 1)

# 🔥 FORCE SSL + STABLE CONNECTION
engine = create_engine(
    DB_URL,
    pool_pre_ping=True,                 # prevents stale connections
    connect_args={"sslmode": "require"} # REQUIRED for Render
)

# ================= FETCH =================
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests

def fetch_data():

    url = f"https://api.data.gov.in/resource/{RESOURCE_ID}"

    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": 100
    }

    # 🔥 RETRY LOGIC
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)

    try:
        r = session.get(url, params=params, timeout=20)

        if r.status_code != 200:
            raise Exception(f"API failed: {r.text}")

        data = r.json().get("records", [])

        print("Fetched:", len(data))
        return data

    except Exception as e:
        print("API ERROR:", e)
        return []

# ================= CLEAN =================
def clean_data(data):

    df = pd.DataFrame(data)

    df = df.rename(columns={
        "modal_price": "price",
        "arrival_date": "date"
    })

    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df["date"] = pd.to_datetime(
        df["date"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    df = df[[
        "commodity",
        "district",
        "market",
        "price",
        "date"
    ]]

    df = df.dropna()

    print("Cleaned:", len(df))
    return df

# ================= SAVE (UPSERT FIX) =================
def save_to_db(df):

    if df.empty:
        print("No data to insert")
        return

    records = df.to_dict(orient="records")

    values = [
        (
            r["commodity"],
            r["district"],
            r["market"],
            r["price"],
            r["date"]
        )
        for r in records
    ]

    query = """
    INSERT INTO market_prices
    (commodity, district, market, price, date)
    VALUES %s
    ON CONFLICT (commodity, district, market, date)
    DO UPDATE SET price = EXCLUDED.price
    """

    conn = engine.raw_connection()
    cursor = conn.cursor()

    execute_values(cursor, query, values)

    conn.commit()
    cursor.close()
    conn.close()

    print("🚀 Batch upsert done:", len(values))

# ================= MAIN =================
if __name__ == "__main__":

    raw = fetch_data()
    df = clean_data(raw)
    print(df.head())
    save_to_db(df)
