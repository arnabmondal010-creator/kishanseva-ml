import requests
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

API_KEY = os.getenv("API_KEY")
RESOURCE_ID = os.getenv("RESOURCE_ID")
DB_URL = os.getenv("DATABASE_URL")

engine = create_engine(DB_URL)

# ================= FETCH =================
def fetch_data():

    url = f"https://api.data.gov.in/resource/{RESOURCE_ID}"

    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": 100
    }

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    r = requests.get(url, params=params, headers=headers)

    data = r.json().get("records", [])

    print("Fetched:", len(data))

    return data


# ================= CLEAN =================
def clean_data(data):

    df = pd.DataFrame(data)

    df = df.rename(columns={
        "commodity": "commodity",
        "district": "district",
        "market": "market",
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


# ================= SAVE =================
def save_to_db(df):

    if df.empty:
        print("No data to insert")
        return

    df.to_sql(
        "market_prices",
        engine,
        if_exists="append",
        index=False
    )

    print("Inserted:", len(df))


# ================= MAIN =================
if __name__ == "__main__":

    raw = fetch_data()

    df = clean_data(raw)

    print(df.head())

    save_to_db(df)
