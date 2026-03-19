import pandas as pd
from sqlalchemy import create_engine

DB_URL = "postgresql://kishanseva_db_user:rQCe3oRP6FtrAVZ6349iWLJZfLLMRtBQ@dpg-d6u33jjuibrs73er47fg-a.singapore-postgres.render.com/kishanseva_db"

engine = create_engine(DB_URL)

def load_csv():

    df = pd.read_csv("market_data.csv")

    df.to_sql(
        "market_prices",
        engine,
        if_exists="append",
        index=False
    )

    print("Inserted:", len(df))


if __name__ == "__main__":
    load_csv()
