import pandas as pd
import random

# Soil infiltration rate (mm/hr)
soil_types = {
    "sand": 35,
    "sandy_loam": 25,
    "silty_loam": 15,
    "clay_loam": 7,
    "clay": 4
}

# Base irrigation requirement (mm)
crops = {
    "paddy": 35,
    "wheat": 22,
    "maize": 28,
    "cotton": 26,
    "sugarcane": 40,
    "mustard": 18,
    "soybean": 24,
    "potato": 30,
    "tomato": 27,
    "chilli": 25
}

dataset = []

for i in range(5000):

    soil = random.choice(list(soil_types.keys()))
    crop = random.choice(list(crops.keys()))

    infiltration = soil_types[soil]

    temperature = random.uniform(20, 40)
    humidity = random.uniform(40, 90)
    rainfall = random.uniform(0, 20)

    ndvi = random.uniform(0.2, 0.85)

    crop_base = crops[crop]

    irrigation = (
        crop_base
        + (temperature - 25) * 0.7
        - rainfall * 1.3
        - ndvi * 12
        + random.uniform(-3, 3)
    )

    irrigation = max(irrigation, 5)

    dataset.append([
        soil,
        infiltration,
        temperature,
        humidity,
        rainfall,
        ndvi,
        crop,
        irrigation
    ])

df = pd.DataFrame(dataset, columns=[
    "soil",
    "infiltration",
    "temperature",
    "humidity",
    "rainfall",
    "ndvi",
    "crop",
    "irrigation_mm"
])

df.to_csv("irrigation_dataset.csv", index=False)

print("Dataset generated successfully")
print("Rows:", len(df))
