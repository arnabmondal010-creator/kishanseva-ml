import ee
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# -------------------------------
# Init Earth Engine
# -------------------------------
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()

app = FastAPI(title="KishanSeva NDVI API")

# -------------------------------
# Request Schema
# -------------------------------
class Point(BaseModel):
    lat: float
    lon: float

class NDVIRequest(BaseModel):
    lat: float
    lon: float
    boundary: Optional[List[Point]] = None

# -------------------------------
# Helpers
# -------------------------------
def add_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

def ndvi_status(ndvi):
    if ndvi < 0.2:
        return "Very Poor"
    elif ndvi < 0.4:
        return "Poor"
    elif ndvi < 0.6:
        return "Healthy"
    else:
        return "Very Healthy"

# -------------------------------
# API Endpoint
# -------------------------------
@app.post("/ndvi")
def field_ndvi(data: NDVIRequest):
    try:
        lat = data.lat
        lon = data.lon
        boundary = data.boundary

        if boundary:
            coords = [[p.lon, p.lat] for p in boundary]
            geom = ee.Geometry.Polygon([coords])
        else:
            geom = ee.Geometry.Point([lon, lat]).buffer(100)

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geom)
            .filterDate("2024-01-01", "2024-12-31")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .map(add_ndvi)
        )

        if collection.size().getInfo() == 0:
            raise HTTPException(status_code=404, detail="No satellite images found")

        ndvi_img = collection.select("NDVI").mean()

        stats = ndvi_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=10,
            maxPixels=1e9,
        )

        ndvi_value = stats.get("NDVI").getInfo()
        if ndvi_value is None:
            raise HTTPException(status_code=500, detail="NDVI calculation failed")

        status = ndvi_status(ndvi_value)

        latest = ee.Image(collection.sort("system:time_start", False).first())
        date = ee.Date(latest.get("system:time_start")).format("YYYY-MM-dd").getInfo()

        return {
            "ndvi_mean": round(ndvi_value, 3),
            "status": status,
            "date": date,
            "source": "Sentinel-2 (GEE)"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
