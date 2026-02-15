import json
import ee
import functions_framework
from flask import jsonify, request

# -------------------------------
# Initialize Earth Engine
# -------------------------------
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()

# -------------------------------
# Helper: NDVI computation
# -------------------------------
def add_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

# -------------------------------
# Helper: NDVI status category
# -------------------------------
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
# Cloud Function Entry Point
# -------------------------------
@functions_framework.http
def field_ndvi(request):
    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({"error": "Invalid JSON body"}), 400

        lat = data.get("lat")
        lon = data.get("lon")
        boundary = data.get("boundary")

        if lat is None or lon is None:
            return jsonify({"error": "lat and lon are required"}), 400

        # -------------------------------
        # Geometry
        # -------------------------------
        if boundary:
            coords = [[p["lon"], p["lat"]] for p in boundary]
            geom = ee.Geometry.Polygon([coords])
        else:
            geom = ee.Geometry.Point([lon, lat]).buffer(100)

        # -------------------------------
        # Sentinel-2 Collection
        # -------------------------------
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geom)
            .filterDate("2024-01-01", "2024-12-31")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .map(add_ndvi)
        )

        if collection.size().getInfo() == 0:
            return jsonify({"error": "No satellite images found"}), 404

        # -------------------------------
        # Mean NDVI
        # -------------------------------
        ndvi_img = collection.select("NDVI").mean()

        stats = ndvi_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=10,
            maxPixels=1e9,
        )

        ndvi_value = stats.get("NDVI").getInfo()

        if ndvi_value is None:
            return jsonify({"error": "NDVI calculation failed"}), 500

        status = ndvi_status(ndvi_value)

        # -------------------------------
        # Latest Image Date
        # -------------------------------
        latest = ee.Image(collection.sort("system:time_start", False).first())
        date = ee.Date(latest.get("system:time_start")).format("YYYY-MM-dd").getInfo()

        # -------------------------------
        # Response
        # -------------------------------
        return jsonify({
            "ndvi_mean": round(ndvi_value, 3),
            "status": status,
            "date": date,
            "source": "Sentinel-2 (GEE)",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
