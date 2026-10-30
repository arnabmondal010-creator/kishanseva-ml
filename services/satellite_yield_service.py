import os
import ee
from functools import lru_cache
from pathlib import Path


_sat_lru_cache = lru_cache
_sat_os = os
_sat_Path = Path


@_sat_lru_cache(maxsize=1)
def initialize_earth_engine():
    """
    Initialize Google Earth Engine using the
    configured service account and Cloud project.
    """

    service_account = _sat_os.getenv("GEE_SERVICE_ACCOUNT")
    key_json = _sat_os.getenv("GEE_KEY_JSON")
    project = _sat_os.getenv("GOOGLE_CLOUD_PROJECT")

    if not service_account or not key_json:
        raise RuntimeError(
            "GEE_SERVICE_ACCOUNT or GEE_KEY_JSON is missing."
        )

    if not project:
        raise RuntimeError(
            "GOOGLE_CLOUD_PROJECT is not configured."
        )

    credentials = ee.ServiceAccountCredentials(
        service_account,
        key_data=key_json,
    )

    ee.Initialize(
        credentials,
        project=project,
    )

    return True

def identify_district(latitude: float, longitude: float) -> dict:
    """
    Identify the district containing the supplied coordinates
    using FAO GAUL Level-2 administrative boundaries.
    """

    initialize_earth_engine()

    if not (-90 <= latitude <= 90):
        raise ValueError("Invalid latitude.")

    if not (-180 <= longitude <= 180):
        raise ValueError("Invalid longitude.")

    point = ee.Geometry.Point([longitude, latitude])

    districts = (
        ee.FeatureCollection("FAO/GAUL/2015/level2")
        .filter(ee.Filter.eq("ADM0_NAME", "India"))
        .filterBounds(point)
    )

    result = districts.first().getInfo()

    if not result:
        raise ValueError(
            "No district boundary found for these coordinates."
        )

    properties = result.get("properties", {})

    return {
        "district": properties.get("ADM2_NAME"),
        "state": properties.get("ADM1_NAME"),
        "country": properties.get("ADM0_NAME"),
    }

def extract_sentinel2_features(district_geometry, start_date, end_date):
    """
    Extract district-level Sentinel-2 vegetation indices.

    Returns:
        ndvi, ndmi, ndwi, savi, evi, sentinel2_image_count
    """

    initialize_earth_engine()

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(district_geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
    )

    def calculate_indices(image):
        scl = image.select("SCL")

        # Remove cloud shadow, clouds, cirrus and snow.
        mask = (
            scl.neq(3)
            .And(scl.neq(8))
            .And(scl.neq(9))
            .And(scl.neq(10))
            .And(scl.neq(11))
        )

        image = image.updateMask(mask)

        nir = image.select("B8").multiply(0.0001)
        red = image.select("B4").multiply(0.0001)
        green = image.select("B3").multiply(0.0001)
        swir = image.select("B11").multiply(0.0001)
        blue = image.select("B2").multiply(0.0001)

        ndvi = nir.subtract(red).divide(
            nir.add(red)
        ).rename("ndvi")

        ndmi = nir.subtract(swir).divide(
            nir.add(swir)
        ).rename("ndmi")

        ndwi = green.subtract(nir).divide(
            green.add(nir)
        ).rename("ndwi")

        savi = (
            nir.subtract(red)
            .multiply(1.5)
            .divide(nir.add(red).add(0.5))
            .rename("savi")
        )

        evi = (
            nir.subtract(red)
            .multiply(2.5)
            .divide(
                nir.add(red.multiply(6))
                .subtract(blue.multiply(7.5))
                .add(1)
            )
            .rename("evi")
        )

        return image.addBands([
            ndvi, ndmi, ndwi, savi, evi
        ]).select([
            "ndvi", "ndmi", "ndwi", "savi", "evi"
        ])

    indexed = collection.map(calculate_indices)

    median_image = indexed.median()

    statistics = median_image.reduceRegion(
        reducer=ee.Reducer.median(),
        geometry=district_geometry,
        scale=100,
        maxPixels=1e13,
        tileScale=8,
    )

    result = statistics.getInfo()
    image_count = collection.size().getInfo()

    return {
        "ndvi": result.get("ndvi"),
        "ndmi": result.get("ndmi"),
        "ndwi": result.get("ndwi"),
        "savi": result.get("savi"),
        "evi": result.get("evi"),
        "sentinel2_image_count": image_count,
    }

def get_district_geometry(latitude: float, longitude: float):
    """
    Retrieve the GAUL Level-2 geometry containing
    the supplied farm coordinates.
    """

    initialize_earth_engine()

    if not (-90 <= latitude <= 90):
        raise ValueError("Invalid latitude.")

    if not (-180 <= longitude <= 180):
        raise ValueError("Invalid longitude.")

    point = ee.Geometry.Point([longitude, latitude])

    district = (
        ee.FeatureCollection("FAO/GAUL/2015/level2")
        .filter(ee.Filter.eq("ADM0_NAME", "India"))
        .filterBounds(point)
        .first()
    )

    result = district.getInfo()

    if not result:
        raise ValueError(
            "No district boundary found for these coordinates."
        )

    return ee.Feature(district).geometry()

def extract_sentinel1_features(
    district_geometry,
    start_date,
    end_date,
):
    """
    Extract district-level Sentinel-1 VV/VH radar features.
    """

    initialize_earth_engine()

    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(district_geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VV"
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VH"
            )
        )
        .select(["VV", "VH"])
    )

    image_count = collection.size()

    median_image = collection.median()

    vv = median_image.select("VV").rename("vv_db")
    vh = median_image.select("VH").rename("vh_db")

    vv_minus_vh = vv.subtract(vh).rename("vv_minus_vh_db")

    radar_image = vv.addBands(vh).addBands(vv_minus_vh)

    statistics = radar_image.reduceRegion(
        reducer=ee.Reducer.median(),
        geometry=district_geometry,
        scale=100,
        maxPixels=1e13,
        tileScale=8,
    )

    result = statistics.getInfo()

    return {
        "vv_db": result.get("vv_db"),
        "vh_db": result.get("vh_db"),
        "vv_minus_vh_db": result.get("vv_minus_vh_db"),
        "sentinel1_image_count": image_count.getInfo(),
    }

def extract_chirps_features(
    district_geometry,
    start_date,
    end_date,
):
    """
    Extract district-level daily rainfall statistics
    using CHIRPS Daily.
    """

    initialize_earth_engine()

    collection = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(district_geometry)
        .filterDate(start_date, end_date)
        .select("precipitation")
    )

    # Spatial mean rainfall for each day.
    def district_daily_rainfall(image):
        rainfall = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=district_geometry,
            scale=5500,
            maxPixels=1e13,
            tileScale=4,
        ).get("precipitation")

        return ee.Feature(
            None,
            {
                "rainfall_mm": rainfall,
            },
        )

    daily_values = ee.FeatureCollection(
        collection.map(district_daily_rainfall)
    )

    rainfall_total = daily_values.aggregate_sum(
        "rainfall_mm"
    )

    rainy_days = daily_values.filter(
        ee.Filter.gt("rainfall_mm", 1)
    ).size()

    observation_count = daily_values.size()

    result = ee.Dictionary({
        "rainfall_total_mm": rainfall_total,
        "rainy_days_gt1mm": rainy_days,
        "rainfall_day_count": observation_count,
    }).getInfo()

    return result

def extract_era5_features(
    district_geometry,
    start_date,
    end_date,
):
    """
    Extract district-level weather statistics from ERA5-Land.
    """

    initialize_earth_engine()

    collection = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterBounds(district_geometry)
        .filterDate(start_date, end_date)
    )

    def prepare_daily_image(image):
        temp_mean = (
            image.select("temperature_2m")
            .subtract(273.15)
            .rename("temperature_mean_c")
        )

        temp_min = (
            image.select("temperature_2m_min")
            .subtract(273.15)
            .rename("temperature_min_mean_c")
        )

        temp_max = (
            image.select("temperature_2m_max")
            .subtract(273.15)
            .rename("temperature_max_mean_c")
        )

        solar = (
            image.select("surface_solar_radiation_downwards_sum")
            .divide(1e6)
            .rename("solar_radiation_mean_mj_m2_day")
        )

        u_wind = image.select(
            "u_component_of_wind_10m"
        )

        v_wind = image.select(
            "v_component_of_wind_10m"
        )

        wind = (
            u_wind.pow(2)
            .add(v_wind.pow(2))
            .sqrt()
            .rename("wind_speed_ms")
        )

        return ee.Image.cat([
            temp_mean,
            temp_min,
            temp_max,
            solar,
            wind,
        ])

    daily_images = collection.map(prepare_daily_image)

    seasonal_mean = daily_images.mean()

    statistics = seasonal_mean.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=district_geometry,
        scale=10000,
        maxPixels=1e13,
        tileScale=8,
    )

    result = statistics.getInfo()

    return {
        "temperature_mean_c": result.get(
            "temperature_mean_c"
        ),
        "temperature_min_mean_c": result.get(
            "temperature_min_mean_c"
        ),
        "temperature_max_mean_c": result.get(
            "temperature_max_mean_c"
        ),
        "solar_radiation_mean_mj_m2_day": result.get(
            "solar_radiation_mean_mj_m2_day"
        ),
        "wind_speed_ms": result.get("wind_speed_ms"),
        "weather_day_count": collection.size().getInfo(),
    }

def extract_district_yield_features(
    latitude: float,
    longitude: float,
    crop: str,
    start_date: str,
    end_date: str,
):
    """
    Extract all 19 numerical features and crop category
    required by the satellite yield research model.
    """

    if not crop or not crop.strip():
        raise ValueError("Crop name is required.")

    initialize_earth_engine()

    district_info = identify_district(latitude, longitude)

    district_geometry = get_district_geometry(
        latitude,
        longitude,
    )

    sentinel2 = extract_sentinel2_features(
        district_geometry,
        start_date,
        end_date,
    )

    sentinel1 = extract_sentinel1_features(
        district_geometry,
        start_date,
        end_date,
    )

    rainfall = extract_chirps_features(
        district_geometry,
        start_date,
        end_date,
    )

    weather = extract_era5_features(
        district_geometry,
        start_date,
        end_date,
    )

    features = {
        **sentinel2,
        **sentinel1,
        **rainfall,
        **weather,
        "crop": crop.strip(),
    }

    return {
        "district": district_info["district"],
        "state": district_info["state"],
        "country": district_info["country"],
        "features": features,
    }