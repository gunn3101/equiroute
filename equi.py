from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import httpx
import h3
import pandas as pd
import numpy as np
from shapely.geometry import shape, Point, MultiPolygon, Polygon
from shapely.ops import unary_union

app = FastAPI(title="EquiRoute")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── OVERPASS CONFIG ───────────────────────────────────────────────────────────
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

RESOURCE_TAGS = {
    "medical": [
        '["amenity"="hospital"]',
        '["amenity"="clinic"]',
        '["amenity"="doctors"]',
        '["amenity"="pharmacy"]',
    ],
    "education": [
        '["amenity"="school"]',
        '["amenity"="university"]',
        '["amenity"="college"]',
    ],
    "evacuation": [
        '["amenity"="shelter"]',
        '["amenity"="community_centre"]',
        '["amenity"="place_of_worship"]',
    ],
}

# ── HTML ROUTES ───────────────────────────────────────────────────────────────
@app.get("/")
def root(): return FileResponse("intro.html")

@app.get("/index.html")
def dashboard(): return FileResponse("index.html")

@app.get("/feedback.html")
def feedback(): return FileResponse("feedback.html")


# ── STEP 1: GET CITY BOUNDARY FROM NOMINATIM ─────────────────────────────────
async def get_boundary(city: str) -> Polygon | MultiPolygon | None:
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(NOMINATIM_URL, params={
            "q": city,
            "format": "json",
            "limit": 5,
            "polygon_geojson": 1,
        }, headers={"User-Agent": "EquiRoute/1.0"})

    results = r.json()
    print(f"[Boundary] Nominatim returned {len(results)} results for {city!r}")

    for res in results:
        geojson = res.get("geojson", {})
        geom_type = geojson.get("type", "")
        if geom_type in ("Polygon", "MultiPolygon"):
            boundary = shape(geojson)
            print(f"[Boundary] Found {geom_type}, area={boundary.area:.4f}, bounds={tuple(round(b,3) for b in boundary.bounds)}")
            return boundary

    print("[Boundary] No polygon found in Nominatim results")
    return None


# ── STEP 2: FETCH FACILITIES VIA OVERPASS ────────────────────────────────────
async def fetch_facilities(city: str, resource: str, bbox: tuple) -> pd.DataFrame:
    """bbox = (south, west, north, east)"""
    s, w, n, e = bbox
    tags = RESOURCE_TAGS.get(resource, RESOURCE_TAGS["medical"])

    # Build Overpass query — fetch nodes AND way centroids
    bbox_str = f"{s},{w},{n},{e}"
    parts = []
    for tag in tags:
        parts.append(f'node{tag}({bbox_str});')
        parts.append(f'way{tag}({bbox_str});')

    query = f"""
[out:json][timeout:60];
(
  {''.join(parts)}
);
out center;
"""
    print(f"[Overpass] Querying {resource} in bbox {bbox_str}")

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(OVERPASS_URL, data={"data": query})

    elements = r.json().get("elements", [])
    print(f"[Overpass] Got {len(elements)} raw elements")

    rows = []
    for el in elements:
        if el["type"] == "node":
            lat, lon = el.get("lat"), el.get("lon")
        elif el["type"] == "way" and "center" in el:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
        else:
            continue
        rows.append({
            "lat":  lat,
            "lon":  lon,
            "name": el.get("tags", {}).get("name", "Facility")
        })

    return pd.DataFrame(rows)


# ── STEP 3: CLIP TO BOUNDARY ──────────────────────────────────────────────────
def clip_to_boundary(df: pd.DataFrame, boundary) -> pd.DataFrame:
    before = len(df)
    mask = df.apply(lambda r: boundary.contains(Point(r["lon"], r["lat"])), axis=1)
    df = df[mask].reset_index(drop=True)
    print(f"[Clip] {before} → {len(df)} facilities ({before - len(df)} outside boundary)")
    return df


# ── STEP 4: H3 SCORING ───────────────────────────────────────────────────────
def score_hexes(df: pd.DataFrame, boundary, resolution: int = 7) -> list:
    # Get all H3 cells covering the boundary
    if isinstance(boundary, Polygon):
        polys = [boundary]
    else:
        polys = list(boundary.geoms)

    all_hexes = set()
    for poly in polys:
        outer = [(c[1], c[0]) for c in poly.exterior.coords]
        cells = h3.polygon_to_cells(h3.LatLngPoly(outer), resolution)
        all_hexes.update(cells)

    print(f"[H3] {len(all_hexes)} cells covering boundary at res={resolution}")

    # Remove cells whose centroid is outside boundary (clips sea/edge cells)
    all_hexes = {hx for hx in all_hexes
                 if boundary.contains(Point(h3.cell_to_latlng(hx)[1],
                                            h3.cell_to_latlng(hx)[0]))}
    print(f"[H3] {len(all_hexes)} cells after centroid clip")

    if not all_hexes:
        return []

    # Map facilities → H3 cells
    resource_counts = {}
    if not df.empty:
        df = df.copy()
        df["h3"] = df.apply(
            lambda r: h3.latlng_to_cell(float(r["lat"]), float(r["lon"]), resolution), axis=1
        )
        resource_counts = df["h3"].value_counts().to_dict()

    results = []
    for hx in all_hexes:
        neighbors   = h3.grid_disk(hx, 1)
        access      = int(sum(resource_counts.get(n, 0) for n in neighbors))
        vuln        = round(0.5 + np.random.random() * 0.5, 4)
        score       = round(vuln / (access + 1), 4)
        lat, lon    = h3.cell_to_latlng(hx)
        results.append({
            "h3_index":       hx,
            "lat":            round(lat, 6),
            "lon":            round(lon, 6),
            "priority_score": score,
            "access_count":   access,
        })

    return results


# ── MAIN ENDPOINT ─────────────────────────────────────────────────────────────
@app.get("/analyze")
async def analyze(city: str, resource: str):
    print(f"\n{'='*60}")
    print(f"[API] city={city!r}  resource={resource!r}")

    # 1. Boundary
    boundary = await get_boundary(city)
    if boundary is None:
        return {"error": f"Could not get a polygon boundary for '{city}'. Try a more specific name."}

    # 2. Fetch facilities from Overpass using boundary bbox
    s, w, n, e = boundary.bounds  # shapely: (minx, miny, maxx, maxy) = (w, s, e, n)
    bbox = (w, s, e, n)           # Overpass wants (south, west, north, east)
    facilities_df = await fetch_facilities(city, resource, bbox)

    if facilities_df.empty:
        return {"error": "No facilities found in this area."}

    # 3. Clip facilities to exact boundary polygon
    facilities_df = clip_to_boundary(facilities_df, boundary)

    # 4. Score H3 cells
    zones = score_hexes(facilities_df, boundary, resolution=7)
    print(f"[API] Returning {len(zones)} zones, {len(facilities_df)} facilities")

    return {
        "city":            city,
        "resource":        resource,
        "total_found":     len(facilities_df),
        "high_risk_zones": zones,
    }


@app.get("/debug")
async def debug(city: str = "Mumbai, India", resource: str = "medical"):
    boundary = await get_boundary(city)
    if boundary is None:
        return {"error": "No boundary found"}
    s, w, n, e = boundary.bounds
    bbox = (w, s, e, n)
    df = await fetch_facilities(city, resource, bbox)
    df = clip_to_boundary(df, boundary)
    zones = score_hexes(df, boundary)
    return {
        "boundary_type":    boundary.geom_type,
        "boundary_bounds":  boundary.bounds,
        "total_facilities": len(df),
        "total_zones":      len(zones),
        "facility_sample":  df.head(5).to_dict(orient="records"),
        "zone_sample":      zones[:5],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8008)