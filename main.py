from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import httpx
import asyncio
import h3
import pandas as pd
import numpy as np
import math
import json
import traceback
from shapely.geometry import shape, Point, MultiPolygon, Polygon
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder

app = FastAPI(title="EquiRoute")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

OVERPASS_URL  = "https://overpass-api.de/api/interpreter"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

RESOURCE_TAGS = {
    "medical":    ['["amenity"="hospital"]','["amenity"="clinic"]','["amenity"="doctors"]','["amenity"="pharmacy"]'],
    "education":  ['["amenity"="school"]','["amenity"="university"]','["amenity"="college"]'],
    "evacuation": ['["amenity"="shelter"]','["amenity"="community_centre"]','["amenity"="place_of_worship"]'],
}
RCOLORS = {
    "medical":    {"primary": "#e74c3c", "name": "Medical"},
    "education":  {"primary": "#3498db", "name": "Education"},
    "evacuation": {"primary": "#2ecc71", "name": "Evacuation"},
}

_boundary_cache: dict = {}

# ── HTML ──────────────────────────────────────────────────────────────────────
@app.get("/")
def root():      return FileResponse("intro.html")
@app.get("/index.html")
def dashboard(): return FileResponse("index.html")
@app.get("/feedback.html")
def feedback():  return FileResponse("feedback.html")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def _err(e): return f"{type(e).__name__}: {str(e)[:180]}"


async def get_boundary(city: str):
    key = city.lower().strip()
    if key in _boundary_cache:
        print(f"[Boundary] cache hit: {city!r}")
        return _boundary_cache[key]
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(25.0, connect=10.0), follow_redirects=True) as c:
                r = await c.get(NOMINATIM_URL,
                                params={"q": city, "format": "json", "limit": 5, "polygon_geojson": 1},
                                headers={"User-Agent": "EquiRoute/1.0"})
            for res in r.json():
                g = res.get("geojson", {})
                if g.get("type") in ("Polygon", "MultiPolygon"):
                    b = shape(g)
                    print(f"[Boundary] {g['type']} area={b.area:.4f} for {city!r}")
                    _boundary_cache[key] = b
                    return b
            return None
        except Exception as e:
            wait = 3 * (2 ** attempt)
            print(f"[Boundary] attempt {attempt+1}/3 failed — {_err(e)} — waiting {wait}s")
            await asyncio.sleep(wait)
    return None


async def fetch_facilities(bbox: tuple, resource: str) -> list:
    """
    Sequential fetch with smart per-status backoff:
      429 / 504 / OSError  →  wait 15s, 30s, 60s  (Overpass rate-limit recovery)
      other network errors →  wait 5s,  10s, 20s
    Returns [] after 4 failed attempts instead of crashing.
    """
    s, w, n, e = bbox
    bstr = f"{s},{w},{n},{e}"
    tags = RESOURCE_TAGS.get(resource, RESOURCE_TAGS["medical"])
    parts = [f'node{t}({bstr});\nway{t}({bstr});' for t in tags]
    query = f"[out:json][timeout:60];\n(\n{''.join(parts)}\n);\nout center tags;\n"

    for attempt in range(4):
        wait_long  = 15 * (2 ** min(attempt, 2))   # 15 / 30 / 60
        wait_short =  5 * (2 ** min(attempt, 2))    #  5 / 10 / 20
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(90.0, connect=15.0),
                follow_redirects=True,
            ) as c:
                r = await c.post(OVERPASS_URL, data={"data": query})

            if r.status_code in (429, 504):
                # Rate-limited or gateway timeout — must wait longer
                print(f"[Overpass] {resource} attempt {attempt+1}/4 — HTTP {r.status_code} (rate limit) — waiting {wait_long}s")
                if attempt < 3:
                    await asyncio.sleep(wait_long)
                continue

            if r.status_code != 200:
                print(f"[Overpass] {resource} attempt {attempt+1}/4 — HTTP {r.status_code} — waiting {wait_short}s")
                if attempt < 3:
                    await asyncio.sleep(wait_short)
                continue

            rows = []
            for el in r.json().get("elements", []):
                if el["type"] == "node":
                    lat, lon = el.get("lat"), el.get("lon")
                elif el["type"] == "way" and "center" in el:
                    lat, lon = el["center"]["lat"], el["center"]["lon"]
                else:
                    continue
                t = el.get("tags", {})
                rows.append({
                    "lat": lat, "lon": lon,
                    "name":          t.get("name") or t.get("name:en") or "Unnamed Facility",
                    "phone":         t.get("phone") or t.get("contact:phone") or None,
                    "website":       t.get("website") or t.get("contact:website") or None,
                    "opening_hours": t.get("opening_hours") or None,
                    "amenity":       t.get("amenity") or resource,
                })
            print(f"[Overpass] {resource}: {len(rows)} raw facilities (attempt {attempt+1})")
            return rows

        except (OSError, ConnectionResetError) as e:
            # Windows WinError 10054 — connection forcibly closed by remote host
            print(f"[Overpass] {resource} attempt {attempt+1}/4 — OS/reset error ({_err(e)}) — waiting {wait_long}s")
            if attempt < 3:
                await asyncio.sleep(wait_long)
        except (httpx.ConnectError, httpx.TimeoutException,
                httpx.RemoteProtocolError, httpx.NetworkError) as e:
            print(f"[Overpass] {resource} attempt {attempt+1}/4 — network ({_err(e)}) — waiting {wait_short}s")
            if attempt < 3:
                await asyncio.sleep(wait_short)
        except Exception as e:
            print(f"[Overpass] {resource} attempt {attempt+1}/4 — unexpected ({_err(e)}) — waiting {wait_short}s")
            if attempt < 3:
                await asyncio.sleep(wait_short)

    print(f"[Overpass] {resource} GAVE UP after 4 attempts — returning []")
    return []


def clip_to_boundary(rows, boundary):
    out = [r for r in rows if boundary.contains(Point(r["lon"], r["lat"]))]
    print(f"[Clip] {len(rows)} → {len(out)}")
    return out



# ── RECOMMENDATIONS ENGINE ────────────────────────────────────────────────────
_RECS = {
    "medical": {
        "critical": [
            "Urgently establish a primary health centre or mobile clinic in this zone",
            "Deploy an emergency medical response unit for rapid incident coverage",
            "Negotiate ambulance pre-positioning with the nearest hospital",
            "Launch a community health-worker programme for door-to-door triage",
            "Apply for PM-JAY / Ayushman Bharat facility-expansion grant",
        ],
        "moderate": [
            "Extend clinic operating hours to evenings and weekends",
            "Set up a telemedicine kiosk to broaden specialist reach",
            "Partner with pharmacies to run basic diagnostic screening camps",
            "Strengthen the referral pathway to the nearest secondary-care hospital",
            "Train 5–10 local volunteers in first-aid and emergency response",
        ],
        "low": [
            "Run quarterly health-awareness camps (diabetes, BP, maternal health)",
            "Publish a public directory of nearby facilities with contact details",
            "Encourage registration under the National Health Mission for subsidised care",
        ],
        "served": [
            "Coverage is adequate — schedule an annual facility audit",
            "Explore preventive-care programmes to reduce future demand",
        ],
    },
    "education": {
        "critical": [
            "Prioritise construction of a government primary school in this zone",
            "Launch bridge-course centres to re-enrol out-of-school children",
            "Deploy mobile school vans for hard-to-reach sub-zones",
            "Partner with NGOs (Pratham, Teach For India) for interim coverage",
            "Apply for Samagra Shiksha Abhiyan infrastructure funding",
        ],
        "moderate": [
            "Add classrooms or a second shift to reduce severe overcrowding",
            "Establish a community learning centre for after-school tutoring",
            "Expand the mid-day meal programme to improve enrolment & retention",
            "Distribute digital tablets / e-learning kits to supplement teaching",
            "Run regular teacher-training workshops to lift learning outcomes",
        ],
        "low": [
            "Organise skill-development workshops for school dropouts",
            "Set up a public library or reading room in the neighbourhood",
            "Monitor attendance and enrolment trends for early intervention",
        ],
        "served": [
            "Coverage is adequate — focus on quality-improvement metrics",
            "Introduce STEM enrichment programmes and extracurricular activities",
        ],
    },
    "evacuation": {
        "critical": [
            "Immediately designate and equip an emergency evacuation shelter",
            "Map all multi-storey buildings suitable as vertical evacuation points",
            "Install an emergency broadcast / siren system for rapid alerts",
            "Conduct a mandatory evacuation drill within the next 30 days",
            "Pre-position 72-hour emergency supplies: water, food, first-aid kits",
        ],
        "moderate": [
            "Establish a secondary evacuation route and signpost it clearly",
            "Train community wardens in evacuation coordination procedures",
            "Update the local disaster-management plan to include this zone",
            "Ensure shelter capacity for at least 500 displaced persons",
            "Coordinate with civil-defence volunteers for regular mock drills",
        ],
        "low": [
            "Verify existing shelters meet current structural-safety standards",
            "Distribute household emergency-preparedness guides to residents",
            "Maintain an updated register of vulnerable residents (elderly, disabled)",
        ],
        "served": [
            "Evacuation infrastructure is adequate — run an annual readiness audit",
            "Improve accessibility at shelters for persons with disabilities",
        ],
    },
}

def get_recommendations(resource: str, score: float, access: int) -> dict:
    pool = _RECS.get(resource, _RECS["medical"])
    if   score > 0.6: tier, items = "critical", pool["critical"]
    elif score > 0.4: tier, items = "moderate", pool["moderate"]
    elif score > 0.2: tier, items = "low",      pool["low"]
    else:             tier, items = "served",   pool["served"]
    # Pick 2 items deterministically (same zone always gets same recs)
    idx   = access % max(1, len(items))
    picks = [items[idx], items[(idx + 1) % len(items)]]
    return {"tier": tier, "items": picks}

def score_hexes(rows, boundary, resolution=7, resource='medical'):
    polys = [boundary] if isinstance(boundary, Polygon) else list(boundary.geoms)
    all_hexes = set()
    for poly in polys:
        outer = [(c[1], c[0]) for c in poly.exterior.coords]
        all_hexes.update(h3.polygon_to_cells(h3.LatLngPoly(outer), resolution))
    all_hexes = {hx for hx in all_hexes
                 if boundary.contains(Point(h3.cell_to_latlng(hx)[1], h3.cell_to_latlng(hx)[0]))}
    if not all_hexes:
        return []
    rcounts = {}
    if rows:
        df = pd.DataFrame(rows)
        df["h3"] = df.apply(lambda r: h3.latlng_to_cell(float(r["lat"]), float(r["lon"]), resolution), axis=1)
        rcounts = df["h3"].value_counts().to_dict()
    results = []
    for hx in all_hexes:
        access = int(sum(rcounts.get(n, 0) for n in h3.grid_disk(hx, 1)))
        vuln   = round(0.5 + np.random.random()*0.5, 4)
        score  = round(vuln / (access + 1), 4)
        lat, lon = h3.cell_to_latlng(hx)
        recs = get_recommendations(resource, score, access)
        results.append({"h3_index": hx, "lat": round(lat,6), "lon": round(lon,6),
                         "priority_score": score, "access_count": access,
                         "rec_tier": recs["tier"], "rec_items": recs["items"]})
    return results


def fig_to_json(fig):
    return json.loads(PlotlyJSONEncoder().encode(fig))

def safe_kmeans(df, n_clusters=6):
    k = max(2, min(n_clusters, len(df) // 2))
    coords = StandardScaler().fit_transform(df[["lat","lon"]].values)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(coords), km


# ── CLUSTER CHART BUILDER ─────────────────────────────────────────────────────
def build_cluster_charts(rows: list, zones: list, city: str, resource: str) -> dict:
    """Build 6 cluster charts for a single service."""
    charts = {}
    col   = RCOLORS[resource]["primary"]
    label = RCOLORS[resource]["name"]
    PALETTE = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c","#e67e22","#e91e63"]

    BASE = dict(
        paper_bgcolor="rgba(248,244,236,0)", plot_bgcolor="rgba(248,244,236,0)",
        font=dict(family="Sora, sans-serif", color="#2a2520", size=12),
    )
    MARGIN     = dict(l=30, r=20, t=50, b=30)
    MARGIN_MAP = dict(l=0,  r=0,  t=40, b=0)

    if len(rows) < 4:
        return {}

    df = pd.DataFrame(rows)
    labels, km = safe_kmeans(df)
    df["cluster"] = labels
    scaler  = StandardScaler().fit(df[["lat","lon"]].values)
    centres = scaler.inverse_transform(km.cluster_centers_)
    csizes  = pd.Series(labels).value_counts().sort_index()

    # 1. SCATTER MAP — each cluster a different colour
    fig_map = go.Figure()
    for cid in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cid]
        fig_map.add_trace(go.Scattermap(
            lat=sub["lat"].tolist(), lon=sub["lon"].tolist(), mode="markers",
            marker=dict(size=7, color=PALETTE[cid % len(PALETTE)], opacity=0.7),
            name=f"Cluster {cid} ({len(sub)})",
            text=sub["name"].tolist(), hovertemplate="<b>%{text}</b><extra></extra>",
        ))
    for i, c in enumerate(centres):
        fig_map.add_trace(go.Scattermap(
            lat=[float(c[0])], lon=[float(c[1])], mode="markers",
            marker=dict(size=18, color=PALETTE[i % len(PALETTE)], opacity=1.0),
            name=f"C{i} centroid",
            text=[f"{label} Cluster {i} — {csizes.get(i,0)} facilities"],
            hovertemplate="<b>%{text}</b><extra></extra>",
            showlegend=False,
        ))
    fig_map.update_layout(
        **BASE, height=460, margin=MARGIN_MAP,
        map=dict(style="open-street-map",
                 center=dict(lat=float(df["lat"].mean()), lon=float(df["lon"].mean())),
                 zoom=11),
        title=dict(text=f"{label} Cluster Map — {city}", font=dict(size=14)),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", font=dict(size=10)),
    )
    charts["cluster_map"] = fig_to_json(fig_map)

    # 2. BAR — facilities per cluster
    counts = csizes.sort_index()
    fig_bar = go.Figure(go.Bar(
        x=[f"Cluster {i}" for i in counts.index],
        y=counts.values.tolist(),
        marker_color=[PALETTE[i % len(PALETTE)] for i in counts.index],
        text=counts.values.tolist(), textposition="outside",
    ))
    fig_bar.update_layout(
        **BASE, height=300, margin=MARGIN, showlegend=False,
        title=dict(text=f"{label} — Facilities per Cluster — {city}", font=dict(size=14)),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Facilities", gridcolor="rgba(0,0,0,0.06)"),
    )
    charts["cluster_bar"] = fig_to_json(fig_bar)

    # 3. PIE — amenity sub-type breakdown within this service
    amenity_counts = df["amenity"].value_counts()
    fig_pie = go.Figure(go.Pie(
        labels=amenity_counts.index.tolist(),
        values=amenity_counts.values.tolist(),
        hole=0.42, textinfo="label+percent",
        marker=dict(colors=[PALETTE[i % len(PALETTE)] for i in range(len(amenity_counts))]),
        hovertemplate="<b>%{label}</b>: %{value} (%{percent})<extra></extra>",
    ))
    fig_pie.update_layout(
        **BASE, height=300, margin=MARGIN, showlegend=False,
        title=dict(text=f"{label} — Sub-type Breakdown — {city}", font=dict(size=14)),
    )
    charts["distribution_pie"] = fig_to_json(fig_pie)

    # 4. ACCESS COUNT HISTOGRAM
    if zones:
        fig_hist = go.Figure(go.Histogram(
            x=[z["access_count"] for z in zones],
            marker_color=col, opacity=0.8, nbinsx=20,
        ))
        fig_hist.update_layout(
            **BASE, height=300, margin=MARGIN, showlegend=False,
            title=dict(text=f"{label} — Access Count Distribution — {city}", font=dict(size=14)),
            xaxis=dict(title="Facilities Nearby (incl. H3 neighbours)", showgrid=False),
            yaxis=dict(title="Hex Zones", gridcolor="rgba(0,0,0,0.06)"),
        )
        charts["access_histogram"] = fig_to_json(fig_hist)

    # 5. PRIORITY vs ACCESS DENSITY HEATMAP
    if zones:
        zdf = pd.DataFrame(zones)
        fig_heat = px.density_heatmap(
            zdf, x="access_count", y="priority_score",
            color_continuous_scale="RdYlGn_r", nbinsx=20, nbinsy=20,
            labels={"access_count": "Facilities Nearby", "priority_score": "Priority Score"},
        )
        fig_heat.update_layout(
            **BASE, height=300, margin=MARGIN,
            title=dict(text=f"{label} — Priority vs Access Density — {city}", font=dict(size=14)),
        )
        charts["density_heat"] = fig_to_json(fig_heat)

    # 6. ELBOW CURVE
    coords = StandardScaler().fit_transform(df[["lat","lon"]].values)
    ks, inertias = [], []
    for k in range(2, min(12, len(df)//3 + 1)):
        km2 = KMeans(n_clusters=k, random_state=42, n_init=8)
        km2.fit(coords); ks.append(k); inertias.append(km2.inertia_)
    if ks:
        fig_elbow = go.Figure(go.Scatter(
            x=ks, y=inertias, mode="lines+markers",
            line=dict(color=col, width=2), marker=dict(size=7, color=col),
            name=label,
        ))
        fig_elbow.update_layout(
            **BASE, height=300, margin=MARGIN, showlegend=False,
            title=dict(text=f"{label} — Elbow Curve (Optimal K) — {city}", font=dict(size=14)),
            xaxis=dict(title="Number of Clusters (K)", showgrid=False, dtick=1),
            yaxis=dict(title="Inertia", gridcolor="rgba(0,0,0,0.06)"),
        )
        charts["elbow_curve"] = fig_to_json(fig_elbow)

    return charts

# ── /analyze ─────────────────────────────────────────────────────────────────
@app.get("/analyze")
async def analyze(city: str, resource: str):
    print(f"\n{'='*60}\n[analyze] {city!r} / {resource!r}")
    boundary = await get_boundary(city)
    if not boundary:
        return {"error": f"Could not get boundary for '{city}'. Check connection and retry."}
    w, s, e, n = boundary.bounds
    rows = await fetch_facilities((s, w, n, e), resource)
    if not rows:
        return {"error": "Facility fetch failed — Overpass may be rate-limiting. Wait 30s and retry."}
    rows = clip_to_boundary(rows, boundary)
    zones = score_hexes(rows, boundary, resource=resource)
    return {"city": city, "resource": resource, "total_found": len(rows), "high_risk_zones": zones}


# ── /nearest ─────────────────────────────────────────────────────────────────
@app.get("/nearest")
async def nearest(lat: float, lon: float, resource: str, radius_km: float = 10):
    print(f"\n[nearest] {lat},{lon} / {resource} / {radius_km}km")
    deg = radius_km / 111.0
    s, w, n, e = lat-deg, lon-deg, lat+deg, lon+deg
    rows = await fetch_facilities((s, w, n, e), resource)
    if not rows:
        return {"error": "Facility fetch failed — please retry.", "results": []}
    for r in rows:
        r["distance_km"] = round(haversine(lat, lon, r["lat"], r["lon"]), 3)
    rows.sort(key=lambda r: r["distance_km"])
    return {"user_lat": lat, "user_lon": lon, "resource": resource, "results": rows[:10]}


# ── /clusters ────────────────────────────────────────────────────────────────
@app.get("/clusters")
async def clusters(city: str, resource: str = "medical"):
    print(f"\n[clusters] city={city!r} resource={resource!r}")
    boundary = await get_boundary(city)
    if not boundary:
        return {"error": f"Could not get boundary for '{city}'. Check connection and retry."}

    w, s, e, n = boundary.bounds
    rows = await fetch_facilities((s, w, n, e), resource)
    if not rows:
        return {"error": f"Facility fetch failed for {resource}. Overpass may be rate-limiting — wait 30s and retry."}

    rows  = clip_to_boundary(rows, boundary)
    zones = score_hexes(rows, boundary, resource=resource)
    print(f"[clusters] {resource}: {len(rows)} facilities, {len(zones)} zones")

    try:
        charts = build_cluster_charts(rows, zones, city, resource)
    except Exception as ex:
        traceback.print_exc()
        return {"error": f"Chart generation failed: {ex}"}

    return {"city": city, "resource": resource, "total": len(rows), "charts": charts}


# ── /debug ────────────────────────────────────────────────────────────────────
@app.get("/debug")
async def debug(city: str = "Mumbai, India", resource: str = "medical"):
    boundary = await get_boundary(city)
    if not boundary:
        return {"error": "No boundary — check internet"}
    w, s, e, n = boundary.bounds
    rows = await fetch_facilities((s, w, n, e), resource)
    rows = clip_to_boundary(rows, boundary)
    zones = score_hexes(rows, boundary, resource=resource)
    return {"boundary_type": boundary.geom_type, "total_facilities": len(rows),
            "total_zones": len(zones), "sample": rows[:3]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8008)