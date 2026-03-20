import h3
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union

print(f"[Engine] H3 version: {h3.__version__}")

class EquityEngine:
    def __init__(self, resolution=7):
        self.res = resolution

    def _shapely_to_h3_cells(self, poly):
        # Shapely coords are (lng, lat) — H3 needs (lat, lng)
        outer_latlng = [(c[1], c[0]) for c in poly.exterior.coords]
        h3_poly = h3.LatLngPoly(outer_latlng)
        return h3.polygon_to_cells(h3_poly, self.res)

    def get_h3_stats(self, resource_df, city_boundary):
        try:
            if isinstance(city_boundary, Polygon):
                polygons = [city_boundary]
            else:
                polygons = list(city_boundary.geoms)

            all_city_hexes = set()
            for poly in polygons:
                hexes = self._shapely_to_h3_cells(poly)
                all_city_hexes.update(hexes)

            print(f"[Engine] {len(all_city_hexes)} hex cells before boundary clip")

            # ── CLIP: keep only hexes whose centre is inside the boundary ──
            # This removes sea hexagons even when boundary is a circle,
            # as long as the boundary itself is correct.
            # More importantly it removes cells that straddle the edge.
            clipped = set()
            for hex_id in all_city_hexes:
                lat, lon = h3.cell_to_latlng(hex_id)
                if city_boundary.contains(Point(lon, lat)):
                    clipped.add(hex_id)

            print(f"[Engine] {len(clipped)} hex cells after boundary clip "
                  f"({len(all_city_hexes) - len(clipped)} edge cells removed)")
            all_city_hexes = clipped

            if not all_city_hexes:
                print("[Engine] WARNING: 0 hexes after clip")
                return pd.DataFrame()

            # Map facilities to H3 cells
            resource_counts = {}
            if not resource_df.empty:
                df = resource_df.copy()
                df['h3_index'] = df.apply(
                    lambda row: h3.latlng_to_cell(float(row['lat']), float(row['lon']), self.res),
                    axis=1
                )
                resource_counts = df['h3_index'].value_counts().to_dict()
                print(f"[Engine] {len(df)} facilities across {len(resource_counts)} unique cells")

            results = []
            for hex_id in all_city_hexes:
                neighbors = h3.grid_disk(hex_id, 1)
                access_count = int(sum(resource_counts.get(n, 0) for n in neighbors))
                vulnerability = round(0.5 + (np.random.random() * 0.5), 4)
                priority_score = round(vulnerability / (access_count + 1), 4)
                lat, lon = h3.cell_to_latlng(hex_id)
                results.append({
                    "h3_index":       hex_id,
                    "lat":            round(lat, 6),
                    "lon":            round(lon, 6),
                    "priority_score": priority_score,
                    "access_count":   access_count
                })

            print(f"[Engine] Returning {len(results)} scored zones")
            return pd.DataFrame(results)

        except Exception as e:
            import traceback
            print(f"[Engine] CRASH: {e}")
            traceback.print_exc()
            return pd.DataFrame()