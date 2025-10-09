
import os, sys

from shapely import transform 

sys.path.append("../../Framework")


# from Experimental.Abstraction.hierarchical_voronoi.example1 import pixel_to_geo
from pathplanning import GCSPathPlanningPipeline, GraphPathPlanningPipeline
from utils import plot_gcs_path, plot_gcs_path_elevation
from cost_models import HumanModelObjective, EuclideanObjective, EuclideanObjective2D, LinearEnergyFunction


class CustomGraphPathPlanningPipeline(GraphPathPlanningPipeline):

    def run_data(self, elevation_map, landcover_map, start_coord, goal_coord):
        print(f"Example: {self.example}\nRegion Count: {self.region_count}\nDecomposition: {self.decomposition}\nMethod: {self.method}")
        self.load_region_builder(True, landcover_map, elevation_map)
        self.build_model_and_graph()
        # start_coord, goal_coord = get_start_goal_coords(self.example)
        region_path = self.compute_path(start_coord, goal_coord)
        # self.compute_gcs_path(start_coord, goal_coord, region_path)

import pickle 

import numpy as np
import matplotlib.pyplot as plt



import numpy as np
from itertools import permutations



def hillshade(z, dx=1.0, dy=1.0, azimuth=315.0, altitude=45.0):
    """Simple hillshade from DEM (radians) with given light azimuth/altitude."""
    # gradients
    gy, gx = np.gradient(z, dy, dx)
    slope = np.pi/2.0 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az = np.deg2rad(azimuth)
    alt = np.deg2rad(altitude)
    hs = (np.sin(alt) * np.sin(slope) +
          np.cos(alt) * np.cos(slope) * np.cos(az - aspect))
    # normalize to [0,1]
    hs = (hs - np.nanmin(hs)) / (np.nanmax(hs) - np.nanmin(hs) + 1e-9)
    return hs



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def compute_segmented_grade_with_outliers(dem, cellsize=30.0, step=5, min_pixels=20):
    """
    Compute %grade from DEM, segment into bins of `step`,
    and mark bins with < min_pixels pixels as non-traversable (-1).
    Removes empty bins from plots/histogram.
    """
    # Compute slope and grade
    dz_dy, dz_dx = np.gradient(dem, cellsize)
    slope = np.sqrt(dz_dx**2 + dz_dy**2)
    grade = slope * 100.0

    # Define bins
    max_grade = np.ceil(grade.max() / step) * step
    bins = np.arange(0, max_grade + step, step)  # 0,5,10,...
    seg = np.digitize(grade, bins) - 1

    # Count pixels per bin
    n_bins = len(bins) - 1
    counts = np.array([(seg == i).sum() for i in range(n_bins)])

    # Mark bins with <min_pixels as non-traversable (1000)
    non_traversible_grade = -1
    seg_corrected = seg.copy()
    for i in range(n_bins):
        if counts[i] < min_pixels:
            seg_corrected[seg == i] = non_traversible_grade

    # Recompute counts after filtering
    valid_bins = [i for i in range(n_bins) if (seg_corrected == i).sum() > 0]
    counts_valid = [(seg_corrected == non_traversible_grade).sum()] + [(seg_corrected == i).sum() for i in valid_bins]
    # bins_reduced = [(-1, "Non-traversable")] + [(i, f"{bins[i]}–{bins[i+1]}%") for i in valid_bins]
    bins_reduced =   [(i, f"{bins[i]}–{bins[i+1]}%") for i in valid_bins]+ [(non_traversible_grade, "Non-traversable")]


    # Build colormap (black for -1, colors for valid bins)
    cmap = plt.cm.get_cmap("turbo", len(valid_bins))
    colors = [cmap(j) for j in range(len(valid_bins))] + ["black"]
    cmap_final = ListedColormap(colors)

    # Segmented grade (filtered)
    # remap seg_corrected to contiguous IDs: -1 → 0, valid bins → 1..N
    remap = {-1: 0}
    for new_idx, old_idx in enumerate(valid_bins, start=1):
        remap[old_idx] = new_idx
    seg_display = np.vectorize(lambda x: remap.get(x, 0))(seg_corrected)


    
    return seg_display, bins_reduced, cmap_final

def get_grade_image(dem):
    
    dx, dy = 0.00026949458523585647, 0.00026949458523585647
    dem_filled = np.where(np.isnan(dem), np.nanmedian(dem[~np.isnan(dem)]) if np.any(~np.isnan(dem)) else 0.0, dem)

    hs = hillshade(dem_filled, dx=dx, dy=dy, azimuth=315.0, altitude=45.0)


    seg_display, bins_reduced, cmap_final = compute_segmented_grade_with_outliers(dem)
    return seg_display


import geopandas as gpd
from shapely.geometry import Polygon

import numpy as np
import matplotlib.pyplot as plt
import imageio

class TaskZoomAnimator:
    def __init__(self, tasks, voronoi_gdf, grade_image, extent, depots=None):
        """
        Parameters
        ----------
        tasks : np.ndarray
            Array of task coordinates, shape (N,2).
        voronoi_gdf : GeoDataFrame
            GeoDataFrame with Voronoi polygons.
        grade_image : np.ndarray
            DEM/grade image to use as background.
        extent : list
            [xmin, xmax, ymin, ymax] extent of the DEM in pixel coords.
        depots : dict, optional
            Dictionary of depot indices, e.g. {"vehicle":2, "human":10}
        """
        self.tasks = tasks
        self.voronoi_gdf = voronoi_gdf
        self.grade_image = grade_image
        self.extent = extent
        self.frames = []
        self.depots = depots if depots is not None else {}

        self.H = extent[3] - extent[2]
        self.W = extent[1] - extent[0]

        # Setup figure/axes once
        self.fig, self.ax = plt.subplots(figsize=(8,6))
        self._draw_static_scene()
        self.highlight, = self.ax.plot([], [], 'o', c="orange", markersize=6)

        self.path_line, = self.ax.plot([], [], '-', c="orange", lw=2, alpha=0.7)  # path trail

        # Store visited points for trail
        self.path_x, self.path_y = [], []


    def _draw_static_scene(self):
        """Draw DEM, Voronoi, and static task points."""
        self.ax.imshow(self.grade_image, cmap='terrain', extent=self.extent,
                       origin='lower', alpha=0.5)
        self.voronoi_gdf.boundary.plot(ax=self.ax, color="blue", linewidth=0.1)
        self.ax.scatter(self.tasks[:,0], self.tasks[:,1], c="lightgray", s=20)

        # Highlight depots if defined
        if "vehicle" in self.depots:
            vidx = self.depots["vehicle"]
            self.ax.scatter(self.tasks[vidx,0], self.tasks[vidx,1],
                            c="green", s=80, label="Vehicle Depot")
        if "human" in self.depots:
            hidx = self.depots["human"]
            self.ax.scatter(self.tasks[hidx,0], self.tasks[hidx,1],
                            c="red", s=80, label="Human Depot")

        self.ax.set_xticks([])
        self.ax.set_yticks([])

    def _capture_frame(self):
        """Capture current frame into buffer."""
        self.fig.canvas.draw()
        image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        self.frames.append(image)

    def animate_zoom_sequence(self, task_sequence, zoom_scales=np.linspace(1.0, 0.2, 20)):
        """
        Animate zoom-in for each task in sequence.

        Parameters
        ----------
        task_sequence : list
            List of task indices to zoom into.
        zoom_scales : iterable
            Sequence of zoom scales, default is np.linspace(1.0, 0.2, 20).
        """
        for idx in task_sequence:
            x, y = self.tasks[idx]
            self.highlight.set_data([x], [y])  # move marker
            for scale in zoom_scales:
                half_w = (self.W/4) * scale
                half_h = (self.H/4) * scale
                self.ax.set_xlim(x-half_w, x+half_w)
                self.ax.set_ylim(y-half_h, y+half_h)
                self._capture_frame()

    def animate_pan_sequence(self, task_sequence, n_steps=30, zoom_scale=0.3):
        """
        Smoothly pan between tasks in sequence (continuous motion).
        Draws a path as the camera moves.
        
        Parameters
        ----------
        task_sequence : list
            List of task indices to follow.
        n_steps : int
            Number of frames per transition.
        zoom_scale : float
            How much of the map width/height to show (smaller = closer zoom).
        """
        half_w = (self.W/4) * zoom_scale
        half_h = (self.H/4) * zoom_scale

        for i in range(len(task_sequence)-1):
            start = self.tasks[task_sequence[i]]
            end = self.tasks[task_sequence[i+1]]

            xs = np.linspace(start[0], end[0], n_steps)
            ys = np.linspace(start[1], end[1], n_steps)

            for x, y in zip(xs, ys):
                # Update highlight marker
                self.highlight.set_data([x], [y])

                # Add to path trail
                self.path_x.append(x)
                self.path_y.append(y)
                self.path_line.set_data(self.path_x, self.path_y)

                # Update view window
                self.ax.set_xlim(x-half_w, x+half_w)
                self.ax.set_ylim(y-half_h, y+half_h)

                self._capture_frame()



    def save_gif(self, filename="task_zoom_sequence.gif", fps=10):
        """Save collected frames into a GIF."""
        imageio.mimsave(filename, self.frames, fps=fps)
        print(f"Saved GIF: {filename}")



import numpy as np
import rasterio

import imageio
import os
os.environ["PYVISTA_OFF_SCREEN"] = "1"
os.environ["PYVISTA_USE_OSMESA"] = "1"

import pyvista as pv

# print(pv.Report())  # should show OSMesa instead of GLX

class TaskZoomAnimator3D:
    def __init__(self, dem_path, tasks=None, sat_image=None):
        """
        Parameters
        ----------
        dem_path : str
            Path to DEM GeoTIFF (e.g. NASADEM).
        tasks : np.ndarray, optional
            Array of task coordinates [[x,y], ...] in DEM CRS.
        sat_image : np.ndarray, optional
            Satellite image aligned with DEM (optional).
        """
        self.dem_path = dem_path
        self.tasks = tasks
        self.sat_image = sat_image
        self.frames = []

        # Load DEM
        with rasterio.open(dem_path) as src:
            self.dem = src.read(1)
            self.bounds = src.bounds
            self.extent = [self.bounds.left, self.bounds.right,
                           self.bounds.bottom, self.bounds.top]
            self.transform = src.transform

        # Mesh grid for DEM
        nrows, ncols = self.dem.shape
        xs = np.linspace(self.extent[0], self.extent[1], ncols)
        ys = np.linspace(self.extent[2], self.extent[3], nrows)
        xx, yy = np.meshgrid(xs, ys)
        self.surface = pv.StructuredGrid(xx, yy, self.dem)

        # Texture if satellite provided
        if self.sat_image is not None:
            self.surface.texture_map_to_plane(inplace=True)
            self.texture = pv.numpy_to_texture(self.sat_image)
        else:
            self.texture = None

    def animate_pan_sequence(self, task_sequence, n_steps=30, zoom_height=2000, filename="zoom3d.gif"):
        """
        Animate camera flying between tasks in 3D terrain.
        
        Parameters
        ----------
        task_sequence : list of indices
            List of task indices to fly over.
        n_steps : int
            Number of frames per transition.
        zoom_height : float
            Camera height above terrain.
        filename : str
            Output filename (.gif or .mp4).
        """
        plotter = pv.Plotter(off_screen=True)

        # Add DEM mesh
        if self.texture:
            plotter.add_mesh(self.surface, texture=self.texture)
        else:
            plotter.add_mesh(self.surface, cmap="terrain")

        # Highlight tasks
        if self.tasks is not None:
            # Convert 2D tasks → 3D with DEM elevation
            with rasterio.open(self.dem_path) as src:
                zvals = [val[0] for val in src.sample([tuple(pt) for pt in self.tasks])]
            tasks3d = np.column_stack([self.tasks, zvals])
            plotter.add_points(tasks3d, color="red", point_size=12, render_points_as_spheres=True)

        # Open movie writer (gif or mp4)
        if filename.endswith(".gif"):
            plotter.open_gif(filename)
        else:
            plotter.open_movie(filename, framerate=15)

        # Fly along sequence
        for i in range(len(task_sequence)-1):
            start = self.tasks[task_sequence[i]]
            end = self.tasks[task_sequence[i+1]]

            xs = np.linspace(start[0], end[0], n_steps)
            ys = np.linspace(start[1], end[1], n_steps)

            for x, y in zip(xs, ys):
                # Get elevation from DEM
                with rasterio.open(self.dem_path) as src:
                    z = [val[0] for val in src.sample([(x,y)])][0]

                # Set camera
                plotter.camera_position = [
                    (x, y, z + zoom_height),  # camera eye
                    (x, y, z),                # look-at target
                    (0, 1, 0)                 # up vector
                ]
                plotter.write_frame()

        plotter.close()
        print(f"Saved animation → {filename}")



import numpy as np
import rasterio
import plotly.graph_objects as go

class TaskZoomAnimatorWebGL:
    def __init__(self, dem_path, tasks=None):
        """
        Parameters
        ----------
        dem_path : str
            Path to DEM GeoTIFF (e.g. NASADEM).
        tasks : np.ndarray, optional
            Array of task coordinates [[x,y], ...] in DEM CRS.
        """
        self.dem_path = dem_path
        self.tasks = tasks

        # Load DEM
        with rasterio.open(dem_path) as src:
            self.dem = src.read(1)
            self.bounds = src.bounds
            self.extent = [self.bounds.left, self.bounds.right,
                           self.bounds.bottom, self.bounds.top]
            self.transform = src.transform

        # Build meshgrid
        nrows, ncols = self.dem.shape
        xs = np.linspace(self.extent[0], self.extent[1], ncols)
        ys = np.linspace(self.extent[2], self.extent[3], nrows)
        self.xx, self.yy = np.meshgrid(xs, ys)

        # Create base figure
        self.fig = go.Figure()

        # DEM surface
        self.fig.add_trace(go.Surface(
            z=self.dem,
            x=self.xx,
            y=self.yy,
            colorscale="earth",
            showscale=False,
            opacity=0.95
        ))

        # Tasks
        if self.tasks is not None:
            with rasterio.open(dem_path) as src:
                zvals = [val[0] for val in src.sample([tuple(pt) for pt in tasks])]
            tasks3d = np.column_stack([tasks, zvals])
            self.fig.add_trace(go.Scatter3d(
                x=tasks3d[:,0], y=tasks3d[:,1], z=tasks3d[:,2],
                mode="markers",
                marker=dict(size=5, color="red"),
                name="Tasks"
            ))

        # Default camera
        self.fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            ),
            margin=dict(l=0, r=0, t=0, b=0)
        )

    def animate_pan_sequence(self, task_sequence, n_steps=30, zoom_height=2000, filename="zoom.html"):
        """
        Animate zoom/pan between tasks in WebGL (Plotly).
        Exports an interactive HTML animation.
        """
        frames = []
        for i in range(len(task_sequence)-1):
            start = self.tasks[task_sequence[i]]
            end = self.tasks[task_sequence[i+1]]
            xs = np.linspace(start[0], end[0], n_steps)
            ys = np.linspace(start[1], end[1], n_steps)

            with rasterio.open(self.dem_path) as src:
                zs = [val[0] for val in src.sample(zip(xs, ys))]

            for x, y, z in zip(xs, ys, zs):
                frames.append(go.Frame(
                    layout=dict(
                        scene_camera=dict(
                            eye=dict(x=1.5, y=1.5, z=0.8),  # orbital angle
                            center=dict(x=x, y=y, z=z),
                        )
                    )
                ))

        self.fig.update(frames=frames)

        # Add play button
        self.fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(label="Play",
                              method="animate",
                              args=[None, {"frame": {"duration": 100, "redraw": True},
                                           "fromcurrent": True,
                                           "transition": {"duration": 0}}])]
            )]
        )

        # Save to HTML
        self.fig.write_html(filename)
        print(f"Saved interactive animation → {filename}")






import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon
from rasterio import features
from rasterstats import zonal_stats


# NALCMS_COLORS = {
#   1:  "#0b3d0b",  # Needleleaf forest (dark green)
#   2:  "#206920",  # Taiga needleleaf forest
#   3:  "#2e8b57",  # Broadleaf evergreen forest
#   4:  "#66c266",  # Broadleaf deciduous forest
#   5:  "#98d398",  # Temperate deciduous forest
#   6:  "#8bab6f",  # Mixed forest
#   7:  "#c2b280",  # Shrubland
#   8:  "#dcd37f",  # Grassland
#   9:  "#d4ca68",  # Tropical grassland
#   10: "#bbaa44",  # Shrubland variant
#   11: "#6ab47b",  # Wetland ( swamp-green )
#   12: "#a9cd8c",  # Cropland (light green/beige)
#   13: "#c4c48e",  # Barren lands
#   14: "#999999",  # Urban (gray)
#   15: "#4a90e2",  # Water (blue)
#   16: "#ffffff"   # Snow / Ice (white)
# };

NALCMS_COLORS = {
    1:  '#033e00',
    2:  '#939b71',
    3:  '#196d12',
    4:  '#1fab01',
    5:  '#5b725c',
    6:  '#6b7d2c',
    7:  '#b29d29',
    8:  '#b48833',
    9:  '#e9da5d',
    10: '#e0cd88',
    11: '#a07451',
    12: '#bad292',
    13: '#3f8970',
    14: '#6ca289',
    15: '#e6ad6a',
    16: '#a9abae',
    17: '#db2126',
    18: '#4c73a1',
    19: '#fff7fe',
}


import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from rasterio import features
from scipy import stats
import numpy as np
import geopandas as gpd
from rasterio import features

def assign_landcover_from_array(vor_polygons, landcover_array, NALCMS_COLORS, crs="EPSG:4326", transform=None):
    H, W = landcover_array.shape
    records = []
    print("Assigning landcover to polygons...")
    for poly in vor_polygons:
        # Drop z if polygon has it
        coords_2d = [(x, y) for x, y, *_ in poly.exterior.coords]
        poly2d = Polygon(coords_2d)

        mask = features.rasterize(
            [(poly2d, 1)],
            out_shape=landcover_array.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8
        ).astype(bool)


        vals = landcover_array[mask]
        if len(vals) == 0:
            continue

        # majority via bincount (faster than stats.mode)
        majority = np.bincount(vals).argmax()

        records.append({
            "geometry": poly2d,
            "landcover": int(majority),
            "stroke": NALCMS_COLORS.get(int(majority), "#FFFFFF")
        })

    gdf = gpd.GeoDataFrame(records, crs=crs)

    return gdf



import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point
from rasterio import features

def assign_landcover_elevation_from_arrays(
    vor_polygons,
    vor_centers,
    landcover_array,
    elevation_array,
    NALCMS_COLORS,
    crs="EPSG:4326",
    transform=None,
):
    """
    Assigns majority landcover, mean elevation, and mean grade (%) to each Voronoi polygon.

    Parameters
    ----------
    vor_polygons : list[Polygon]
        List of Shapely polygons (may contain z-values)
    landcover_array : np.ndarray
        2D array of landcover class IDs
    elevation_array : np.ndarray
        2D array of elevation values (same shape as landcover)
    NALCMS_COLORS : dict
        Mapping of landcover class -> color hex string
    crs : str, default 'EPSG:4326'
        Coordinate reference system for output GeoDataFrame
    transform : affine.Affine, optional
        Raster transform for spatial referencing

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with columns: geometry, landcover, stroke, elevation_mean, grade_mean
    """

    H, W = landcover_array.shape
    records = []
    print("Assigning landcover + elevation + grade to polygons...")

    # Precompute elevation gradients (pixel-based slope)
    dy, dx = np.gradient(elevation_array)
    grade_percent = np.sqrt(dx**2 + dy**2) * 100  # approximate % grade (rise/run * 100)

    for (poly, center) in zip(vor_polygons, vor_centers):
        # Drop Z if present
        coords_2d = [(x, y) for x, y, *_ in poly.exterior.coords]
        poly2d = Polygon(coords_2d)

        # Rasterize polygon to mask
        mask = features.rasterize(
            [(poly2d, 1)],
            out_shape=landcover_array.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
        ).astype(bool)

        if not mask.any():
            continue

        # --- Landcover ---
        lc_vals = landcover_array[mask]
        majority = np.bincount(lc_vals).argmax() if len(lc_vals) > 0 else -1
        stroke_color = NALCMS_COLORS.get(int(majority), "#FFFFFF")

        # --- Elevation ---
        elev_vals = elevation_array[mask]
        mean_elev = float(np.nanmean(elev_vals)) if elev_vals.size else np.nan

        # --- Grade ---
        grade_vals = grade_percent[mask]
        mean_grade = float(np.nanmean(grade_vals)) if grade_vals.size else np.nan

        # center2d = Point(center[0], center[1])
        center_z = center[2]
        records.append(
            dict(
                geometry=poly2d,
                center_lon=float(center[0]),
                center_lat=float(center[1]),
                landcover=int(majority),
                stroke=stroke_color,
                elevation_mean=mean_elev,
                grade_mean=mean_grade,
            )
        )

    gdf = gpd.GeoDataFrame(records, crs=crs)
    return gdf


import geopandas as gpd
import numpy as np
from rasterio import features
from rasterio.transform import from_bounds
from PIL import Image
import matplotlib.pyplot as plt

def generate_overlay_from_geojson(
    geojson_path,
    attr,
    color_map=None,
    bounds=None,
    shape=(1024, 1024),
    alpha=180,
    out_png="overlay.png"
):
    """
    Converts polygons in a GeoJSON into a color-coded RGBA raster (PNG).

    Parameters:
    -----------
    geojson_path : str
        Input GeoJSON file path.
    attr : str
        Attribute name to colorize by (e.g. 'landcover', 'elevation', 'grade').
    color_map : dict or None
        Optional dict mapping attribute values to hex colors.
    bounds : tuple or None
        (minx, miny, maxx, maxy). If None, use GeoJSON bounds.
    shape : tuple
        (H, W) of the output raster.
    alpha : int
        Transparency (0–255).
    out_png : str
        Output PNG file name.
    """
    # gdf = gpd.read_file(geojson_path, driver="GeoJSON")
    # if gdf.empty:
    #     raise ValueError("GeoJSON contains no features")

    # # Compute bounding box and affine transform
    # if bounds is None:
    #     bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    # minx, miny, maxx, maxy = bounds
    # width, height = shape
    # transform = from_bounds(minx, miny, maxx, maxy, width, height)

    print(f"Generating overlay from {geojson_path} using attribute '{attr}'")

    # Read GeoJSON safely
    try:
        gdf = gpd.read_file(geojson_path, engine="pyogrio")
    except Exception:
        gdf = gpd.read_file(geojson_path)

    minx, miny, maxx, maxy = gdf.total_bounds
    width, height = shape

    print(f"GeoJSON bounds: {minx}, {miny}, {maxx}, {maxy}")
    print(f"Output shape: {width} x {height}")

    # # ✅ Correct transform (Affine)
    # transform = from_bounds(minx, miny, maxx, maxy, width, height)


    from rasterio.transform import from_bounds

    minx, miny, maxx, maxy = gdf.total_bounds
    width, height = shape
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    print(transform)



    img = np.zeros((height, width, 4), dtype=np.uint8)

    for _, row in gdf.iterrows():
        val = row.get(attr)
        if val is None:
            continue

        # Determine color
        if color_map:
            hex_color = color_map.get(int(val), "#FFFFFF")
        elif "stroke" in row:
            hex_color = row["stroke"]
        else:
            # Use gradient if numeric
            cmap = plt.cm.viridis
            norm_val = (val - gdf[attr].min()) / (gdf[attr].max() - gdf[attr].min())
            rgba = (np.array(cmap(norm_val)) * 255).astype(np.uint8)
            hex_color = "#{:02x}{:02x}{:02x}".format(rgba[0], rgba[1], rgba[2])

        # Convert hex to RGB
        r, g, b = [int(hex_color[i:i+2], 16) for i in (1, 3, 5)]

        mask = features.rasterize(
            [(row.geometry, 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        ).astype(bool)

        img[mask, 0] = r
        img[mask, 1] = g
        img[mask, 2] = b
        img[mask, 3] = alpha

    Image.fromarray(img).save(out_png)
    print(f"✅ Saved {out_png} for {attr}")
    return out_png



import numpy as np
import rasterio
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
import os


def generate_overlay_from_arrays(
    landcover_array: np.ndarray,
    elevation_array: np.ndarray,
    bounds: tuple,
    nalcms_colors: dict,
    out_dir: str = "static",
    crs: str = "EPSG:4326"
):
    """
    Generate Cesium-ready overlays (PNG + .wld) from landcover and elevation arrays.

    Parameters
    ----------
    landcover_array : np.ndarray
        2D array of landcover class IDs (integers).
    elevation_array : np.ndarray
        2D array of elevation values (same shape as landcover_array).
    bounds : tuple
        (minx, miny, maxx, maxy) in geographic coordinates (lon/lat).
    nalcms_colors : dict
        Mapping from class ID to hex color (e.g., {1:"#476BA0", 2:"#D1DEF0", ...}).
    out_dir : str
        Output directory to save overlay PNGs and world files.
    crs : str
        Coordinate Reference System for georeferencing (default EPSG:4326).
    """

    os.makedirs(out_dir, exist_ok=True)

    H, W = landcover_array.shape
    assert elevation_array.shape == (H, W), "Arrays must have the same shape."

    minx, miny, maxx, maxy = bounds
    transform = from_bounds(minx, miny, maxx, maxy, W, H)

    # --- Convert landcover IDs to RGB ---
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for k, hex_color in nalcms_colors.items():
        mask = landcover_array == k
        if np.any(mask):
            rgb[mask] = np.array(
                [int(hex_color[i:i+2], 16) for i in (1, 3, 5)],
                dtype=np.uint8
            )

    # --- Write GeoTIFF (optional, for GIS validation) ---
    with rasterio.open(
        os.path.join(out_dir, "landcover_overlay.tif"),
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=3,
        dtype=rgb.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        for i in range(3):
            dst.write(rgb[..., i], i + 1)

    # --- Save PNG (for Cesium) ---
    plt.imsave(os.path.join(out_dir, "landcover_overlay.png"), rgb)

    # --- Save elevation as colorized PNG ---
    plt.imsave(
        os.path.join(out_dir, "elevation_overlay.png"),
        elevation_array,
        cmap="terrain"
    )

    # --- Write World Files (.wld) for Cesium georeferencing ---
    pixel_width = (maxx - minx) / W
    pixel_height = (miny - maxy) / H  # north-up → negative Y step
    for name in ["landcover_overlay", "elevation_overlay"]:
        with open(os.path.join(out_dir, f"{name}.wld"), "w") as f:
            f.write(f"{pixel_width}\n0.0\n0.0\n{pixel_height}\n{minx}\n{maxy}\n")

    print(f"✅ Overlays saved to '{out_dir}'")


import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from rasterio import features
from rasterio.transform import from_bounds
from PIL import Image

from scipy.spatial import Voronoi, voronoi_plot_2d

def generate_voronoi_overlay(vor, bounds,  out_png="static/voronoi_overlay.png", shape=(1024, 1024)):
    """
    Rasterize Voronoi polygons into a color-coded overlay PNG.

    Parameters
    ----------
    vor : scipy.spatial.Voronoi
        Voronoi object containing vertices and regions.
    bounds : tuple
        (minx, miny, maxx, maxy) of your AOI (in lon/lat).
    attr_values : list or np.ndarray, optional
        Per-point attribute (e.g., elevation mean or landcover id) for coloring.
        Must match number of input points (len(vor.points)).
    out_png : str
        Path to save PNG.
    shape : tuple
        (H, W) output raster size.
    """
    H, W = shape
    minx, miny, maxx, maxy = bounds
    transform = from_bounds(minx, miny, maxx, maxy, W, H)

    print(f"⏳ Generating Voronoi overlay ({W}×{H})...")
    import matplotlib.image as mpimg

    fig, ax = plt.subplots(figsize=(W/300, H/300))
    # img = mpimg.imread("static/landcover_overlay.png")
    # ax.imshow(img, origin='lower')

    voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False, line_colors="black", line_width=0.7)
    
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # invert Y to match image orientation
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout(pad=0)
    plt.savefig(out_png, dpi=300, transparent=True)
    plt.close(fig)
    print(f"✅ Saved Voronoi overlay plot: {out_png}")

    # Image.fromarray(img).save(out_png)
    # print(f"✅ Saved {out_png}")

    # Write .wld file for Cesium alignment
    pixel_width = (maxx - minx) / W
    pixel_height = (miny - maxy) / H
    with open(out_png.replace(".png", ".wld"), "w") as f:
        f.write(f"{pixel_width}\n0.0\n0.0\n{pixel_height}\n{minx}\n{maxy}\n")

    print(f"🗺️  Saved world file: {out_png.replace('.png', '.wld')}")





def generate_voronoi_seeds_overlay(vor, bounds, out_png="static/voronoi_seeds_overlay.png", shape=(1024, 1024)):
    """
    Rasterize Voronoi polygons into a color-coded overlay PNG.

    Parameters
    ----------
    vor : scipy.spatial.Voronoi
        Voronoi object containing vertices and regions.
    bounds : tuple
        (minx, miny, maxx, maxy) of your AOI (in lon/lat).
    attr_values : list or np.ndarray, optional
        Per-point attribute (e.g., elevation mean or landcover id) for coloring.
        Must match number of input points (len(vor.points)).
    out_png : str
        Path to save PNG.
    shape : tuple
        (H, W) output raster size.
    """
    H, W = shape
    minx, miny, maxx, maxy = bounds
    transform = from_bounds(minx, miny, maxx, maxy, W, H)

    print(f"⏳ Generating Voronoi overlay ({W}×{H})...")

    fig, ax = plt.subplots(figsize=(W/300, H/300))

    voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=True, point_colors="red", point_size=0.5, line_colors="black", line_width=0.7, line_alpha=0.0)

    points = vor.points
    # ax.plot(points[:, 0], points[:, 1], 'o', color='red', markersize=0.25)
    ax.scatter(points[:, 0], points[:, 1], s=1, color="red", edgecolor="white", linewidth=0.3, zorder=5)



    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # invert Y to match image orientation
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout(pad=0)
    plt.savefig(out_png, dpi=300, transparent=True)
    plt.close(fig)
    print(f"✅ Saved Voronoi overlay plot: {out_png}")

    # Image.fromarray(img).save(out_png)
    # print(f"✅ Saved {out_png}")

    # Write .wld file for Cesium alignment
    pixel_width = (maxx - minx) / W
    pixel_height = (miny - maxy) / H
    with open(out_png.replace(".png", ".wld"), "w") as f:
        f.write(f"{pixel_width}\n0.0\n0.0\n{pixel_height}\n{minx}\n{maxy}\n")

    print(f"🗺️  Saved world file: {out_png.replace('.png', '.wld')}")




import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import LineString
from rasterio.transform import from_bounds

def generate_voronoi_neighbor_overlay(vor, bounds, out_png="static/voronoi_graph_overlay.png", shape=(1024, 1024)):
    """
    Create a PNG overlay showing Voronoi neighbor edges (ridges).

    Parameters
    ----------
    points : ndarray (N, 2)
        Seed coordinates (lon, lat or projected x, y).
    bounds : tuple
        (minx, miny, maxx, maxy) bounding box of area of interest.
    out_png : str
        Path to save PNG file.
    shape : tuple
        (H, W) image size in pixels.
    """
    H, W = shape
    minx, miny, maxx, maxy = bounds
    transform = from_bounds(minx, miny, maxx, maxy, W, H)

    # vor = Voronoi(points)
    points = vor.points

    print(f"⏳ Generating Voronoi neighbor overlay ({W}×{H})...")

    fig, ax = plt.subplots(figsize=(W/300, H/300), dpi=300)

    for (i, j) in vor.ridge_points:
        p1 = points[i]
        p2 = points[j]
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color="cyan",
            linewidth=0.7,
            alpha=0.9
        )


    # voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=True, point_colors="red", point_size=0.5, line_colors="black", line_width=0.7, line_alpha=0.0)

    # Optionally draw the seed points
    # ax.scatter(points[:, 0], points[:, 1], s=6, color="red", edgecolor="white", linewidth=0.3, zorder=5)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    plt.tight_layout(pad=0)

    plt.savefig(out_png, dpi=300, transparent=True)
    plt.close(fig)
    print(f"✅ Saved Voronoi neighbor overlay PNG → {out_png}")

    # --- Save world file for Cesium/georeferencing ---
    pixel_width = (maxx - minx) / W
    pixel_height = -(maxy - miny) / H  # negative to match north-up images
    wld_path = out_png.replace(".png", ".wld")
    with open(wld_path, "w") as f:
        f.write(f"{pixel_width}\n0.0\n0.0\n{pixel_height}\n{minx}\n{maxy}\n")
    print(f"🗺️  Saved world file: {wld_path}")




def generate_georef_overlays():
    with open("./samples/map_1_data_with_voronoi.pkl", "rb") as f:
        map_1_data = pickle.load(f)
    print(map_1_data.keys())
    dem_array1 = map_1_data['dem_array']
    landcover_array1 = map_1_data['landcover_array']
    sat_image_array1 = map_1_data['sat_image_array']
    # all_paths_1 = map_1_data['all_paths']
    vor = map_1_data['voronoi']
    # rb_planner_map_1 = map_1_data['rb_planner_map']
    # start_coord_1= map_1_data['start_coord']
    # goal_coord_1= map_1_data['goal_coord']

    import rasterio
    from rasterio.transform import xy

    with rasterio.open("./samples/nasadem_bbox.tif") as src:
        transform = src.transform
        dem = src.read(1)
        crs = src.crs
        H, W = src.shape
        bounds = src.bounds




    # # Assuming you already have rb_current.voronoi for one of your decompositions
    # rb_current = rb_map_1["Boundary"]  # example key
    # vor = rb_current.voronoi


    # new_map_data = {}
    # new_map_data['dem_array'] = dem_array1
    # new_map_data['landcover_array'] = landcover_array1
    # new_map_data['sat_image_array'] = sat_image_array1
    # new_map_data['voronoi'] = vor
    # new_map_data['transform'] = transform
    # new_map_data['crs'] = str(crs)
    # new_map_data['bounds'] = (bounds.left, bounds.bottom, bounds.right, bounds.top)

    # with open("static/map_1_data_with_voronoi.pkl", "wb") as f:
    #     pickle.dump(new_map_data, f)

    generate_voronoi_neighbor_overlay(
        vor=vor,
        bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
        out_png="static/voronoi_graph_overlay.png",
        shape=dem_array1.shape
    )



    generate_overlay_from_arrays(
        landcover_array=landcover_array1,
        elevation_array=dem_array1,
        bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
        nalcms_colors=NALCMS_COLORS,
        out_dir="static",
        crs=str(crs)
    )

    generate_voronoi_overlay(
        vor=vor,
        bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
        out_png="static/voronoi_overlay.png",
        shape=dem_array1.shape
    )

    generate_voronoi_seeds_overlay(
        vor=vor,
        bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
        out_png="static/voronoi_seeds_overlay.png",
        shape=dem_array1.shape
    )




def save_voronoi_graph_geojson_old():

    # Your bounding box: [lat_min, lon_min, lat_max, lon_max]
    bbox_bound = [34.202242, -116.71692, 34.753553, -115.71606]
    lat_min, lon_min, lat_max, lon_max = bbox_bound


    with open("../map_1_data.pkl", "rb") as f:
        map_1_data = pickle.load(f)
    print(map_1_data.keys())
    dem_array1 = map_1_data['dem_array']
    landcover_array1 = map_1_data['landcover_array']
    sat_image_array1 = map_1_data['sat_image_array']
    all_paths_1 = map_1_data['all_paths']
    rb_map_1 = map_1_data['rb_map']
    rb_planner_map_1 = map_1_data['rb_planner_map']
    start_coord_1= map_1_data['start_coord']
    goal_coord_1= map_1_data['goal_coord']


    paths1 = []
    for decomp in all_paths_1:
        path, cost, time = all_paths_1[decomp]
        path_yx = np.array(path)[:, [1,0]]
        path_with_attributes = (path_yx, cost, time)
        paths1.append((decomp, path_with_attributes))
        # print(f"Decomposition: {decomp}, Cost: {cost}, Time: {time}")
        print(f"Decomposition: {decomp}, Cost: {cost}, Time: {time}")
        rb_current = rb_map_1[decomp]
        print(rb_current.all_output)
        vor = rb_current.voronoi
        print("Shape", landcover_array1.shape)


    # generate voronoi neighbor graph
    import json
    from scipy.spatial import Voronoi

    # Suppose you already have your Voronoi object
    # vor = Voronoi(points)

    edges = set()

    # Each ridge connects two points: point_indices = (p1, p2)
    for (p1, p2), ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
        # Skip ridges that extend to infinity (open)
        if -1 in ridge_vertices:
            continue
        # Add bidirectional edge (smallest index first for uniqueness)
        edges.add(tuple(sorted((int(p1), int(p2)))))

    # Convert to adjacency list
    adj = {}
    for i, j in edges:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)

    # Optional: convert to simple list of pairs [[i, j], ...] for Cesium
    adj_json = {str(k): [int(x) for x in v] for k, v in adj.items()}
    edge_list = [[int(i), int(j)] for i, j in edges]

    # ✅ Save both versions
    with open("static/voronoi_adj.json", "w") as f:
        json.dump(adj, f, indent=2)

    with open("static/voronoi_edges.json", "w") as f:
        json.dump(edge_list, f, indent=2)

    print(f"✅ Saved {len(edges)} edges connecting {len(adj)} nodes.")




import json
import pickle
import numpy as np
from shapely.geometry import mapping, LineString, Point
import geopandas as gpd
from scipy.spatial import Voronoi

def save_voronoi_graph_geojson():
    # Bounding box [lat_min, lon_min, lat_max, lon_max]
    bbox_bound = [34.202242, -116.71692, 34.753553, -115.71606]
    lat_min, lon_min, lat_max, lon_max = bbox_bound

    with open("../map_1_data.pkl", "rb") as f:
        map_1_data = pickle.load(f)

    print(map_1_data["rb_map"].keys())
    vor = map_1_data["rb_map"]["Boundary"].voronoi  # adjust key if needed
    print(f"Voronoi points: {vor.points.shape}")

    # ---------------------------------------------------
    # 1️⃣ Extract all finite edges (no infinite ridges)
    # ---------------------------------------------------
    edges = []
    for (p1, p2), ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 in ridge_vertices:
            continue  # skip infinite edges

        v1, v2 = vor.vertices[ridge_vertices]
        edges.append((p1, p2, v1, v2))

    print(f"Total finite edges: {len(edges)}")

    # ---------------------------------------------------
    # 2️⃣ Convert points to geographic coordinates
    # ---------------------------------------------------
    # Your pixel coordinates → geographic (lon/lat)
    H, W = map_1_data["dem_array"].shape
    lons = np.linspace(lon_min, lon_max, W)
    lats = np.linspace(lat_min, lat_max, H)
    # y-axis is inverted in raster coordinates
    def to_geo(y, x):
        lon = lons[int(np.clip(x, 0, W - 1))]
        lat = lats[int(np.clip(H - y - 1, 0, H - 1))]
        return (float(lon), float(lat))

    # ---------------------------------------------------
    # 3️⃣ Build GeoDataFrames for nodes & edges
    # ---------------------------------------------------
    node_geoms = []
    node_records = []
    for i, (x, y) in enumerate(vor.points):
        lon, lat = to_geo(y, x)
        node_geoms.append(Point(lon, lat))
        node_records.append({"id": int(i)})

    edge_geoms = []
    edge_records = []
    for p1, p2, v1, v2 in edges:
        # edge between two seed nodes (by their geo-coordinates)
        lon1, lat1 = to_geo(*vor.points[p1][::-1])
        lon2, lat2 = to_geo(*vor.points[p2][::-1])
        edge_geoms.append(LineString([(lon1, lat1), (lon2, lat2)]))
        edge_records.append({"source": int(p1), "target": int(p2)})

    gdf_nodes = gpd.GeoDataFrame(node_records, geometry=node_geoms, crs="EPSG:4326")
    gdf_edges = gpd.GeoDataFrame(edge_records, geometry=edge_geoms, crs="EPSG:4326")

    # ---------------------------------------------------
    # 4️⃣ Combine into one FeatureCollection
    # ---------------------------------------------------
    all_features = []

    for _, row in gdf_nodes.iterrows():
        feat = mapping(row.geometry)
        feat["properties"] = {"id": row.id, "type": "node"}
        all_features.append({"type": "Feature", "geometry": feat, "properties": feat["properties"]})

    for _, row in gdf_edges.iterrows():
        feat = mapping(row.geometry)
        feat["properties"] = {"source": row.source, "target": row.target, "type": "edge"}
        all_features.append({"type": "Feature", "geometry": feat, "properties": feat["properties"]})

    feature_collection = {"type": "FeatureCollection", "features": all_features}

    # ---------------------------------------------------
    # 5️⃣ Save combined GeoJSON
    # ---------------------------------------------------
    out_path = "static/voronoi_graph.geojson"
    with open(out_path, "w") as f:
        json.dump(feature_collection, f, indent=2)

    print(f"✅ Saved graph with {len(node_geoms)} nodes and {len(edge_geoms)} edges → {out_path}")



def save_polygon_geojson_old():

    # Your bounding box: [lat_min, lon_min, lat_max, lon_max]
    bbox_bound = [34.202242, -116.71692, 34.753553, -115.71606]
    lat_min, lon_min, lat_max, lon_max = bbox_bound


    with open("../map_1_data.pkl", "rb") as f:
        map_1_data = pickle.load(f)
    print(map_1_data.keys())
    dem_array1 = map_1_data['dem_array']
    landcover_array1 = map_1_data['landcover_array']
    sat_image_array1 = map_1_data['sat_image_array']
    all_paths_1 = map_1_data['all_paths']
    rb_map_1 = map_1_data['rb_map']
    rb_planner_map_1 = map_1_data['rb_planner_map']
    start_coord_1= map_1_data['start_coord']
    goal_coord_1= map_1_data['goal_coord']


    paths1 = []
    for decomp in all_paths_1:
        path, cost, time = all_paths_1[decomp]
        path_yx = np.array(path)[:, [1,0]]
        path_with_attributes = (path_yx, cost, time)
        paths1.append((decomp, path_with_attributes))
        # print(f"Decomposition: {decomp}, Cost: {cost}, Time: {time}")
        print(f"Decomposition: {decomp}, Cost: {cost}, Time: {time}")
        rb_current = rb_map_1[decomp]
        print(rb_current.all_output)
        vor = rb_current.voronoi
        print("Shape", landcover_array1.shape)


    import rasterio
    from rasterio.transform import xy

    with rasterio.open("../nasadem_bbox.tif") as src:
        transform = src.transform
        dem = src.read(1)
        crs = src.crs
        H, W = src.shape


    from shapely.geometry import box

    H, W = dem_array1.shape
    bbox = box(0, 0, W, H)  # DEM extent as a bounding box polygon

    clipped_polygons = []
    lc_classes = []
    colors = []

    records = []
    for region_index in vor.point_region:
        region = vor.regions[region_index]
        if not -1 in region and region != []:
            poly = Polygon([vor.vertices[i] for i in region])
            clipped = poly.intersection(bbox)  # clip to DEM bounds
            if not clipped.is_empty:
                # clipped_polygons.append(clipped)

                coords = []
                lc_curr = []
                for x_pix, y_pix in np.array(clipped.exterior.coords):
                    # pixel → world (lon, lat)
                    lon, lat = transform * (x_pix, y_pix)
                    # sample DEM height (nearest pixel)
                    col, row = map(int, ~transform * (lon, lat))
                    if (0 <= row < dem_array1.shape[0]) and (0 <= col < dem_array1.shape[1]):
                        z = float(dem_array1[row, col])
                    else:
                        z = 0.0
                    coords.append((lon, lat, z))
                    # lc_curr.append(landcover_array1[row, col])
                clipped_polygons.append(Polygon(coords))

    
    print("Start assigning landcover...")
    # gdf = assign_landcover_from_array(clipped_polygons, landcover_array1, NALCMS_COLORS, crs, transform=transform)
    gdf = assign_landcover_elevation_from_arrays(clipped_polygons, landcover_array1, dem_array1, NALCMS_COLORS, crs, transform=transform)
    gdf = gdf.to_crs("EPSG:4326")

    print("Saving voronoi_3d.geojson...")

    gdf.to_file("static/voronoi_3d.geojson", driver="GeoJSON")
    print("Saved voronoi_3d.geojson")

    return



def save_polygon_geojson():

    # Your bounding box: [lat_min, lon_min, lat_max, lon_max]
    bbox_bound = [34.202242, -116.71692, 34.753553, -115.71606]
    lat_min, lon_min, lat_max, lon_max = bbox_bound


    with open("../map_1_data.pkl", "rb") as f:
        map_1_data = pickle.load(f)
    print(map_1_data.keys())
    dem_array1 = map_1_data['dem_array']
    landcover_array1 = map_1_data['landcover_array']
    sat_image_array1 = map_1_data['sat_image_array']
    all_paths_1 = map_1_data['all_paths']
    rb_map_1 = map_1_data['rb_map']
    rb_planner_map_1 = map_1_data['rb_planner_map']
    start_coord_1= map_1_data['start_coord']
    goal_coord_1= map_1_data['goal_coord']


    paths1 = []
    for decomp in all_paths_1:
        path, cost, time = all_paths_1[decomp]
        path_yx = np.array(path)[:, [1,0]]
        path_with_attributes = (path_yx, cost, time)
        paths1.append((decomp, path_with_attributes))
        # print(f"Decomposition: {decomp}, Cost: {cost}, Time: {time}")
        print(f"Decomposition: {decomp}, Cost: {cost}, Time: {time}")
        rb_current = rb_map_1[decomp]
        print(rb_current.all_output)
        vor = rb_current.voronoi
        print("Shape", landcover_array1.shape)


    import rasterio
    from rasterio.transform import xy

    with rasterio.open("../nasadem_bbox.tif") as src:
        transform_ = src.transform
        dem = src.read(1)
        crs = src.crs
        H, W = src.shape


    from shapely.geometry import box

    H, W = dem_array1.shape
    bbox = box(0, 0, W, H)  # DEM extent as a bounding box polygon

    # ---------------------------------------------------
    # 2️⃣ Convert points to geographic coordinates
    # ---------------------------------------------------
    # Your pixel coordinates → geographic (lon/lat)

    bbox_bound = [34.202242, -116.71692, 34.753553, -115.71606]
    lat_min, lon_min, lat_max, lon_max = bbox_bound


    H, W = map_1_data["dem_array"].shape
    lons = np.linspace(lon_min, lon_max, W)
    lats = np.linspace(lat_min, lat_max, H)
    # y-axis is inverted in raster coordinates
    def to_geo(y, x):
        lon = lons[int(np.clip(x, 0, W - 1))]
        lat = lats[int(np.clip(H - y - 1, 0, H - 1))]
        return (float(lon), float(lat))

    def to_pixel(lon, lat):
        # Find nearest pixel coordinates for given lon/lat
        col = int(np.clip((lon - transform_.c) / transform_.a, 0, W - 1))
        row = int(np.clip((transform_.f - lat) / -transform_.e, 0, H - 1))
        return (row, col)





    clipped_polygons = []
    lc_classes = []
    colors = []

    records = []
    voronoi_points_3d = []
    for i, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        point = vor.points[i]
        lon, lat = to_geo(point[1], point[0])
        col, row = map(int, ~transform_ * (lon, lat))
        if (0 <= row < dem_array1.shape[0]) and (0 <= col < dem_array1.shape[1]):
            z = float(dem_array1[row, col])
        else:
            z = 0.0
        voronoi_points_3d.append((lon, lat, z))
        if not -1 in region and region != []:
            poly = Polygon([vor.vertices[i] for i in region])
            clipped = poly.intersection(bbox)  # clip to DEM bounds
            if not clipped.is_empty:
                # clipped_polygons.append(clipped)

                coords = []
                lc_curr = []
                for x_pix, y_pix in np.array(clipped.exterior.coords):
                    # pixel → world (lon, lat)
                    lon, lat = to_geo(y_pix, x_pix) # transform * (x_pix, y_pix)
                    # sample DEM height (nearest pixel)
                    col, row = map(int, ~transform_ * (lon, lat))
                    if (0 <= row < dem_array1.shape[0]) and (0 <= col < dem_array1.shape[1]):
                        z = float(dem_array1[row, col])
                    else:
                        z = 0.0
                    coords.append((lon, lat, z))
                    # lc_curr.append(landcover_array1[row, col])
                clipped_polygons.append(Polygon(coords))

    
    print("Start assigning landcover...")
    # gdf = assign_landcover_from_array(clipped_polygons, landcover_array1, NALCMS_COLORS, crs, transform=transform)
    gdf = assign_landcover_elevation_from_arrays(clipped_polygons, voronoi_points_3d, landcover_array1, dem_array1, NALCMS_COLORS, crs, transform=transform_)
    gdf = gdf.to_crs("EPSG:4326")

    print("Saving voronoi_3d.geojson...")

    gdf.to_file("static/voronoi_3d.geojson", driver="GeoJSON")
    print("Saved voronoi_3d.geojson")

    return






# if __name__ == "__main__":
def old_main():

    # Your bounding box: [lat_min, lon_min, lat_max, lon_max]
    bbox_bound = [34.202242, -116.71692, 34.753553, -115.71606]
    lat_min, lon_min, lat_max, lon_max = bbox_bound


    with open("map_1_data.pkl", "rb") as f:
        map_1_data = pickle.load(f)
    print(map_1_data.keys())
    dem_array1 = map_1_data['dem_array']
    landcover_array1 = map_1_data['landcover_array']
    sat_image_array1 = map_1_data['sat_image_array']
    all_paths_1 = map_1_data['all_paths']
    rb_map_1 = map_1_data['rb_map']
    rb_planner_map_1 = map_1_data['rb_planner_map']
    start_coord_1= map_1_data['start_coord']
    goal_coord_1= map_1_data['goal_coord']


    paths1 = []
    for decomp in all_paths_1:
        path, cost, time = all_paths_1[decomp]
        path_yx = np.array(path)[:, [1,0]]
        path_with_attributes = (path_yx, cost, time)
        paths1.append((decomp, path_with_attributes))
        # print(f"Decomposition: {decomp}, Cost: {cost}, Time: {time}")
        print(f"Decomposition: {decomp}, Cost: {cost}, Time: {time}")
        rb_current = rb_map_1[decomp]
        print(rb_current.all_output)
        vor = rb_current.voronoi
        print("Shape", landcover_array1.shape)

    import rasterio
    from rasterio.transform import xy

    with rasterio.open("nasadem_bbox.tif") as src:
        transform = src.transform
        crs = src.crs
        H, W = src.shape

    def pixel_to_geo(polygon, transform):
        """Convert shapely polygon in pixel coords to geographic coords using raster transform."""
        geo_coords = [xy(transform, y, x) for x, y in polygon.exterior.coords]  
        return Polygon(geo_coords)



    from shapely.geometry import box

    H, W = dem_array1.shape
    bbox = box(0, 0, W, H)  # DEM extent as a bounding box polygon

    clipped_polygons = []
    for region_index in vor.point_region:
        region = vor.regions[region_index]
        if not -1 in region and region != []:
            poly = Polygon([vor.vertices[i] for i in region])
            clipped = poly.intersection(bbox)  # clip to DEM bounds
            if not clipped.is_empty:
                clipped_polygons.append(clipped)

    geo_polygons = [pixel_to_geo(poly, transform) for poly in clipped_polygons]


    print(f"Number of clipped polygons: {len(geo_polygons)}")

    gdf_voronoi = gpd.GeoDataFrame(geometry=geo_polygons, crs=crs)
    gdf_voronoi = gdf_voronoi.to_crs("EPSG:4326")
    gdf_voronoi.to_file("static/voronoi.geojson", driver="GeoJSON")


    # Example usage
    tasks = np.array([[100, 1900], 
                    [3500, 750], 
                    [1600, 1000], #Vehicle Depot
                    [300, 400], 
                    [3200, 800],
                    [3100, 1500],
                    [500, 1100],
                    [1300, 1750],
                    [2450, 1800], 
                    [3000, 300],
                    [3500, 1200] #Human Depot
                    ])
    # Generate all start–goal pairs
    pairs = list(permutations(tasks, 2))  # order matters, excludes same=start=goal

    # Split into start and goal lists
    start_pts = [p[0] for p in pairs]
    goal_pts = [p[1] for p in pairs]

    print(f"Number of pairs: {len(pairs)}")



    grade_image = get_grade_image(dem_array1)

    # import geopandas as gpd
    # import matplotlib.pyplot as plt
    # from shapely.geometry import box
    # import contextily as ctx



    # # Create a bounding box polygon (in lon/lat order)
    # bbox = box(lon_min, lat_min, lon_max, lat_max)

    # # Put into GeoDataFrame with WGS84 CRS (EPSG:4326)
    # gdf_bbox = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326")

    # # Project to Web Mercator (needed for basemap tiles)
    # gdf_bbox_web = gdf_bbox.to_crs(epsg=3857)

    # # Plot
    # fig, ax = plt.subplots(figsize=(8, 8))
    # gdf_bbox_web.boundary.plot(ax=ax, edgecolor="red", linewidth=2)

    # # Add basemap (satellite tiles)
    # ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
    # # ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # ax.set_axis_off()
    # plt.savefig("vis_results/bbox_map.png", bbox_inches='tight', dpi=300)



    # # Example usage
    # animator = TaskZoomAnimator(
    #     tasks=tasks,
    #     voronoi_gdf=gdf_voronoi,
    #     grade_image=grade_image,
    #     extent=[0, W, 0, H],
    #     depots={"vehicle":2, "human":10}
    # )

    # task_sequence = [3, 7]   # zoom into tasks 3 and 7
    # animator.animate_zoom_sequence(task_sequence[:1], zoom_scales=np.linspace(1.0, 0.1, 20))
    # animator.animate_pan_sequence(task_sequence, n_steps=40, zoom_scale=0.1)
    # animator.save_gif("vis_results/zoom_example.gif", fps=10)


    dem_path = "nasadem_bbox.tif"   # DEM for your bounding box
    tasks = np.array([
        [-116.6, 34.3],
        [-116.2, 34.5],
        [-115.9, 34.6]
    ])

    # animator3d = TaskZoomAnimator3D(dem_path=dem_path, tasks=tasks)
    # animator3d.animate_pan_sequence([0,1,2], n_steps=40, zoom_height=2000, filename="terrain_zoom.gif")


    tasks = np.array([
        [-116.6, 34.3],
        [-116.2, 34.5],
        [-115.9, 34.6]
    ])

    animator = TaskZoomAnimatorWebGL("nasadem_bbox.tif", tasks=tasks)
    animator.animate_pan_sequence([0, 1, 2], n_steps=40, filename="terrain_zoom.html")


from flask import Flask, render_template_string
import numpy as np
import rasterio
import plotly.graph_objects as go

app = Flask(__name__)





from rasterio.windows import from_bounds as from_bounds_windows

def crop_dem(dem_path, bounds):
    with rasterio.open(dem_path) as src:
        window = from_bounds_windows(*bounds, transform=src.transform)
        dem = src.read(1, window=window).astype(float)
        dem[dem < -1000] = np.nan
        return dem, src.transform

@app.route("/2")
def index2():
    dem_path = "nasadem_bbox.tif"

    from rasterio.windows import from_bounds  as from_bounds_windows

    # minx, miny, maxx, maxy = [34.202242, -116.71692, 34.753553, -115.71606]

    with rasterio.open(dem_path) as src:
        # window = from_bounds_windows(minx, miny, maxx, maxy, src.transform)
        # dem = src.read(1, window=window).astype(float)
        dem = src.read(1).astype(float)
        print("DEM shape:", dem.shape)
        print("DEM stats:", np.nanmin(dem), np.nanmax(dem))
        # print("Unique nodata?", np.unique(dem[:100,:100]))  # peek sample

        # dem = src.read(1).astype(float)
        dem[dem < -1000] = np.nan  # mask nodata
        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    nrows, ncols = dem.shape
    xs = np.linspace(extent[0], extent[1], ncols)
    ys = np.linspace(extent[2], extent[3], nrows)
    xx, yy = np.meshgrid(xs, ys)

    fig = go.Figure(go.Surface(
        z=dem, x=xx, y=yy,
        colorscale="earth", showscale=False, opacity=0.95
    ))

    # dem_small = dem#[::50, ::50]  # maybe 100x100 max
    # fig = go.Figure(go.Surface(z=dem_small, colorscale="earth"))


    fig.update_layout(
        scene=dict(xaxis=dict(visible=False),
                   yaxis=dict(visible=False),
                   zaxis=dict(visible=False)),
        margin=dict(l=0, r=0, t=0, b=0),
        height=700
    )

    fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    template = """
    <!DOCTYPE html>
    <html>
      <head><title>3D DEM</title><meta charset="utf-8"></head>
      <body>
        <h2 style="text-align:center;">DEM Surface</h2>
        {{ fig_html | safe }}
      </body>
    </html>
    """
    return render_template_string(template, fig_html=fig_html)

def downsample(dem, max_size=500):
    step = max(1, max(dem.shape) // max_size)
    return dem[::step, ::step]


@app.route("/dem")
def index():

    with open("map_1_data.pkl", "rb") as f:
        map_1_data = pickle.load(f)
    print(map_1_data.keys())
    dem_array1 = map_1_data['dem_array']
    landcover_array1 = map_1_data['landcover_array']
    sat_image_array1 = map_1_data['sat_image_array']
    all_paths_1 = map_1_data['all_paths']
    rb_map_1 = map_1_data['rb_map']
    rb_planner_map_1 = map_1_data['rb_planner_map']
    start_coord_1= map_1_data['start_coord']
    goal_coord_1= map_1_data['goal_coord']

    sat_img = sat_image_array1

    bounds = (-116.71692, 34.202242, -115.71606, 34.753553)  # your earlier box
    dem, transform = crop_dem("nasadem_bbox.tif", bounds)

    max_size = 1000
    dem_small = downsample(dem, max_size=max_size)

    nrows, ncols = dem.shape
    xs = np.arange(ncols) * transform.a + transform.c   # pixel_col * 30 + xmin
    ys = np.arange(nrows) * transform.e + transform.f   # pixel_row * -30 + ymax
    xx, yy = np.meshgrid(xs, ys)
    step = max(1, max(dem.shape) // max_size)

    # fig = go.Figure(data=[
    #     go.Surface(
    #         z=dem_small,
    #         x=xx[::step, ::step],
    #         y=yy[::step, ::step],
            
    #         colorscale="earth"
    #     )
    # ])
    smooth = True
    upscale = 1

    from scipy.ndimage import zoom

    cell_size=30.0
    z_exaggeration = None
    # Optional resampling
    if smooth and upscale > 1:
        dem = zoom(dem, upscale, order=3)  # cubic interp
        sat_img = zoom(sat_img, (upscale, upscale, 1), order=1)
        cell_size = cell_size / upscale

    H, W = dem.shape
    x = np.arange(0, W * cell_size, cell_size)
    y = np.arange(0, H * cell_size, cell_size)
    X, Y = np.meshgrid(x, y)

    # # Normalize satellite image to [0,1]
    sat_img = sat_img.astype(np.float32)
    if sat_img.max() > 1.0:
        sat_img /= 255.0

    # Auto vertical exaggeration
    if z_exaggeration is None:
        z_range = np.nanmax(dem) - np.nanmin(dem)
        xy_range = max(W * cell_size, H * cell_size)
        z_exaggeration = xy_range / z_range / 5.0 if z_range > 0 else 1.0

    # figsize = (8, 6)
    # # Plot
    # fig = plt.figure(figsize=figsize)
    # ax = fig.add_subplot(111, projection="3d")
    # ax.plot_surface(X, Y, dem * z_exaggeration, facecolors=sat_img,
    #                 rstride=3, cstride=3,
    #                 linewidth=0, antialiased=False,
    #                 shade=False, edgecolor='none')


    # fig = go.Figure(data=[
    #     go.Surface(
    #         z=dem * z_exaggeration,
    #         x=X,#[::step, ::step],
    #         y=Y,#[::step, ::step],
            
    #         # colorscale="earth", 
    #         colorscale=sat_img,


    #     )
    # ])

    sat_gray = np.dot(sat_img[...,:3], [0.299, 0.587, 0.114])  # luminance

    fig = go.Figure(data=[
        go.Surface(
            z=dem_small * z_exaggeration,
            x=X[::step, ::step],
            y=Y[::step, ::step],
            surfacecolor=sat_gray[::step, ::step],   # <-- use image intensity
            colorscale="gray",       # colormap for mapping grayscale
            cmin=0,
            cmax=1
        )
    ])

    # H, W, _ = sat_img.shape
    # colors_hex = np.array([
    #     ['rgb({},{},{})'.format(r,g,b) for r,g,b in row]
    #     for row in sat_img.astype(np.uint8)
    # ])

    # # Make an integer index for each pixel
    # color_idx = np.arange(H*W).reshape(H, W)

    # fig = go.Figure(go.Surface(
    #     z=dem_small * z_exaggeration,
    #     x=X[::step, ::step],
    #     y=Y[::step, ::step],
    #     surfacecolor=color_idx[::step, ::step],   # use index grid
    #     colorscale=[(i/(H*W), c) for i,c in enumerate(colors_hex.flatten())],
    #     showscale=False
    # ))


    # fig = go.Figure(go.Surface(z=dem_small, colorscale="earth"))
    fig.update_layout(
        # autosize=True,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)#dict(title="Elevation (m)")
        )
    )
    return fig.to_html(full_html=True, include_plotlyjs="cdn")




import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def reproject_dem_to_meters(dem_path, out_path):
    with rasterio.open(dem_path) as src:
        # If input CRS is geographic, convert to UTM
        if src.crs.is_geographic:
            lon_center = (src.bounds.left + src.bounds.right) / 2
            lat_center = (src.bounds.top + src.bounds.bottom) / 2
            utm_zone = int((lon_center + 180) / 6) + 1
            if lat_center >= 0:
                dst_crs = f"EPSG:{32600 + utm_zone}"  # Northern hemisphere
            else:
                dst_crs = f"EPSG:{32700 + utm_zone}"  # Southern hemisphere
        else:
            dst_crs = src.crs

        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "dtype": src.dtypes[0],
            "nodata": src.nodata
        })

        with rasterio.open(out_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear
                )
    return out_path




import cesiumpy

from flask import Flask
import cesiumpy

app = Flask(__name__)

@app.route("/cesiumw")
def index():
    v = cesiumpy.Viewer()

    # Add rectangle entity (bounding box)
    rect = cesiumpy.Rectangle([-116.71692, 34.202242, -115.71606, 34.753553])
    rect = cesiumpy.Cartesian4.fromDegrees(-116.71692, 34.202242, -115.71606, 34.753553)
    v.camera.flyTo(rect)

    v.camera.flyTo(
        cesiumpy.Cartesian4.fromDegrees(
            -116.71692, 34.202242, -115.71606, 34.753553
        )
    )



    # Add a point inside the rectangle
    pos = cesiumpy.Cartesian3.fromDegrees(-116.7, 34.2, 1000)
    point = cesiumpy.Point(position=pos, color="red", pixelSize=12)
    # v.add(point)

    # Generate HTML/JS
    return v.to_html()


from flask import Flask, render_template_string

app = Flask(__name__)

@app.route("/cesium4")
def index():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
    <script src="https://cesium.com/downloads/cesiumjs/releases/1.111/Build/Cesium/Cesium.js"></script>
    <link href="https://cesium.com/downloads/cesiumjs/releases/1.111/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
    <style>
        html, body, #cesiumContainer {
        width:100%; height:100%; margin:0; padding:0; overflow:hidden;
        }
    </style>
    </head>
    <body>
    <div id="cesiumContainer"></div>
    <script>
        Cesium.Ion.defaultAccessToken = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI5ZDFkYTk0Ny0yZTU0LTRlZGMtYTQ0Ni1kOGIxZjA5Y2VlYmIiLCJpZCI6MzQ2OTE2LCJpYXQiOjE3NTk1MDkzODl9.BQVyOQvlP-jAGxrYzHnqdziLHjg5pio2V7ke_HiA6D0";

        const viewer = new Cesium.Viewer('cesiumContainer', {
        terrainProvider: Cesium.createWorldTerrain()
        });

        viewer.camera.flyTo({
        destination: Cesium.Rectangle.fromDegrees(
            -116.71692, 34.202242, -115.71606, 34.753553
        )
        });

        viewer.entities.add({
        position: Cesium.Cartesian3.fromDegrees(-116.7, 34.2, 1000),
        point: { pixelSize: 12, color: Cesium.Color.RED }
        });
    </script>
    </body>
    </html>

    """
    return render_template_string(html)

if __name__ == "__main__":
    app.run(debug=True)



if __name__ == "__main__df":


    viewer = cesiumpy.Viewer()
    # Cartesian3 takes lon, lat, height
    pos = cesiumpy.Cartesian3.fromDegrees(-116.7, 34.2, 1000)

    # PointGraphics is created via Point
    p = cesiumpy.Point(position=pos, color="red", pixelSize=12)
    viewer.entities.add(p)

    viewer.render('map.html')




    # reproject_dem_to_meters("nasadem_bbox.tif", "nasadem_utm.tif")


    # app.run(host="0.0.0.0", port=8050, debug=True)


