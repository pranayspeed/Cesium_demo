import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Color and label dictionaries
nalcms_palette = {
    1:  '#033e00', 2:  '#939b71', 3:  '#196d12', 4:  '#1fab01', 5:  '#5b725c',
    6:  '#6b7d2c', 7:  '#b29d29', 8:  '#b48833', 9:  '#e9da5d', 10: '#e0cd88',
    11: '#a07451', 12: '#bad292', 13: '#3f8970', 14: '#6ca289', 15: '#e6ad6a',
    16: '#a9abae', 17: '#db2126', 18: '#4c73a1', 19: '#fff7fe',
}
nalcms_classes = {
    1:  'Temperate or sub-polar needleleaf forest',
    2:  'Sub-polar taiga needleleaf forest',
    3:  'Tropical or sub-tropical broadleaf evergreen forest',
    4:  'Tropical or sub-tropical broadleaf deciduous forest',
    5:  'Temperate or sub-polar broadleaf deciduous forest',
    6:  'Mixed forest',
    7:  'Tropical or sub-tropical shrubland',
    8:  'Temperate or sub-polar shrubland',
    9:  'Tropical or sub-tropical grassland',
    10: 'Temperate or sub-polar grassland',
    11: 'Sub-polar or polar shrubland-lichen-moss',
    12: 'Sub-polar or polar grassland-lichen-moss',
    13: 'Sub-polar or polar barren-lichen-moss',
    14: 'Wetland',
    15: 'Cropland',
    16: 'Barren land',
    17: 'Urban and built-up',
    18: 'Water',
    19: 'Snow and ice',
}


import pickle 

with open("new_map_1_data.pkl", "rb") as f:
    map_1_data = pickle.load(f)
    
print(map_1_data.keys())
dem_array1 = map_1_data['dem_array']
landcover_array1 = map_1_data['landcover_array']
sat_image_array1 = map_1_data['sat_image_array']

import matplotlib.pyplot as plt
import numpy as np

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

task_symbols = ["Symbols/Telecommunications.png",
                "Symbols/EngineeringEquipment.png",
                "Symbols/MilitaryBase.png",
                "Symbols/MissileSystem.png",
                "Symbols/Telecommunications.png",
                "Symbols/MissileSystem.png",
                "Symbols/LightArmoredRecon.png",
                "Symbols/EngineeringEquipment.png",
                "Symbols/MissileSystem.png",
                "Symbols/LightArmoredRecon.png",
                "Symbols/MilitaryBase.png"]

agents_to_paths_file = {
    "HeavyVehicle": "map_1_edges_with_attributes_HeavyVehicleModelv2.pkl",
    "LightVehicle": "map_1_edges_with_attributes_LightVehicleModelv2.pkl",
    "HeavyHuman": "map_1_edges_with_attributes_HeavyHumanModelv2.pkl",
    "LightHuman": "map_1_edges_with_attributes_LightHumanModelv2.pkl",
}

import pickle

heavy_human_file = agents_to_paths_file["HeavyHuman"]
print(heavy_human_file)

light_human_file = agents_to_paths_file["LightHuman"]
print(light_human_file)

with open(heavy_human_file, "rb") as f:
    heavy_human = pickle.load(f)

with open(light_human_file, "rb") as f:
    light_human = pickle.load(f)



heavy_vehicle_file = agents_to_paths_file["HeavyVehicle"]
print(heavy_vehicle_file)
light_vehicle_file = agents_to_paths_file["LightVehicle"]
print(light_vehicle_file)

with open(heavy_vehicle_file, "rb") as f:
    heavy_vehicle = pickle.load(f)

with open(light_vehicle_file, "rb") as f:
    light_vehicle = pickle.load(f)
    


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import Image, ImageChops
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

def load_and_crop_white(symbol_path, size=(40, 40), white_thresh=220):
    img = Image.open(symbol_path).convert("RGBA")
    arr = np.array(img)
    # Create mask for white pixels
    white_mask = np.all(arr[..., :3] >= white_thresh, axis=-1)
    arr[..., 3][white_mask] = 0  # Set alpha to 0 for white pixels
    img = Image.fromarray(arr)
    img = img.resize(size, Image.LANCZOS)
    return np.array(img)


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
    # return grade, seg_corrected, bins_reduced, counts_valid


list_of_paths_dict = {}

list_of_paths_dict["HeavyHuman"] =[]
for edge in heavy_human:
    path = edge['path']
    list_of_paths_dict["HeavyHuman"].append(path)

list_of_paths_dict["LightHuman"] =[]
for edge in light_human:
    path = edge['path']
    list_of_paths_dict["LightHuman"].append(path)
    

list_of_paths_dict["HeavyVehicle"] =[]
list_of_paths_dict["LightVehicle"] =[]
for edge in heavy_vehicle:
    path = edge['path']
    list_of_paths_dict["HeavyVehicle"].append(path)

for edge in light_vehicle:
    path = edge['path']
    list_of_paths_dict["LightVehicle"].append(path)
    

# plot_paths_on_dem_with_grade(dem_array1, tasks, paths=list_of_paths_dict["HeavyHuman"], task_symbols=task_symbols)

# Example usage
tasks = np.array([[100, 1900],                  #Telecom  
                  [3500, 750],                  #Engineering equipment
                  [1600, 1000], #Vehicle Depot
                  [300, 400],                   #Missile System
                  [3200, 800],                  #Telecom
                  [3100, 1500],                 #Missile System
                  [500, 1100],                  #Light armored recon
                  [1300, 1750],                 #Engineering equipment
                  [2450, 1800],                 #Missile
                  [3000, 300],                  #Light armored recon           
                  [3500, 1200] #Human Depot
                  ])


# COA 1

ag_waypoints = {}
ag_waypoints["HeavyVehicle"] = [[1600, 1000], [100, 1900], [300, 400], [1300, 1750], [2450, 1800], [1600, 1000]]
ag_waypoints["LightVehicle"] = [[1600, 1000], [500, 1100], [1600, 1000]]
ag_waypoints["HeavyHuman"] = [[3500, 1200], [3200, 800], [3100, 1500], [3500, 1200]]
ag_waypoints["LightHuman"] = [[3500, 1200], [3500, 750], [3000, 300], [3500, 1200]]


agent_dicts = []

for key in ag_waypoints.keys():
    dic = {}
    waypoints = []
    for i in range(1, len(ag_waypoints[key])):
        start = ag_waypoints[key][i-1]
        end = ag_waypoints[key][i]
        paths = list_of_paths_dict[key]
        for path in paths:
            if list(path[0]) == start and list(path[-1]) == end:
                break
        for point in path:
            waypoints.append(point)
    dic["waypoints"] = waypoints
    if key == "LightVehicle":
        dic["speed"] = 5
        dic["symbol"] = "Symbols\LightArmoredRecon.png"
    elif key == "HeavyVehicle":
        dic["speed"] = 3
        dic["symbol"] = "Symbols\WheeledArmored.png"
    elif key == "LightHuman":
        dic["speed"] = 1.5
        dic["symbol"] = "Symbols\LightInfantry.png"
    elif key == "HeavyHuman":
        dic["speed"] = 1.5
        dic["symbol"] = "Symbols\ArmoredInfantry.png"
    
    agent_dicts.append(dic)
        
        
# COA 2

ag_waypoints = {}
ag_waypoints["HeavyVehicle"] = [[1600, 1000], [100, 1900], [2450, 1800], [1600, 1000]]
ag_waypoints["LightVehicle"] = [[1600, 1000], [300, 400], [500, 1100], [1300, 1750], [1600, 1000]]
ag_waypoints["HeavyHuman"] = [[3500, 1200], [3500, 750], [3500, 1200]]
ag_waypoints["LightHuman"] = [[3500, 1200], [3200, 800], [3100, 1500], [3000, 300], [3500, 1200]]


agent_dicts = []

for key in ag_waypoints.keys():
    dic = {}
    waypoints = []
    for i in range(1, len(ag_waypoints[key])):
        start = ag_waypoints[key][i-1]
        end = ag_waypoints[key][i]
        paths = list_of_paths_dict[key]
        for path in paths:
            if list(path[0]) == start and list(path[-1]) == end:
                break
        for point in path:
            waypoints.append(point)
    dic["waypoints"] = waypoints
    if key == "LightVehicle":
        dic["speed"] = 5
        dic["symbol"] = "Symbols\LightArmoredRecon.png"
    elif key == "HeavyVehicle":
        dic["speed"] = 3
        dic["symbol"] = "Symbols\WheeledArmored.png"
    elif key == "LightHuman":
        dic["speed"] = 1.5
        dic["symbol"] = "Symbols\LightInfantry.png"
    elif key == "HeavyHuman":
        dic["speed"] = 1.5
        dic["symbol"] = "Symbols\ArmoredInfantry.png"
    
    agent_dicts.append(dic)


from matplotlib.animation import FuncAnimation

# Example agent data
# agents = [
#     {
#         "waypoints": np.array([[100, 1900], [500, 1100], [1300, 1750]]),
#         "speed": 2.0,  # units per frame
#         "symbol": "Symbols/ArmoredInfantry.png"
#     },
#     # Add more agents as needed
# ]

# Interpolate agent paths for smooth animation
def interpolate_path(waypoints, speed):
    points = [waypoints[0]]
    for i in range(1, len(waypoints)):
        start, end = waypoints[i-1], waypoints[i]
        dist = np.linalg.norm(end - start)
        steps = max(int(dist // speed), 1)
        for t in np.linspace(0, 1, steps, endpoint=False)[1:]:
            points.append(start + t * (end - start))
        points.append(end)
    return np.array(points)

for agent in agent_dicts:
    agent["path"] = interpolate_path(agent["waypoints"], agent["speed"])
    agent["img"] = load_and_crop_white(agent["symbol"], size=(60, 60))

fig, ax = plt.subplots(figsize=(16, 9))
# ... plot your DEM and background as before ...


dx, dy = 0.00026949458523585647, 0.00026949458523585647
dem_filled = np.where(np.isnan(dem_array1), np.nanmedian(dem_array1[~np.isnan(dem_array1)]) if np.any(~np.isnan(dem_array1)) else 0.0, dem_array1)

hs = hillshade(dem_filled, dx=dx, dy=dy, azimuth=315.0, altitude=45.0)


seg_display, bins_reduced, cmap_final = compute_segmented_grade_with_outliers(dem_array1)

fig, ax = plt.subplots(figsize=(10, 9))
ax.imshow(hs, cmap="gray", origin="upper")  # hillshade background
im = ax.imshow(seg_display, cmap=cmap_final, origin="upper", alpha=0.6)  # overlay

# Plot tasks if provided
if tasks is not None and len(tasks) > 0:
    ax.scatter(tasks[:, 0], tasks[:, 1], s=80, c="black", marker="x", label="Tasks")
    
if tasks is not None and len(tasks) > 0:
    ax.scatter(tasks[:, 0], tasks[:, 1], s=80, c="black", marker="x", label="Tasks")
    # Overlay PNG symbols if provided
    if task_symbols is not None: # and len(task_symbols) == len(tasks):
        for (x, y), symbol_path in zip(tasks, task_symbols):
            try:
                img = load_and_crop_white(symbol_path, size=(70, 70))
                imagebox = OffsetImage(img, zoom=0.5)  # Adjust zoom as needed
                ab = AnnotationBbox(imagebox, (x, y), frameon=False)
                ax.add_artist(ab)
            except Exception as e:
                print(f"Could not load symbol {symbol_path}: {e}")

# # Plot paths if provided
# if paths is not None:
#     for i, path in enumerate(paths):
#         path = np.array(path)
#         if len(path) > 0:
#             ax.plot(path[:, 0], path[:, 1], '-', lw=2, label=f"Path {i+1}")

plt.xticks([])
plt.yticks([])
plt.tight_layout()
# plt.show()




# Prepare animated artists
agent_artists = []
agent_lines = []

for agent in agent_dicts:
    imagebox = OffsetImage(agent["img"], zoom=0.5)
    ab = AnnotationBbox(imagebox, agent["path"][0], frameon=False)
    ab.set_animated(True)  # Ensure the artist is marked as animated for blitting
    ax.add_artist(ab)
    agent_artists.append(ab)
    (line,) = ax.plot([], [], '-', lw=2)
    line.set_animated(True)
    agent_lines.append(line)


def animate(frame):
    # Remove and re-add agent symbols at new positions
    for i, agent in enumerate(agent_dicts):
        path = agent["path"]
        idx = min(frame, len(path)-1)
        # Remove the old artist
        try:
            agent_artists[i].remove()
        except Exception:
            pass
        # Create a new AnnotationBbox at the new position
        imagebox = OffsetImage(agent["img"], zoom=0.5)
        ab = AnnotationBbox(imagebox, path[idx], frameon=False)
        ab.set_animated(True)
        ax.add_artist(ab)
        agent_artists[i] = ab
        agent_lines[i].set_data(path[:idx+1, 0], path[:idx+1, 1])
    return agent_artists + agent_lines

max_frames = max(len(agent["path"]) for agent in agent_dicts)
print(max_frames)
ani = FuncAnimation(fig, animate, frames=max_frames, interval=10, blit=True)
plt.show()