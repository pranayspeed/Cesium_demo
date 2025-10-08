from flask import Flask, send_from_directory, render_template_string
import os
import sys 

# sys.path.append("../")

# from onr_vis_utils import save_polygon_geojson, CustomGraphPathPlanningPipeline


app = Flask(__name__)

# Path to Cesium Build directory
CESIUM_DIR = os.path.abspath("./Cesium-1.134/Build/Cesium")


# Path to your custom JS files
MY_JS_DIR = "./static"


@app.route("/Cesium/<path:filename>")
def cesium_static(filename):
    """Serve Cesium library files"""
    return send_from_directory(CESIUM_DIR, filename)


@app.route("/js/<path:filename>")
def serve_js(filename):
    """Serve your custom JS files"""
    return send_from_directory(MY_JS_DIR, filename)



@app.route("/")
def index():

    if not os.path.exists("static/voronoi_3d.geojson"):
        # save_polygon_geojson()
        pass
    # generate_overlay("static/voronoi_3d.geojson")
    # generate_georef_overlays()

    html= """
<!DOCTYPE html>
<html>
<head>
    <script src="/Cesium/Cesium.js"></script>
    <link href="/Cesium/Widgets/widgets.css" rel="stylesheet">
    <style>
        html, body, #cesiumContainer {
            width: 100%; height: 100%;
            margin: 0; padding: 0; overflow: hidden;
        }


    /* 🟢 Toggle button styling */
    #maskToggle {
      position: absolute;
      top: 12px;
      left: 12px;
      z-index: 1000; /* make sure it's above the Cesium canvas */
      background-color: rgba(0, 0, 0, 0.75);
      color: white;
      font-family: sans-serif;
      padding: 8px 14px;
      border-radius: 6px;
      border: 1px solid #888;
      cursor: pointer;
      user-select: none;
      transition: background 0.3s;
    }

    #maskToggle:hover {
      background-color: rgba(30, 30, 30, 0.9);
    }



    </style>
</head>
<body>

  <!-- 🟢 The toggle button  -->
  <div id="maskTogglecontainer"></div>
 
    <div id="cesiumContainer"></div>
    <script>


  // ✅ Required: set Ion token
  Cesium.Ion.defaultAccessToken = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI5ZDFkYTk0Ny0yZTU0LTRlZGMtYTQ0Ni1kOGIxZjA5Y2VlYmIiLCJpZCI6MzQ2OTE2LCJpYXQiOjE3NTk1MDkzODl9.BQVyOQvlP-jAGxrYzHnqdziLHjg5pio2V7ke_HiA6D0";

    let viewer, voronoiLayer, updatedPositions = [];

  // -------------------------------
  // 1️⃣ Initialize viewer
  // -------------------------------
  async function initViewer() {
    const terrainProvider = await Cesium.createWorldTerrainAsync();

    const viewer = new Cesium.Viewer("cesiumContainer", {
      terrainProvider: terrainProvider,
      scene3DOnly: true,
      animation: false,
      timeline: false
    });

    // Add OSM 3D buildings
    Cesium.createOsmBuildingsAsync().then(buildings => {
      viewer.scene.primitives.add(buildings);
    });

    return viewer;
  }

  // -------------------------------
  // 2️⃣ Bounding Box and Base Scene
  // -------------------------------
  function addBoundingBox(viewer) {
    const west = -116.71692;
    const south = 34.202242;
    const east = -115.71606;
    const north = 34.753553;
    const height = 3000;

    const corners = [
      [west, south, height],
      [east, south, height],
      [east, north, height],
      [west, north, height],
      [west, south, height]
    ];

    viewer.entities.add({
      name: "Bounding Box Outline",
      polyline: {
        positions: Cesium.Cartesian3.fromDegreesArrayHeights(corners.flat()),
        width: 4,
        material: Cesium.Color.RED,
        clampToGround: false
      }
    });
  }

  // -------------------------------
  // 3️⃣ Fly to Area of Interest
  // -------------------------------
  async function flyToAOI(viewer) {
    const rectangle = Cesium.Rectangle.fromDegrees(-116.71692, 34.202242, -115.71606, 34.753553);
    return new Promise(resolve => {
      viewer.camera.flyTo({
        destination: rectangle,
        duration: 7,
        complete: resolve
      });
    });
  }

  // -------------------------------
  // 4️⃣ Sample Terrain for Tasks
  // -------------------------------
  async function addTasks(viewer) {
    const tasks = [
      [-116.6, 34.3],
      [-116.2, 34.5],
      [-115.9, 34.6]
    ];

    const cartographicTasks = tasks.map(t => Cesium.Cartographic.fromDegrees(t[0], t[1]));
    const updated = await Cesium.sampleTerrainMostDetailed(viewer.terrainProvider, cartographicTasks);

    updated.forEach((pos, i) => {
      viewer.entities.add({
        position: Cesium.Cartesian3.fromRadians(pos.longitude, pos.latitude, pos.height),
        point: {
          pixelSize: 12,
          color: Cesium.Color.RED,
          outlineColor: Cesium.Color.WHITE,
          outlineWidth: 2
        },
        label: {
          text: "Task " + (i + 1),
          font: "14px sans-serif",
          pixelOffset: new Cesium.Cartesian2(0, -20),
          fillColor: Cesium.Color.YELLOW,
          showBackground: true,
          backgroundColor: Cesium.Color.BLACK.withAlpha(0.6)
        }
      });
    });

    updatedPositions = updated;
  }

  // -------------------------------
  // 5️⃣ Fly Between Tasks
  // -------------------------------
  async function flyTasksSequentially(viewer) {
    for (let i = 0; i < updatedPositions.length; i++) {
      const pos = updatedPositions[i];
      const cartesian = Cesium.Cartesian3.fromRadians(pos.longitude, pos.latitude, pos.height);

      const sphere = new Cesium.BoundingSphere(cartesian, 3000.0);

      await new Promise(resolve => {
        viewer.camera.flyToBoundingSphere(sphere, {
          duration: 3,
          offset: new Cesium.HeadingPitchRange(
            Cesium.Math.toRadians(0),
            Cesium.Math.toRadians(-30),
            1000.0
          ),
          complete: resolve
        });
      });

      // Draw connecting lines
      if (i > 0) {
        const prev = updatedPositions[i - 1];
        viewer.entities.add({
          polyline: {
            positions: Cesium.Cartesian3.fromRadiansArrayHeights([
              prev.longitude, prev.latitude, prev.height,
              pos.longitude, pos.latitude, pos.height
            ]),
            width: 3,
            material: Cesium.Color.ORANGE
          }
        });
      }
    }
  }

  // -------------------------------
  // 6️⃣ Preload Voronoi GeoJSON (hidden)
  // -------------------------------
  async function preloadVoronoi(viewer) {
    const ds = await Cesium.GeoJsonDataSource.load("/static/voronoi_3d.geojson", {
      clampToGround: true
    });
    viewer.dataSources.add(ds);
    voronoiLayer = ds;

    const entities = ds.entities.values;
    for (let e of entities) {
      if (!e.polygon) continue;
      const strokeValue = e.properties.stroke?.getValue?.() || e.properties.stroke;
      const color = Cesium.Color.fromCssColorString(strokeValue || "#FFFF00").withAlpha(0.9);
      e.polygon.heightReference = Cesium.HeightReference.CLAMP_TO_GROUND;
      e.polygon.extrudedHeightReference = Cesium.HeightReference.RELATIVE_TO_GROUND;
      e.polygon.extrudedHeight = 200.0;
      e.polygon.material = color;
      e.polygon.outline = false;
      e.polygon.closeTop = false;
      e.polygon.closeBottom = false;
      e.show = false;
    }

    console.log("✅ Voronoi preloaded:", entities.length, "polygons");

  }

  


  // --- Create container for all toggles ---
  const toggleContainer = document.createElement("div");
  toggleContainer.style.display = "flex";
  toggleContainer.style.flexDirection = "column";
  toggleContainer.style.gap = "8px";
  toggleContainer.style.position = "absolute";
  toggleContainer.style.top = "10px";
  toggleContainer.style.left = "10px";
  toggleContainer.style.zIndex = "999";
  toggleContainer.style.fontFamily = "sans-serif";
  toggleContainer.style.userSelect = "none";


  // Helper to make toggles
  function makeToggle(label, colorOn, colorOff, onClick) {
    const btn = document.createElement("div");
    btn.textContent = `🟢 ${label}`;
    btn.style.background = "#202020";
    btn.style.color = "white";
    btn.style.padding = "6px 12px";
    btn.style.borderRadius = "6px";
    btn.style.cursor = "pointer";
    btn.style.border = "1px solid #555";
    btn.dataset.active = "true";

    btn.addEventListener("click", () => {
      const isActive = btn.dataset.active === "true";
      btn.dataset.active = (!isActive).toString();
      btn.textContent = `${isActive ? "⚫" : "🟢"} ${label}`;
      onClick(!isActive);
    });

    toggleContainer.appendChild(btn);
    return btn;
  }
  
function addDynamicToggles(viewer) {
  // Find the existing mask toggle div
  const maskToggle = document.getElementById("maskTogglecontainer");

  // Ensure it exists
  if (!maskToggle) {
    console.warn("⚠️ maskToggle div not found — cannot attach buttons.");
    return;
  }


  // Move existing mask toggle into this new container
  //toggleContainer.appendChild(maskToggle);

  // Attach to body (if not already)
  if (!document.body.contains(toggleContainer)) {
    document.body.appendChild(toggleContainer);
  }


  // === Landcover Toggle ===
  makeToggle("Landcover Layer", "🟢", "⚫", (show) => {
    if (viewer._landcoverLayer) viewer._landcoverLayer.show = show;
  });

  // === Elevation Toggle ===
  makeToggle("Elevation Layer", "🟢", "⚫", (show) => {
    if (viewer._elevationLayer) viewer._elevationLayer.show = show;
  });
}



function add_map_layers(viewer) {
  // --- Define AOI rectangle ---
  const rectangle = Cesium.Rectangle.fromDegrees(
    -116.71692, 34.202242, -115.71606, 34.753553
  );

  // === LAYER 1: Landcover ===
  const landcoverLayer = viewer.entities.add({
    name: "Landcover Layer",
    rectangle: {
      coordinates: rectangle,
      height: 10000, // meters above terrain
      material: new Cesium.ImageMaterialProperty({
        image: "/static/landcover_overlay.png",
        transparent: true,
        color: Cesium.Color.WHITE.withAlpha(0.8),
      }),
      outline: true,
      outlineColor: Cesium.Color.BLACK.withAlpha(0.9),
    },
    show: true,
  });

  // === LAYER 2: Elevation ===
  const elevationLayer = viewer.entities.add({
    name: "Elevation Layer",
    rectangle: {
      coordinates: rectangle,
      height: 5000, // meters above terrain
      material: new Cesium.ImageMaterialProperty({
        image: "/static/elevation_overlay.png",
        transparent: true,
        color: Cesium.Color.WHITE.withAlpha(0.8),
      }),
      outline: true,
      outlineColor: Cesium.Color.BLACK.withAlpha(0.9),
    },
    show: true,
  });

  // --- Store references globally or on viewer for later access ---
  viewer._landcoverLayer = landcoverLayer;
  viewer._elevationLayer = elevationLayer;

  // --- Create Toggle UI ---
  addDynamicToggles(viewer);
}




  // -------------------------------
  // 7️⃣ Show Voronoi Layer
  // -------------------------------
 
  
async function toggleVoronoiLayer(show = null) {
  if (!voronoiLayer) {
    console.warn("Voronoi layer not loaded yet");
    return;
  }

  // Determine desired visibility:
  // If `show` is explicitly true/false, use it; otherwise toggle current state
  const currentlyVisible = voronoiLayer.entities.values.some(e => e.show);
  const newVisible = (show === null) ? !currentlyVisible : show;

  // Apply visibility to all entities
  voronoiLayer.entities.values.forEach(e => e.show = newVisible);

  // Smoothly transition if layer is being shown
  if (newVisible) {
    try {
      await viewer.flyTo(voronoiLayer, { duration: 2 });
      console.log("✅ Voronoi layer shown and camera adjusted.");
    } catch (err) {
      console.warn("⚠️ Fly-to interrupted:", err);
    }
  } else {
    console.log("🟡 Voronoi layer hidden.");
  }
}

  // === Voronoi Toggle ===
  makeToggle("Voronoi Layer", "🟢", "⚫", (show) => {
    toggleVoronoiLayer();
  });

      // === Mask layer setup ===




// --- Utility to crop terrain + imagery ---
function applyAOIMask(rect) {


// --- Initial setup ---
viewer.scene.globe.showGroundAtmosphere = false; // no blue fog
viewer.scene.skyBox.show = false;                // hide stars
viewer.scene.skyAtmosphere.show = false;         // hide atmosphere glow
viewer.scene.backgroundColor = Cesium.Color.BLACK; // pure black background


  viewer.scene.globe.cartographicLimitRectangle = rect;

}

// --- Utility to remove AOI mask (show full Earth) ---
function clearAOIMask() {

// --- Initial setup ---
viewer.scene.globe.showGroundAtmosphere = true; // no blue fog
viewer.scene.skyBox.show = true;                // hide stars
viewer.scene.skyAtmosphere.show = true;         // hide atmosphere glow
viewer.scene.backgroundColor = Cesium.Color.BLACK; // pure black background

  viewer.scene.globe.cartographicLimitRectangle = undefined;

}


// --- Track mask state ---
//let aoiEnabled = false;
      // Toggle logic
      //const toggleButton = document.getElementById("maskToggle");

let aoiEnabled = false;
makeToggle("AOI Mask", "🟢", "🔴", (aoiEnabled) =>   {   
  if (aoiEnabled) {
    const rectangle = Cesium.Rectangle.fromDegrees(
      -116.71692, 34.202242, -115.71606, 34.753553
    );
    applyAOIMask(rectangle);
  } else {
    clearAOIMask();
  }
  aoiEnabled = !aoiEnabled;
});





  // -------------------------------
  // 8️⃣ Master Animation Sequence
  // -------------------------------
  async function runAnimationSequence(viewer) {
    console.log("🎬 Starting animation sequence...");

    addBoundingBox(viewer);
    await flyToAOI(viewer);
    await addTasks(viewer);
    await flyTasksSequentially(viewer);
    await toggleVoronoiLayer();

    console.log("🏁 Animation complete!");
  }

  // -------------------------------
  // 🚀 Main Entry Point
  // -------------------------------
  initViewer().then(async v => {
    viewer = v;
    await preloadVoronoi(viewer);  // precompute before run

    await runAnimationSequence(viewer);
    await add_map_layers(viewer);
    await toggleVoronoiLayer(false);  // start hidden


  });


    </script>
</body>
</html>

"""



    html= """
<!DOCTYPE html>
<html>
<head>
    <script src="/Cesium/Cesium.js"></script>
    <link href="/Cesium/Widgets/widgets.css" rel="stylesheet">
    <style>
        html, body, #cesiumContainer {
            width: 100%; height: 100%;
            margin: 0; padding: 0; overflow: hidden;
        }


    /* 🟢 Toggle button styling */


    #maskToggle {
    position: absolute;
    top: 8px;
    left: 8px;
    z-index: 1000; /* keep above Cesium canvas */
    background-color: rgba(0, 0, 0, 0.6);
    color: white;
    font-family: sans-serif;
    font-size: 8px;
    padding: 4px 8px;
    border-radius: 4px;
    border: 1px solid #666;
    cursor: pointer;
    user-select: none;
    transition: background 0.3s;
    }

    #maskToggle:hover {
    background-color: rgba(255, 255, 255, 0.15);
    }
    #coordsBox {
      position: absolute;
      top: 10px;
      left: 10px;
      background: rgba(0,0,0,0.7);
      color: white;
      font-size: 12px;
      padding: 6px 10px;
      border-radius: 4px;
      z-index: 1000;
      font-family: monospace;
    }

    </style>
</head>
<body>
<!--   -->
<div id="cesiumContainer"></div>
  <div id="coordsBox">Click + drag to draw rectangle</div>

    <script src="/static/sandcastle_demo.js"></script>
</body>
</html>

"""


    return render_template_string(html)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=True)



