// import * as Cesium from "cesium";
// import Sandcastle from "Sandcastle";



async function initViewer() {
// How to use the 3D Tiles Styling language to style individual features, like buildings.
// Styling language specification: https://github.com/CesiumGS/3d-tiles/tree/main/specification/Styling
const viewer = new Cesium.Viewer("cesiumContainer", {
  terrain: Cesium.Terrain.fromWorldTerrain(),
  shouldAnimate: true,
  timeline: true,
  animation: true,
});
const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);


// Add Cesium OSM buildings to the scene as our example 3D Tileset.
const osmBuildingsTileset = await Cesium.createOsmBuildingsAsync();
viewer.scene.primitives.add(osmBuildingsTileset);

    return viewer;
}



initViewer().then(async v => {
    viewer = v;
const west = -116.71692;
const south = 34.202242;
const east = -115.71606;
const north = 34.753553;

// Elevation offset in meters
const height = 3000;

// Corners with height
const corners = [
  [west, south, height],
  [east, south, height],
  [east, north, height],
  [west, north, height],
  [west, south, height]  // close loop
];

// Polyline floating above terrain
const bboxEntity = viewer.entities.add({
  name: "Bounding Box Outline",
  polyline: {
    positions: Cesium.Cartesian3.fromDegreesArrayHeights(corners.flat()),
    width: 4,
    material: Cesium.Color.RED,
    clampToGround: false,
  },
});






var rectangle = Cesium.Rectangle.fromDegrees(
    west,  // west (lon_min)
    south,   // south (lat_min)
    east,  // east (lon_max)
    north    // north (lat_max)
);

// Get center of rectangle
var center = Cesium.Rectangle.center(rectangle);
var target = Cesium.Cartesian3.fromRadians(center.longitude, center.latitude, 0);

// Make a bounding sphere around the center
var sphere = new Cesium.BoundingSphere(target, 60000.0);  // radius ~ controls how far away

// Fly with custom heading/pitch/range
viewer.camera.flyToBoundingSphere(sphere, {
  duration: 4,
  offset: new Cesium.HeadingPitchRange(
    Cesium.Math.toRadians(45),   // heading → rotate east/north
    Cesium.Math.toRadians(-30),  // pitch → tilt downward
    100000                       // range → distance away from target
  )
});




// Define your tasks (lon, lat)
var tasks = [
  [-116.6, 34.3],
 // [-116.2, 34.5],
 // [-115.9, 34.6]
];

async function show_tasks(tasks)
{

// Convert tasks into Cesium Cartographic positions (lon/lat → radians)
var cartographicTasks = tasks.map(t =>
  Cesium.Cartographic.fromDegrees(t[0], t[1])
);

// Query terrain heights for all tasks
Cesium.sampleTerrainMostDetailed(viewer.terrainProvider, cartographicTasks).then(function(updatedPositions) {
  // Add task markers with elevation
  updatedPositions.forEach((pos, i) => {
    viewer.entities.add({
      position: Cesium.Cartesian3.fromRadians(pos.longitude, pos.latitude, pos.height+10),
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


  // Fly to bounding box first
  var rectangle = Cesium.Rectangle.fromDegrees(
    -116.71692, 34.202242, -115.71606, 34.753553
  );

  viewer.camera.flyTo({
    destination: rectangle,
    duration: 3,
    complete: function() {
      // After AOI, start zooming tasks in sequence
      flyTasksSequentially(viewer, updatedPositions, 0);
    }
  });
});

}

// --- Function to fly between tasks and draw lines ---
function flyTasksSequentially(viewer, taskPositions, index) {
  if (index >= taskPositions.length) return;

  var pos = taskPositions[index];
  var cartesian = Cesium.Cartesian3.fromRadians(
    pos.longitude, pos.latitude, pos.height
  );

  // Define a bounding sphere around the task point
  var sphere = new Cesium.BoundingSphere(cartesian, 5000.0); // radius = how far out the camera frames it

  viewer.camera.flyToBoundingSphere(sphere, {
    duration: 3,
    offset: new Cesium.HeadingPitchRange(
      Cesium.Math.toRadians(0),   // heading relative to north
      Cesium.Math.toRadians(-30), // tilt down
      500.0                     // distance from the target
    ),
    complete: function() {
      if (index > 0) {
        var prev = taskPositions[index - 1];
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
      flyTasksSequentially(viewer, taskPositions, index + 1);
    }
  });
}



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



/**
 * Preloads Voronoi polygons and centroids.
 * Stores both layers hidden and ready for toggling.
 */
let voronoiLayer = null;
let centroidLayer = null;
let voronoiReady = false;

/**
 * Preload Voronoi polygons + precompute elevation-based colors/materials once.
 */
async function preloadVoronoi_seeds(viewer) {
  console.log("🔄 Loading Voronoi data...");

  const ds = await Cesium.GeoJsonDataSource.load(
    "https://raw.githubusercontent.com/pranayspeed/Cesium_demo/main/static/voronoi_3d.geojson",
    { clampToGround: true }
  );

  viewer.dataSources.add(ds);
  voronoiLayer = ds;
  const entities = ds.entities.values;

  // --- Precompute min/max elevation range ---
  let minElev = Infinity, maxElev = -Infinity;
  for (let e of entities) {
    const val = e.properties?.elevation_mean?.getValue?.() ||
                e.properties?.elevation_mean || 0;
    if (val < minElev) minElev = val;
    if (val > maxElev) maxElev = val;
  }
  console.log(`Elevation range: ${minElev.toFixed(1)} – ${maxElev.toFixed(1)} m`);

  // --- Build centroid (seed) layer ---
  centroidLayer = new Cesium.CustomDataSource("centroids");
  const allPositions = [];

  // --- Precompute elevation color/material once ---
  for (let e of entities) {
    if (!e.polygon) continue;

    const elevVal = e.properties?.elevation_mean?.getValue?.() ||
                    e.properties?.elevation_mean || 0;
    const t = Math.min(1.0, Math.max(0.0, (elevVal - minElev) / (maxElev - minElev)));
    const color = Cesium.Color.fromHsl(0.6 - 0.6 * t, 0.9, 0.5, 0.9);

    // // Attach precomputed material + extrude height to polygon once
    // e.polygon.material = color;
    // e.polygon.outline = true;
    // e.polygon.outlineColor = Cesium.Color.BLACK;
    // e.polygon.outlineWidth = 2;
    // e.polygon.height=0.0;
    // e.polygon.heightReference = Cesium.HeightReference.CLAMP_TO_GROUND;
    // e.polygon.extrudedHeightReference = Cesium.HeightReference.NONE;
    // e.polygon.extrudedHeight = elevVal * 4.0;
    // e.polygon.closeTop = true;
    // e.polygon.closeBottom = true;
    e.polygon.show = false; // start hidden

    e.asynchronous = false; // ensure immediate load

    // --- Compute centroid ---

    // Compute top edge elevation (match top of extruded polygon)
    const elev =
    e.properties?.elevation_mean?.getValue?.() ||
    e.properties?.elevation_mean ||
    0;

    const hierarchy = e.polygon.hierarchy?.getValue?.();
    if (!hierarchy?.positions?.length) continue;
    const positions = hierarchy.positions;

    // const hierarchy = e.polygon.hierarchy.getValue();
    if (hierarchy?.positions?.length) {
      const bs = Cesium.BoundingSphere.fromPoints(positions); //hierarchy.positions);
      const centroid = bs.center;
      const carto = Cesium.Cartographic.fromCartesian(centroid);
      const elevated = Cesium.Cartesian3.fromRadians(
        carto.longitude,
        carto.latitude,
        elev * 4.0 + 2.0 // just above top surface
      );
      allPositions.push(elevated);

      centroidLayer.entities.add({
        position: elevated,
        point: {
          pixelSize: 4,
          color: Cesium.Color.RED,
          outlineColor: Cesium.Color.WHITE,
          outlineWidth: 1,
        },
        show: true,
      });
    }
  }

  viewer.dataSources.add(centroidLayer);

  // --- Compute overall bounding sphere for seeds ---
  const globalSphere = Cesium.BoundingSphere.fromPoints(allPositions);
  viewer.camera.flyToBoundingSphere(globalSphere, {
    duration: 3,
    offset: new Cesium.HeadingPitchRange(
      Cesium.Math.toRadians(0),
      Cesium.Math.toRadians(-90),
      globalSphere.radius * 2.0
    ),
  });

  voronoiReady = true;
  console.log("✅ Voronoi polygons + elevation colors preloaded");
}

/**
 * Fast toggle for seed layer.
 */
function toggleSeeds(show = null) {
  if (!centroidLayer) return console.warn("⚠️ Seeds not loaded yet");
  const newState = show ?? !centroidLayer.show;
  centroidLayer.show = newState;
  console.log(`🔴 Seeds ${newState ? "shown" : "hidden"}`);
}

/**
 * Fast toggle for elevation-colored Voronoi polygons.
 */
function toggleElevationVoronoi(show = null) {

  const instances = viewer._voronoiPrimitive;
  if (instances) instances.show = show;
  viewer.scene.requestRender();
  console.log(`🟢 Voronoi outlines ${instances?.show ? "shown" : "hidden"}`);


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
  async function makeToggle(label, colorOn, colorOff, onClick) {
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
  // // Find the existing mask toggle div
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
  // const rectangle = Cesium.Rectangle.fromDegrees(
  //   -116.71692, 34.202242, -115.71606, 34.753553
  // );

  // === LAYER 1: Landcover ===
  const landcoverLayer = viewer.entities.add({
    name: "Landcover Layer",
    rectangle: {
      coordinates: rectangle,
      height: 10000, // meters above terrain
      material: new Cesium.ImageMaterialProperty({
        image: "https://raw.githubusercontent.com/pranayspeed/Cesium_demo/main/static/landcover_overlay.png",
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
        image: "https://raw.githubusercontent.com/pranayspeed/Cesium_demo/main/static/elevation_overlay.png",
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



// 🟢 Define all your polygon styles in a dictionary
const POLYGON_STYLES = {
  landcover: {
    getColor: (entity) => {
      const lc = entity.properties?.landcover?.getValue?.() || entity.properties?.landcover || 0;
    // 🌎 NALCMS (North American Land Cover Classification System) — color map
    const NALCMS_COLORS = {
    0: "#000000", // No data / Background
    1: "#476ba0", // Water
    2: "#d1def8", // Snow and Ice
    3: "#decaca", // Barren Land
    4: "#98ea8f", // Grassland
    5: "#54a84e", // Shrubland
    6: "#1f8b4c", // Deciduous Broadleaf Forest
    7: "#006400", // Evergreen Needleleaf Forest
    8: "#8dc33f", // Mixed Forest
    9: "#b38f00", // Cropland
    10: "#e3e2c3", // Wetland
    11: "#ff0000", // Urban / Built-up
    12: "#ffff00", // Lichens and Mosses
    13: "#808080", // Rock, Bare Soil
    14: "#b2b2b2", // Sparse Vegetation
    15: "#ffffff", // Clouds
    16: "#a9a9a9", // Tundra (general)
    17: "#87ceeb", // Coastal / Shallow Water
    18: "#8b4513", // Bare Ground
    19: "#556b2f", // Wet Shrubland
    20: "#daa520", // Mixed Rangeland / Herbaceous
    };

      return Cesium.Color.fromCssColorString(NALCMS_COLORS[lc] || "#cccccc").withAlpha(1.0);
    },
    outlineColor: Cesium.Color.BLACK,
    outlineWidth: 1.5,
    extrudedHeightFactor: 0.0, // flat
  },

  elevation: {
    getColor: (entity) => {
      const elev = entity.properties?.elevation_mean?.getValue?.() || entity.properties?.elevation_mean || 0;
      const t = Math.min(1, Math.max(0, (elev - 200) / (2200 - 200)));
      return Cesium.Color.fromHsl(0.6 - 0.6 * t, 0.9, 0.5, 0.95);
    },
    outlineColor: Cesium.Color.DARKGRAY,
    outlineWidth: 2.0,
    extrudedHeightFactor: 4.0, // exaggerate elevation
  },

  grade: {
    getColor: (entity) => {
      const g = entity.properties?.grade_mean?.getValue?.() || entity.properties?.grade_mean || 0;
      const t = Math.min(1, Math.max(0, g / 40.0)); // 0–40% slope
      return Cesium.Color.fromHsl(0.3 - 0.3 * t, 0.9, 0.45, 0.9);
    },
    outlineColor: Cesium.Color.WHITE,
    outlineWidth: 2.5,
    extrudedHeightFactor: 10.0,
  },
};




async function create3DPrimitiveFromVoronoi(viewer, voronoiLayer) {
  if (!voronoiLayer) {
    console.error("❌ Voronoi layer not loaded");
    return;
  }

  const entities = voronoiLayer.entities.values;
  const instances = [];
  const cachedColors = {};

  console.log(`⏳ Building 3D primitive for ${entities.length} polygons...`);

  for (const e of entities) {
    if (!e.polygon) continue;
    const hierarchyValue = e.polygon.hierarchy?.getValue?.();
    if (!hierarchyValue?.positions?.length) continue;
    const positions = hierarchyValue.positions;

    // Fetch attributes
    const elev = e.properties?.elevation_mean?.getValue?.() || e.properties?.elevation_mean || 0;

    // Default color (landcover)
    const color = POLYGON_STYLES.landcover.getColor(e);

    // Cache all styles for toggling later
    cachedColors[e.id] = {};
    for (const [styleName, def] of Object.entries(POLYGON_STYLES)) {
      cachedColors[e.id][styleName] = def.getColor(e);
    }

    try {
      // ✅ Create closed-top-and-bottom polygon geometry
      const geom = new Cesium.PolygonGeometry({
        polygonHierarchy: new Cesium.PolygonHierarchy(positions),
        height: 0,                   // base height
        extrudedHeight: elev * 4.0,  // 3D height
        closeTop: true,              // ✅ top face
        closeBottom: true,           // ✅ bottom face
        vertexFormat: Cesium.PerInstanceColorAppearance.VERTEX_FORMAT,
        // outline: true,
        // outlineColor: Cesium.Color.BLACK,
        // outlineWidth: 2.0,
      });

      instances.push(
        new Cesium.GeometryInstance({
          id: e.id || Cesium.createGuid(),
          geometry: geom,
          attributes: {
            color: Cesium.ColorGeometryInstanceAttribute.fromColor(color),
          },
        })
      );



    } catch (err) {
      console.warn("⚠️ Skipped invalid polygon:", err);
    }
  }

  if (instances.length === 0) {
    console.warn("⚠️ No polygons found for primitive.");
    return;
  }

  // ✅ Build single batched primitive with per-instance color
  const primitive = new Cesium.Primitive({
    geometryInstances: instances,
    appearance: new Cesium.PerInstanceColorAppearance({
      translucent: false,
      closed: true, // ensures backfaces rendered (3D solid look)
    }),
    asynchronous: false,
  });


  viewer._voronoiInstances = instances;

  viewer._voronoiPrimitive = primitive;

  viewer._cachedColors = cachedColors;

  viewer.scene.primitives.add(primitive);



  console.log(`✅ 3D Primitive created with ${instances.length} closed polygons`);
}

function toggle3DPrimitiveStyle(viewer, styleName) {
  const primitive = viewer._voronoiPrimitive;
  const instances = viewer._voronoiInstances;
  const cached = viewer._cachedColors;

  if (!primitive || !cached) {
    console.warn("⚠️ Primitive or cache missing");
    return;
  }


  const updatedInstances = instances.map((instance) => {
    const id = instance.id;
    const color = cached[id]?.[styleName] || Cesium.Color.GRAY;
    return new Cesium.GeometryInstance({
      id,
      geometry: instance.geometry,
      attributes: {
        color: Cesium.ColorGeometryInstanceAttribute.fromColor(color),
      },
    });
  });

  viewer.scene.primitives.remove(primitive);
  viewer._voronoiPrimitive = viewer.scene.primitives.add(
    new Cesium.Primitive({
      geometryInstances: updatedInstances,
      appearance: new Cesium.PerInstanceColorAppearance({
        translucent: false,
        closed: true,
      }),
      asynchronous: false,
    })
  );

  console.log(`🎨 Style switched to: ${styleName}`);
}

async function addBlackOutlinesForVoronoi(viewer, voronoiLayer) {
  if (!voronoiLayer) {
    console.error("❌ Voronoi layer not loaded");
    return;
  }

  const entities = voronoiLayer.entities.values;
  const outlineInstances = [];

  console.log(`⏳ Building black outlines for ${entities.length} polygons...`);

  for (const e of entities) {
    if (!e.polygon) continue;
    const hierarchy = e.polygon.hierarchy?.getValue?.();
    if (!hierarchy?.positions?.length) continue;
    const positions = hierarchy.positions;

    try {
      // Compute top edge elevation (match top of extruded polygon)
      const elev =
        e.properties?.elevation_mean?.getValue?.() ||
        e.properties?.elevation_mean ||
        0;
      const topPositions = positions.map(p => {
        const carto = Cesium.Cartographic.fromCartesian(p);
        return Cesium.Cartesian3.fromRadians(
          carto.longitude,
          carto.latitude,
          elev * 4.0 + 2.0 // just above top surface
        );
      });

      // ✅ Add closed polyline for outline
      const outline = new Cesium.PolylineGeometry({
        positions: topPositions.concat([topPositions[0]]), // close loop
        width: 1.0,
        vertexFormat: Cesium.PolylineColorAppearance.VERTEX_FORMAT,
      });

      outlineInstances.push(
        new Cesium.GeometryInstance({
          id: e.id || Cesium.createGuid(),
          geometry: outline,
          attributes: {
            color: Cesium.ColorGeometryInstanceAttribute.fromColor(Cesium.Color.BLACK),
          },
        })
      );
    } catch (err) {
      console.warn("⚠️ Skipped outline polygon:", err);
    }
  }

  if (outlineInstances.length === 0) {
    console.warn("⚠️ No outlines created.");
    return;
  }

  // ✅ Combine all outlines into one batched primitive
  const outlinePrimitive = new Cesium.Primitive({
    geometryInstances: outlineInstances,
    appearance: new Cesium.PolylineColorAppearance(),
    asynchronous: false,
    releaseGeometryInstances: false,
  });

  viewer._voronoiOutlinePrimitive = outlinePrimitive;
  viewer.scene.primitives.add(outlinePrimitive);


  console.log(`✅ Black outline primitive added (${outlineInstances.length} edges)`);
}


function toggleBlackOutlines(viewer, show = true) {
  const outline = viewer._voronoiOutlinePrimitive;
  if (outline) outline.show = show;
  viewer.scene.requestRender();
}


makeToggle("Voronoi Outlines", "🟢", "⚫", (show) => {
  toggleBlackOutlines(viewer, show);
});


await add_map_layers(viewer);


// === Voronoi Toggle ===
makeToggle("Voronoi Layer", "🟢", "⚫", (show) => {
    toggleElevationVoronoi(show);
    
  });


// === seed Toggle ===
makeToggle("Voronoi Seed", "🟢", "⚫", (show) => {
    toggleSeeds(show);
  });


// --- Track mask state ---

let aoiEnabled = false;
makeToggle("AOI Mask", "🟢", "🔴", (aoiEnabled) =>   {   
  if (aoiEnabled) {
    // const rectangle = Cesium.Rectangle.fromDegrees(
    //   -116.71692, 34.202242, -115.71606, 34.753553
    // );
    applyAOIMask(rectangle);
  } else {
    clearAOIMask();
  }
  aoiEnabled = !aoiEnabled;
});

// ANIMATION sequence starts here================

// === Setup clock ===
viewer.clock.shouldAnimate = true;
viewer.clock.startTime = Cesium.JulianDate.fromDate(new Date());
viewer.clock.currentTime = viewer.clock.startTime;
viewer.clock.stopTime = Cesium.JulianDate.addSeconds(viewer.clock.startTime, 60, new Cesium.JulianDate());
viewer.clock.clockRange = Cesium.ClockRange.CLAMPED;
viewer.timeline.zoomTo(viewer.clock.startTime, viewer.clock.stopTime);
viewer.clock.multiplier = 2;
// === Define reversible schedule ===
const reversibleSteps = [
  {
    time: 0,
    forward: async () => {
      console.log("▶ Step1: Show seeds + top-down view");
    // await preloadVoronoi_seeds(viewer);
    await toggleSeeds(false);
    
    },
    reverse: async () => {
      console.log("⬅ Undo Step1: Hide seeds + clear AOI");
      await toggleSeeds(false);
      await clearAOIMask();
    }
  },
  {
    time: 5,
    forward: async () => await viewer.flyTo(voronoiLayer, { duration: 2 }),
    reverse: async () => console.log("⬅ Undo fly-to (optional reset camera)")
  },
  {
    time: 7,
    forward: async () => await applyAOIMask(rectangle),
    reverse: async () => await  clearAOIMask()
  },
  {
    time: 8,
    forward: async () => viewer.entities.remove(bboxEntity),
    reverse: async () => viewer.entities.add(bboxEntity)
  },
  {
    time: 10,
    forward: async () => await toggleSeeds(true),
    reverse: async () => await toggleSeeds(false)
  },
  {
    time: 14,
    forward: async () => {
    await toggleSeeds(false);
    await toggleElevationVoronoi(true);
    },
    reverse: async () => {
      await toggleElevationVoronoi(false);
      await toggleSeeds(true);
    },

  }
];

let executedForward = new Set();
let executedReverse = new Set();
let lastTime = viewer.clock.startTime;

console.log("⏳ Preloading Voronoi data...");
await preloadVoronoi_seeds(viewer);
await toggleSeeds(true);
await toggleElevationVoronoi(true);
console.log("✅ Voronoi data ready. Use timeline or play/pause to control.");
await toggleSeeds(false);
await toggleElevationVoronoi(false);

console.log("Cacheing polygon styles...");
await create3DPrimitiveFromVoronoi(viewer, voronoiLayer);
console.log("✅ Polygon styles cached.");


// toggle3DPrimitiveStyle(viewer, "landcover");
await addBlackOutlinesForVoronoi(viewer, voronoiLayer);

document.addEventListener("keydown", (ev) => {
  if (ev.key === "1") toggle3DPrimitiveStyle(viewer, "landcover");
  if (ev.key === "2") toggle3DPrimitiveStyle(viewer, "elevation");
  if (ev.key === "3") toggle3DPrimitiveStyle(viewer, "grade");
});


// === On every clock tick ===
viewer.clock.onTick.addEventListener(async (clock) => {
  

  const currentSeconds = Cesium.JulianDate.secondsDifference(clock.currentTime, clock.startTime);
  const lastSeconds = Cesium.JulianDate.secondsDifference(lastTime, clock.startTime);
  const movingForward = currentSeconds > lastSeconds;
  lastTime = clock.currentTime;

  for (const step of reversibleSteps) {
    if (movingForward && currentSeconds >= step.time && !executedForward.has(step.time)) {
      executedForward.add(step.time);
      executedReverse.delete(step.time);
      await step.forward();
    } else if (!movingForward && currentSeconds < step.time && !executedReverse.has(step.time)) {
      executedReverse.add(step.time);
      executedForward.delete(step.time);
      await step.reverse();
    }
  }
});



  });






