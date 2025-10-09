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
  sceneModePicker: false,
  baseLayerPicker: true,
  scene3DOnly: true,
  // ✅ Required for clamped polylines:
  contextOptions: {
    requestWebgl2: true
  },
  scene: {
    groundPrimitives: true
  }

});
const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);

viewer.scene.globe.depthTestAgainstTerrain = true;
viewer.scene.globe.enableLighting = true;


//Pranay: not added buildings for now
// // Add Cesium OSM buildings to the scene as our example 3D Tileset.
// const osmBuildingsTileset = await Cesium.createOsmBuildingsAsync();
// viewer.scene.primitives.add(osmBuildingsTileset);

    return viewer;
}



initViewer().then(async v => {
    viewer = v;

      await new Promise(resolve => setTimeout(resolve, 1000)); // 1s sleep

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

function toggle_bbox_outline(show = true) {
  if (bboxEntity) bboxEntity.show = show;
  viewer.scene.requestRender();
}







var rectangle = Cesium.Rectangle.fromDegrees(
    west,  // west (lon_min)
    south,   // south (lat_min)
    east,  // east (lon_max)
    north    // north (lat_max)
);

// // Get center of rectangle
// var center = Cesium.Rectangle.center(rectangle);
// var target = Cesium.Cartesian3.fromRadians(center.longitude, center.latitude, 0);

// // Make a bounding sphere around the center
// var sphere = new Cesium.BoundingSphere(target, 60000.0);  // radius ~ controls how far away

// // Fly with custom heading/pitch/range
// viewer.camera.flyToBoundingSphere(sphere, {
//   duration: 4,
//   offset: new Cesium.HeadingPitchRange(
//     Cesium.Math.toRadians(45),   // heading → rotate east/north
//     Cesium.Math.toRadians(-30),  // pitch → tilt downward
//     100000                       // range → distance away from target
//   )
// });


async function zoom_to_custom_height_1() {
// Get center of rectangle
var center = Cesium.Rectangle.center(rectangle);
var target = Cesium.Cartesian3.fromRadians(center.longitude, center.latitude, 0);

// Make a bounding sphere around the center
var sphere = new Cesium.BoundingSphere(target, 60000.0);  // radius ~ controls how far away

// Fly with custom heading/pitch/range
viewer.camera.flyToBoundingSphere(sphere, {
  duration: 2,
  offset: new Cesium.HeadingPitchRange(
    Cesium.Math.toRadians(45),   // heading → rotate east/north
    Cesium.Math.toRadians(-30),  // pitch → tilt downward
    150000                       // range → distance away from target
  )
});

}



// Define your tasks (lon, lat)
var tasks_old = [
  [-116.6, 34.3],
 [-116.2, 34.5],
 [-115.9, 34.6]
];

async function show_tasks(tasks, flyToAOI = true)
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


  // // Fly to bounding box first
  // var rectangle = Cesium.Rectangle.fromDegrees(
  //   -116.71692, 34.202242, -115.71606, 34.753553
  // );
  if (!flyToAOI) return;
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
async function applyAOIMask(rect) {


// --- Initial setup ---
viewer.scene.globe.showGroundAtmosphere = false; // no blue fog
viewer.scene.skyBox.show = false;                // hide stars
viewer.scene.skyAtmosphere.show = false;         // hide atmosphere glow
viewer.scene.backgroundColor = Cesium.Color.BLACK; // pure black background


  viewer.scene.globe.cartographicLimitRectangle = rect;

}

// --- Utility to remove AOI mask (show full Earth) ---
async function clearAOIMask() {

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
// let voronoiLayer = null;
let centroidLayer = null;
let voronoiReady = false;

/**
 * Preload Voronoi polygons + precompute elevation-based colors/materials once.
 */
async function preloadVoronoi_seeds(viewer, voronoiLayer) {
  // console.log("🔄 Loading Voronoi data...");
  // //https://raw.githubusercontent.com/pranayspeed/Cesium_demo/main
  // const ds = await Cesium.GeoJsonDataSource.load(
  //   "/static/voronoi_3d.geojson",
  //   { clampToGround: true }
  // );

  // viewer.dataSources.add(ds);
  // voronoiLayer = ds;
  const entities = voronoiLayer.entities.values;

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
  const centroidMap = {}; // id → entity

  // --- Precompute elevation color/material once ---
  // for (let e of entities) {
  entities.forEach((e, i) => {
    // if (!e.polygon) continue;

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
    // if (!hierarchy?.positions?.length) continue;
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


      centroidMap[i] = elevated;
      // console.log(`Polygon ${i} centroid: ${elevated}, ${carto.longitude}, ${carto.latitude}`);

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
  });

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

  return centroidMap;
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

}


// --- Create Toggle UI ---
addDynamicToggles(viewer);





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
      height: 0, //10000, // meters above terrain
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
      height: 0, //5000, // meters above terrain
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


  // === Landcover Toggle ===
  makeToggle("Landcover Layer", "🟢", "⚫", (show) => {
    if (viewer._landcoverLayer) viewer._landcoverLayer.show = show;
  });

  // === Elevation Toggle ===
  makeToggle("Elevation Layer", "🟢", "⚫", (show) => {
    if (viewer._elevationLayer) viewer._elevationLayer.show = show;
  });



}




// 🟢 Define all your polygon styles in a dictionary
const POLYGON_STYLES = {
  landcover: {
    getColor: (entity) => {
      const lc = entity.properties?.landcover?.getValue?.() || entity.properties?.landcover || 0;
    // 🌎 NALCMS (North American Land Cover Classification System) — color map
    
        const NALCMS_COLORS = {
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


// makeToggle("Voronoi Outlines", "🟢", "⚫", (show) => {
//   toggleBlackOutlines(viewer, show);
// });





async function loadVoronoiWireframe(viewer, graphUrl, centroidMap,
  color = Cesium.Color.CYAN, width = 1.0, scaleZ = 4.0) {

  try {
    const resp = await fetch(graphUrl);
    const geojson = await resp.json();
    const lineInstances = [];

    for (const feature of geojson.features) {
      if (feature.geometry.type !== "LineString") continue;

      const props = feature.properties || {};
      // const srcId = props.src_id ?? props.source ?? 0;
      // const dstId = props.dst_id ?? props.target ?? 0;

      // const elevSrc = (elevationMap[srcId] ?? 0) * scaleZ;
      // const elevDst = (elevationMap[dstId] ?? 0) * scaleZ;

      // console.log(`Edge ${srcId}→${dstId} elevations: ${elevSrc.toFixed(1)} → ${elevDst.toFixed(1)} m`);

      const coords = feature.geometry.coordinates;


      cartographicTasks = coords.map(t =>
        Cesium.Cartographic.fromDegrees(t[0], t[1])
      );

      console.log('feature coords:', coords[0], coords[1]);
      console.log('cartographicTasks:', cartographicTasks[0], cartographicTasks[1]);
      // Query terrain heights for all tasks
      Cesium.sampleTerrainMostDetailed(viewer.terrainProvider, cartographicTasks).then(function(updatedPositions) {
        // Add task markers with elevation
        const positions = [];
        updatedPositions.forEach((pos, i) => {

          positions.push(Cesium.Cartesian3.fromRadians(pos.longitude, pos.latitude, pos.height*4.0 + 2));
        });


          lineInstances.push(
            new Cesium.GeometryInstance({
              geometry: new Cesium.PolylineGeometry({
                positions,
                width,
                vertexFormat: Cesium.PolylineColorAppearance.VERTEX_FORMAT,
              }),
              attributes: {
                color: Cesium.ColorGeometryInstanceAttribute.fromColor(color),
              },
            })
          );

      });


    }


      // Convert tasks into Cesium Cartographic positions (lon/lat → radians)
      var cartographicTasks = tasks.map(t =>
        Cesium.Cartographic.fromDegrees(t[0], t[1])
      );


    if (!lineInstances.length) {
      console.warn("⚠️ No LineString features found in graph:", graphUrl);
      return null;
    }

    const primitive = new Cesium.Primitive({
      geometryInstances: lineInstances,
      appearance: new Cesium.PolylineColorAppearance({ translucent: true }),
      asynchronous: false,
    });

    viewer.scene.primitives.add(primitive);
    console.log(`✅ Wireframe loaded: ${lineInstances.length} edges using elevation lookup`);
    return primitive;

  } catch (err) {
    console.error("❌ Failed to load Voronoi wireframe:", err);
  }
}



async function addVoronoiGraph(viewer, graphUrl = "/static/voronoi_graph.geojson", centroidMap = null) {


(async () => {
  const wire = await loadVoronoiWireframe(viewer, "/static/voronoi_graph.geojson", centroidMap,
                             Cesium.Color.CYAN, 1.5, 4.0);


viewer._voronoiGraphLayer = wire;

console.log("✅ Voronoi graph overlay added from GeoJSON");

})();

}


function toggleVoronoiGraph(viewer, show = null) {
  const layer = viewer._voronoiGraphLayer;
  if (!layer) return console.warn("⚠️ Graph layer not yet added");

  if (show === null) {
    layer.show = !layer.show;
  } else {
    layer.show = !!show;
  }

  console.log(`🔁 Graph layer ${layer.show ? "visible" : "hidden"}`);
  viewer.scene.requestRender();
}


let rectangleEntity = null;
function select_bounds_draw(viewer) {
let startCartographic = null;
    let endCartographic = null;
    

    const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
    const coordsBox = document.getElementById("coordsBox");

    // 2️⃣ On first click: set start point
    handler.setInputAction(function (click) {
      const cartesian = viewer.scene.pickPosition(click.position);
      if (!cartesian) return;

      if (!startCartographic) {
        if (rectangleEntity) viewer.entities.remove(rectangleEntity);
        rectangleEntity = null;
            startCartographic = Cesium.Cartographic.fromCartesian(cartesian);
            endCartographic = null;
            if (rectangleEntity) viewer.entities.remove(rectangleEntity);
            coordsBox.textContent = `Start set — drag to size rectangle... ${startCartographic}`;
        }
        else {

            endCartographic = Cesium.Cartographic.fromCartesian(cartesian);
            const rect = Cesium.Rectangle.fromCartographicArray([
                startCartographic,
                endCartographic,
            ]);
            rectangleEntity.rectangle.coordinates = rect;
            const west = Cesium.Math.toDegrees(rect.west).toFixed(6);
            const south = Cesium.Math.toDegrees(rect.south).toFixed(6);
            const east = Cesium.Math.toDegrees(rect.east).toFixed(6);
            const north = Cesium.Math.toDegrees(rect.north).toFixed(6);

            console.log("✅ Rectangle drawn:", { west, south, east, north });
            coordsBox.textContent = `Final Rectangle → W S E N:${west}, ${south}, ${east}, ${north}`;

            startCartographic = null;
            endCartographic = null;
            handler.destroy();
            // Optionally, do something with the final rectangle here
    }


    }, Cesium.ScreenSpaceEventType.LEFT_DOWN);

    // 3️⃣ On mouse move: update rectangle dynamically
    handler.setInputAction(function (movement) {
      if (!startCartographic) return;
      const cartesian = viewer.scene.pickPosition(movement.endPosition);
      if (!cartesian) return;
      endCartographic = Cesium.Cartographic.fromCartesian(cartesian);

      const rect = Cesium.Rectangle.fromCartographicArray([
        startCartographic,
        endCartographic,
      ]);

      if (!rectangleEntity) {
        rectangleEntity = viewer.entities.add({
          name: "Drawn Rectangle",
          rectangle: {
            coordinates: rect,
            material: Cesium.Color.YELLOW.withAlpha(0.3),
            outline: true,
            outlineColor: Cesium.Color.RED,
            heightReference: 200, //Cesium.HeightReference.CLAMP_TO_GROUND,
            clampToGround: false,
            height: 0,
            extrudedHeight: 200,
            outlineWidth: 2,
          },
        });
      } else {
        rectangleEntity.rectangle.coordinates = rect;
      }


      const west = Cesium.Math.toDegrees(rect.west).toFixed(5);
      const south = Cesium.Math.toDegrees(rect.south).toFixed(5);
      const east = Cesium.Math.toDegrees(rect.east).toFixed(5);
      const north = Cesium.Math.toDegrees(rect.north).toFixed(5);

      coordsBox.textContent = `W S E N:${west}, ${south}, ${east}, ${north}`;
    }, Cesium.ScreenSpaceEventType.MOUSE_MOVE);


}


makeToggle("Draw AOI", "🟢", "⚫", (active) => {
  if (active) {
    select_bounds_draw(viewer);
  } else {
    // Stop drawing
    if (rectangleEntity) viewer.entities.remove(rectangleEntity);
  }
});




function add_overlay_texture(viewer, imageUrl, rectangle, layer_name = "Overlay", alpha=1.0) {
  const overlay = viewer.imageryLayers.addImageryProvider(
    new Cesium.SingleTileImageryProvider({
      url: imageUrl,
      rectangle: rectangle,
    })
  );

  // Optional: adjust transparency
  overlay.alpha = alpha;
  overlay.brightness = 1.0;

  // === Voronoi Toggle ===
  makeToggle(layer_name, "🟢", "⚫", (show) => {
    if (overlay) {
      overlay.show = show;
    }
  });

  return overlay;
}

// base_url = "https://raw.githubusercontent.com/pranayspeed/Cesium_demo/main/static/";
base_url = "/static/";


const texture_layer_names = ["Landcover Texture", "Elevation Texture", "Voronoi Texture", "Graph Texture", "Seed Texture"];

viewer._landcoverOverlay = add_overlay_texture(viewer, base_url + "landcover_overlay.png", rectangle, "Landcover Texture", alpha=0.7);
viewer._elevationOverlay = add_overlay_texture(viewer, base_url + "elevation_overlay.png", rectangle, "Elevation Texture", alpha=0.7);

viewer._voronoiOverlay = add_overlay_texture(viewer, base_url + "voronoi_overlay.png", rectangle, "Voronoi Texture", alpha=0.7);

// viewer._voronoiGraphOverlay = add_overlay_texture(viewer, base_url + "voronoi_graph_overlay.png", rectangle, "Graph Texture", alpha=1.0);

viewer._voronoiGraphOverlay = add_overlay_at_height(viewer, base_url + "voronoi_graph_overlay.png", rectangle, 3000, "Graph Texture", alpha=0.5);



viewer._voronoiSeedOverlay = add_overlay_texture(viewer, base_url + "voronoi_seeds_overlay.png", rectangle, "Seed Texture", alpha=1.0);

viewer._voronoiSeed_heightOverlay = add_overlay_at_height(viewer, base_url + "voronoi_seeds_overlay.png", rectangle, 3010, "Seed layer", alpha=1.0);


// await add_map_layers(viewer);


//abstract AOI mask functions




// ANIMATION sequence starts here================


let voronoiLayer = null;

  async function load_voronoi_layer_ds() {
    console.log("🔄 Loading Voronoi data...");
    //https://raw.githubusercontent.com/pranayspeed/Cesium_demo/main
    const ds = await Cesium.GeoJsonDataSource.load(
      "/static/voronoi_3d.geojson",
      { clampToGround: true,
        show: false  // start hidden
      }
    );

viewer.dataSources.add(ds);
voronoiLayer = ds;


const entities = voronoiLayer.entities.values;

entities.forEach((entity) => {

  entity.polygon.show = false; // start hidden
});

}

await load_voronoi_layer_ds();

if (!voronoiLayer) {
  console.error("❌ Voronoi layer failed to load");
  return;
}




// // 3D content

// console.log("⏳ Preloading Voronoi data...");
// const centroidMap = await preloadVoronoi_seeds(viewer, voronoiLayer);


// // === seed Toggle ===
// makeToggle("Voronoi Seed", "🟢", "⚫", (show) => {
//     toggleSeeds(show);
//   });


// await toggleSeeds(false);
// await toggleElevationVoronoi(false);


// console.log("Cacheing polygon styles...");
// await create3DPrimitiveFromVoronoi(viewer, voronoiLayer);
// console.log("✅ Polygon styles cached.");

// // toggle3DPrimitiveStyle(viewer, "landcover");
// await addBlackOutlinesForVoronoi(viewer, voronoiLayer);

// document.addEventListener("keydown", (ev) => {
//   if (ev.key === "1") toggle3DPrimitiveStyle(viewer, "landcover");
//   if (ev.key === "2") toggle3DPrimitiveStyle(viewer, "elevation");
//   if (ev.key === "3") toggle3DPrimitiveStyle(viewer, "grade");
// });


// // === Voronoi Toggle ===
// makeToggle("Voronoi Layer", "🟢", "⚫", (show) => {
//     toggleElevationVoronoi(show);
//     toggleBlackOutlines(viewer, show);
//   });



// show_tasks(tasks_old);
// --- Track mask state ---

let aoiEnabled = false;
makeToggle("AOI Mask", "🟢", "🔴", (aoiEnabled) =>   {   
  if (aoiEnabled) {
    applyAOIMask(rectangle);
  } else {
    clearAOIMask();
  }
  aoiEnabled = !aoiEnabled;
});





// Helper to add a normal action button (not a toggle)
function makeButton(label, color = "#0078D7", onClick) {
  const btn = document.createElement("div");
  btn.textContent = label;
  btn.style.background = color;
  btn.style.color = "white";
  btn.style.padding = "6px 14px";
  btn.style.borderRadius = "6px";
  btn.style.cursor = "pointer";
  btn.style.margin = "4px";
  btn.style.fontWeight = "500";
  btn.style.border = "1px solid #444";
  btn.style.boxShadow = "0 1px 2px rgba(0,0,0,0.3)";
  btn.style.transition = "background 0.2s";

  btn.addEventListener("mouseenter", () => btn.style.background = "#0a84ff");
  btn.addEventListener("mouseleave", () => btn.style.background = color);
  btn.addEventListener("click", onClick);

  toggleContainer.appendChild(btn);
  return btn;
}



// // === Define reversible schedule ===
// const reversibleSteps = [
//   {
//     time: 0,
//     forward: async () => {
//       console.log("▶ Step1: Show seeds + top-down view");
//     await toggleSeeds(false);
//     await clearAOIMask();
    
//     },
//     reverse: async () => {
//       console.log("⬅ Undo Step1: Hide seeds + clear AOI");
//       await toggleSeeds(false);
//       await clearAOIMask();
//     }
//   },
//   {
//     time: 5,
//     forward: async () => await viewer.flyTo(voronoiLayer, { duration: 2 }),
//     reverse: async () => console.log("⬅ Undo fly-to (optional reset camera)")
//   },
//   {
//     time: 7,
//     forward: async () => {
//       await applyAOIMask(rectangle);
//       zoom_to_custom_height_1();
//     },
//     reverse: async () => await  clearAOIMask()
//   },
//   {
//     time: 8,
//     forward: async () => viewer.entities.remove(bboxEntity),
//     reverse: async () => viewer.entities.add(bboxEntity)
//   },
//   {
//     time: 10,
//     forward: async () => await toggleSeeds(true),
//     reverse: async () => await toggleSeeds(false)
//   },
//   {
//     time: 14,
//     forward: async () => {
//     await toggleSeeds(false);
//     await toggleElevationVoronoi(true);
//     },
//     reverse: async () => {
//       await toggleElevationVoronoi(false);
//       await toggleSeeds(true);
//     },

//   }
// ];


function hideall_textures() {
  viewer._landcoverOverlay.show = false;
  viewer._elevationOverlay.show = false;
  viewer._voronoiOverlay.show = false;
  viewer._voronoiGraphOverlay.show = false;
  viewer._voronoiSeedOverlay.show = false;
}

// === Define reversible schedule ===
const reversibleSteps = [
  {
    time: 0,
    forward: async () => {
      console.log("▶ Step1: Show seeds + top-down view");
    await toggleSeeds(false);
    await clearAOIMask();
    
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
    forward: async () => {
      await applyAOIMask(rectangle);
      zoom_to_custom_height_1();
    },
    reverse: async () => await  clearAOIMask()
  },
  {
    time: 8,
    forward: async () => viewer.entities.remove(bboxEntity),
    reverse: async () => viewer.entities.add(bboxEntity)
  }
];



const abstraction_roi_rectangle = [-116.351152, 34.399434, -116.276720, 34.460928]; 
// var bboxEntity_new = null;


function draw_abstraction_roi() {
// Get center of rectangle
// W, S, E, N
const west = abstraction_roi_rectangle[0];
const south = abstraction_roi_rectangle[1];
const east = abstraction_roi_rectangle[2];
const north = abstraction_roi_rectangle[3];
const centerLon = (west + east) / 2;
const centerLat = (south + north) / 2;

var scale=1;
corners_curr = [
  [west, south, 2000*scale],
  [west, north, 2000*scale],
  [east, north, 2000*scale],
  [east, south, 2000*scale],
  [west, south, 2000*scale], // close loop
];


// Polyline floating above terrain
const bboxEntity_new = viewer.entities.add({
  name: "Bounding Box Outline",
  polyline: {
    positions: Cesium.Cartesian3.fromDegreesArrayHeights(corners_curr.flat()),
    width: 4,
    material: Cesium.Color.BLUE,
    clampToGround: false,
  },
});
return bboxEntity_new;
}

const bboxEntity_new = draw_abstraction_roi();

const rectangle_curr = Cesium.Rectangle.fromDegrees(abstraction_roi_rectangle[0], abstraction_roi_rectangle[1], abstraction_roi_rectangle[2], abstraction_roi_rectangle[3]);


function toggle_abstraction_bbox(show = true) {
  if (bboxEntity_new) bboxEntity_new.show = show;
  viewer.scene.requestRender();
}



makeButton("Reset View", "#0078D7", async () => {
  hideall_textures();
  clearAOIMask();
  await viewer.flyTo(voronoiLayer, { duration: 2 });
  toggle_bbox_outline(false);
  toggle_abstraction_bbox(false);

});

makeButton("Step 1", "#0078D7", async () => {
hideall_textures();
toggle_bbox_outline(true);
clearAOIMask();
await zoom_to_custom_height_1();

});


makeButton("Step 2", "#0078D7", async () => {

  
await applyAOIMask(rectangle);
// manually show Elevation
// manually show Landcover

});






async function zoom_to_custom_abstract_region() {
// Get center of rectangle
// W, S, E, N
const west = abstraction_roi_rectangle[0];
const south = abstraction_roi_rectangle[1];
const east = abstraction_roi_rectangle[2];
const north = abstraction_roi_rectangle[3];
const centerLon = (west + east) / 2;
const centerLat = (south + north) / 2;

  var cartesian = Cesium.Cartesian3.fromDegrees(centerLon, centerLat, 2000);

  // Define a bounding sphere around the task point
  var sphere = new Cesium.BoundingSphere(cartesian, 5000.0); // radius = how far out the camera frames it

  viewer.camera.flyToBoundingSphere(sphere, {
    duration: 3,
    offset: new Cesium.HeadingPitchRange(
      Cesium.Math.toRadians(0),   // heading relative to north
      Cesium.Math.toRadians(-30), // tilt down
      10000.0                     // distance from the target
    ),
  });




  // await viewer.flyTo(rectangle_curr, { duration: 2 })

}




makeButton("Step 3", "#0078D7", async () => {
toggle_abstraction_bbox(true);
await zoom_to_custom_abstract_region();
// manually show Elevation
// manually show Landcover
// await clearAOIMask();
applyAOIMask(rectangle_curr);

});


makeButton("Step 4", "#0078D7", async () => {
  toggle_abstraction_bbox(false);
hideall_textures();
clearAOIMask();
toggle_bbox_outline(true);
clearAOIMask();
await zoom_to_custom_height_1();
applyAOIMask(rectangle);
});


// show tasks

const tasks = [[-116.69014300133762, 34.242955807461634],
 [-115.7741717137646, 34.55212071346959],
 [-116.286038021526, 34.48491095129395],
 [-116.63626233736274, 34.64621438051549],
 [-115.85499270972693, 34.53867876103446],
 [-115.88193304171438, 34.35049142694266],
 [-116.58238167338786, 34.45802704642369],
 [-116.36685901748832, 34.283281664767024],
 [-116.05704519963274, 34.269839712331894],
 [-115.90887337370181, 34.67309828538574],
 [-115.7741717137646, 34.431143141553434]];


async function add_task_markers(viewer, taskLocations) {
// Create a data source to group them
// const taskLayer = new Cesium.CustomDataSource("taskPoints");

const taskLayer  = await Cesium.CzmlDataSource.load("/static/task_points_1.czml");
viewer.dataSources.add(taskLayer);
viewer.zoomTo(taskLayer);

return taskLayer;

var cartographicTasks = taskLocations.map(t =>
  Cesium.Cartographic.fromDegrees(t[0], t[1])
);
await Cesium.sampleTerrainMostDetailed(viewer.terrainProvider, cartographicTasks).then(function(updatedPositions) {

  updatedPositions.forEach((pos, i) => {
//     taskLocations[i][2] = pos.height;
//   });
// });


// Add markers
// taskLocations.forEach((coord, i) => {
//   const [lon, lat, height] = coord;
  taskLayer.entities.add({
    id: `task_${i}`,
    name: `Task ${i}`,
    position: Cesium.Cartesian3.fromRadians(pos.longitude, pos.latitude, pos.height+20),// Cesium.Cartesian3.fromDegrees(lon, lat, height),
    point: {
      pixelSize: 10,
      color: (i === 2) ? Cesium.Color.ORANGE :       // Vehicle Depot
             (i === 10) ? Cesium.Color.BLUE :        // Human Depot
             Cesium.Color.RED,
      outlineColor: Cesium.Color.WHITE,
      outlineWidth: 1.5
    },
    label: {
      text: `T${i}`,
      font: "14px sans-serif",
      style: Cesium.LabelStyle.FILL_AND_OUTLINE,
      fillColor: Cesium.Color.BLACK,
      outlineColor: Cesium.Color.BLACK,
      outlineWidth: 2,
      verticalOrigin: Cesium.VerticalOrigin.TOP,
      pixelOffset: new Cesium.Cartesian2(0, -16)
    }
  });
});

});

// Add to viewer
viewer.dataSources.add(taskLayer);

// Focus camera on all tasks
viewer.flyTo(taskLayer);

return taskLayer;
}

// const taskLayer = await add_task_markers(viewer,tasks);

// show_tasks(tasks, false);

const taskLayer = await add_task_markers(viewer,tasks);


function toggle_tasks_markers(show = true) {
  if (taskLayer) taskLayer.show = show;
  viewer.scene.requestRender();
}

// === Tasks Toggle ===
makeToggle("Task Markers", "🟢", "⚫", (show) => {
    toggle_tasks_markers(show);
  });


function add_overlay_at_height(viewer, imageUrl, rectangle, height = 1000, layer_name = "Overlay", alpha=1.0) {
  // --- Define AOI rectangle ---
  // const rectangle = Cesium.Rectangle.fromDegrees(
  //   -116.71692, 34.202242, -115.71606, 34.753553
  // );

  // === LAYER 1: Landcover ===
  const current_layer = viewer.entities.add({
    name: layer_name,
    rectangle: {
      coordinates: rectangle,
      height: height, //10000, // meters above terrain
      material: new Cesium.ImageMaterialProperty({
        image: imageUrl,
        transparent: true,
        color: Cesium.Color.WHITE.withAlpha(alpha),
      }),
      outline: true,
      outlineColor: Cesium.Color.BLACK.withAlpha(0.9),
    },
    show: true,
  });


  // === Landcover Toggle ===
  makeToggle(layer_name, "🟢", "⚫", (show) => {
    if (current_layer) current_layer.show = show;
  });
  return current_layer;


}


// viewer._agent1_all_paths = add_overlay_at_height(viewer, base_url + "HeavyHumanmission_all_paths_overlay.png", rectangle, 3000, "H Human", alpha=1.0);
// viewer._agent2_all_paths = add_overlay_at_height(viewer, base_url + "LightHumanmission_all_paths_overlay.png", rectangle, 3000, "L Human", alpha=1.0);
// viewer._agent3_all_paths = add_overlay_at_height(viewer, base_url + "HeavyVehiclemission_all_paths_overlay.png", rectangle, 3000, "H Vehicle", alpha=1.0);
// viewer._agent4_all_paths = add_overlay_at_height(viewer, base_url + "LightVehiclemission_all_paths_overlay.png", rectangle, 3000, "L Vehicle", alpha=1.0);



// ALL PATHS LAYER (flat on ground)

// viewer._agent1_all_paths = add_overlay_texture(viewer, base_url + "HeavyHumanmission_all_paths_overlay.png", rectangle, "H Human", alpha=1.0);
// viewer._agent2_all_paths = add_overlay_texture(viewer, base_url + "LightHumanmission_all_paths_overlay.png", rectangle, "L Human", alpha=1.0);
// viewer._agent3_all_paths = add_overlay_texture(viewer, base_url + "HeavyVehiclemission_all_paths_overlay.png", rectangle, "H Vehicle", alpha=1.0);
// viewer._agent4_all_paths = add_overlay_texture(viewer, base_url + "LightVehiclemission_all_paths_overlay.png", rectangle, "L Vehicle", alpha=1.0);




async function clampCZMLToTerrain(ds) {
  for (const ent of ds.entities.values) {
    if (!ent.position) continue;
    const property = ent.position;
    const times = property._property?._times || property._times || [];
    if (!times.length) continue;

    const positions = times.map(t => property.getValue(t));
    const cartographics = positions.map(p => Cesium.Cartographic.fromCartesian(p));
    const updated = await Cesium.sampleTerrainMostDetailed(viewer.terrainProvider, cartographics);

    // Replace entity’s position samples with terrain heights
    const newSamples = updated.map(c => Cesium.Cartesian3.fromRadians(c.longitude, c.latitude, c.height));
    const newProp = new Cesium.SampledPositionProperty();
    times.forEach((t, i) => newProp.addSample(t, newSamples[i]));
    ent.position = newProp;
  }
  console.log("✅ All agent paths clamped to terrain heights");
}




makeButton("COA 1", "#0078D7", async () => {

  // const czmlPromise = Cesium.CzmlDataSource.load("/static/agents_coa_1.czml");

  const czmlPromise = Cesium.CzmlDataSource.load("/static/agents_coa_3d_1.czml");

czmlPromise.then(ds => {
  viewer.dataSources.add(ds);

  clampCZMLToTerrain(ds).then(() => {

  // viewer.zoomTo(ds);
  // Speed up the simulation:
  viewer.clock.multiplier = 1;   // 10× faster than real time
  viewer.clock.shouldAnimate = true;
  ds.entities.values.forEach(ent => {
    if (ent.position) {
      const fullPath = new Cesium.PolylineGraphics({
        positions: ent.position, // uses the same dynamic path
        clampToGround: true,
        width: 1.5,
        material: Cesium.Color.WHITE.withAlpha(0.3)
      });
      viewer.entities.add({
        polyline: fullPath
      });
    }
  });
  });
});

});


makeButton("COA 2", "#0078D7", async () => {

  // const czmlPromise = Cesium.CzmlDataSource.load("/static/agents_coa_1.czml");

  const czmlPromise = Cesium.CzmlDataSource.load("/static/agents_coa_3d_2.czml");

czmlPromise.then(ds => {
  viewer.dataSources.add(ds);

  clampCZMLToTerrain(ds).then(() => {

  // viewer.zoomTo(ds);
  // Speed up the simulation:
  viewer.clock.multiplier = 1;   // 10× faster than real time
  viewer.clock.shouldAnimate = true;
  ds.entities.values.forEach(ent => {
    if (ent.position) {
      const fullPath = new Cesium.PolylineGraphics({
        positions: ent.position, // uses the same dynamic path
        clampToGround: true,
        width: 1.5,
        material: Cesium.Color.WHITE.withAlpha(0.3)
      });
      viewer.entities.add({
        polyline: fullPath
      });
    }
  });
  });
});

});



function addWeatherEffectForRectangle(viewer, rectangleEntity, type = "snow") {
  const scene = viewer.scene;
  scene.globe.depthTestAgainstTerrain = true;


// snow
const snowParticleSize = 12.0;
const snowRadius = 100000.0;
const minimumSnowImageSize = new Cesium.Cartesian2(
  snowParticleSize,
  snowParticleSize,
);
const maximumSnowImageSize = new Cesium.Cartesian2(
  snowParticleSize * 2.0,
  snowParticleSize * 2.0,
);
let snowGravityScratch = new Cesium.Cartesian3();
const snowUpdate = function (particle, dt) {
  snowGravityScratch = Cesium.Cartesian3.normalize(
    particle.position,
    snowGravityScratch,
  );
  Cesium.Cartesian3.multiplyByScalar(
    snowGravityScratch,
    Cesium.Math.randomBetween(-30.0, -300.0),
    snowGravityScratch,
  );
  particle.velocity = Cesium.Cartesian3.add(
    particle.velocity,
    snowGravityScratch,
    particle.velocity,
  );
  const distance = Cesium.Cartesian3.distance(
    scene.camera.position,
    particle.position,
  );
  if (distance > snowRadius) {
    particle.endColor.alpha = 0.0;
  } else {
    particle.endColor.alpha = 1.0 / (distance / snowRadius + 0.1);
  }
};
  // scene.primitives.removeAll();
  scene.primitives.add(
    new Cesium.ParticleSystem({
      modelMatrix: new Cesium.Matrix4.fromTranslation(scene.camera.position),
      minimumSpeed: -1.0,
      maximumSpeed: 0.0,
      lifetime: 15.0,
      emitter: new Cesium.SphereEmitter(snowRadius),
      startScale: 0.5,
      endScale: 1.0,
      image: "/static/snowflake_particle.png",
      emissionRate: 7000.0,
      startColor: Cesium.Color.WHITE.withAlpha(0.0),
      endColor: Cesium.Color.WHITE.withAlpha(1.0),
      minimumImageSize: minimumSnowImageSize,
      maximumImageSize: maximumSnowImageSize,
      updateCallback: snowUpdate,
    }),
  );

  scene.skyAtmosphere.hueShift = -0.8;
  scene.skyAtmosphere.saturationShift = -0.7;
  scene.skyAtmosphere.brightnessShift = -0.33;
  scene.fog.density = 0.001;
  scene.fog.minimumBrightness = 0.8;
// }


  // scene.primitives.add(system);
}



// // Create a rectangle entity
// const rainRect = viewer.entities.add({
//   rectangle: {
//     coordinates: rectangle,
//     material: Cesium.Color.BLUE.withAlpha(0.2),
//     heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
//     height: 0,
//   }
// });

// addWeatherEffectForRectangle(viewer, rainRect, "rain");


function animation_setup()
{
  
let executedForward = new Set();
let executedReverse = new Set();




// === Setup clock ===
viewer.clock.shouldAnimate = true;
viewer.clock.startTime = Cesium.JulianDate.fromDate(new Date());
viewer.clock.currentTime = viewer.clock.startTime;
viewer.clock.stopTime = Cesium.JulianDate.addSeconds(viewer.clock.startTime, 60, new Cesium.JulianDate());
viewer.clock.clockRange = Cesium.ClockRange.CLAMPED;
viewer.timeline.zoomTo(viewer.clock.startTime, viewer.clock.stopTime);
viewer.clock.multiplier = 2;




let lastTime = viewer.clock.startTime;

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

}


  });






