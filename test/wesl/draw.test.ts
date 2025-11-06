import { afterAll, beforeAll, test } from "vitest";
import { imageMatcher } from "vitest-image-snapshot";
import { destroySharedDevice, getGPUDevice } from "wesl-test";
import { lygiaExampleImage } from "./testUtil.ts";
import "./shaders/draw-shapes.wesl?raw"; // not used, but nice to trigger watch mode rebuild in vitest

imageMatcher();

let device: GPUDevice;

beforeAll(async () => {
  device = await getGPUDevice();
});

afterAll(() => {
  destroySharedDevice();
});

test("stroke and strokeEdge - grid pattern", async () => {
  // Renders a 4x4 grid of circles:
  // Top 2 rows: stroke() with varying widths (0.05 to 0.2)
  // Bottom 2 rows: strokeEdge() with varying edge smoothness (0.001 to 0.051)
  await lygiaExampleImage(device, "draw-stroke" );
});

test("2D SDF shapes - 14 shape grid (matches GLSL draw_shapes)", async () => {
  // Renders a 4×4 grid showing 14 different 2D SDF shapes with fill():
  // Row 1: circleSDF, vesicaSDF, rhombSDF, triSDF
  // Row 2: rectSDF, polySDF (pentagon), hexSDF, starSDF (5 points)
  // Row 3: flowerSDF (5 petals), crossSDF, gearSDF (10 teeth), heartSDF
  // Row 4: raysSDF (14 rays), spiralSDF, (empty), (empty)
  //
  // Coordinate system notes:
  // - WGSL Y-down requires negating rotation angle to maintain visual direction
  // - SDF functions using atan2 negate Y internally to match GLSL tooth/ray patterns
  // - Grid indexing uses natural Y-down order (no flip needed unlike GLSL)
  await lygiaExampleImage(device, "draw-shapes", {
    size: [512, 512],
    uniforms: {
      time: .2155,  // Rotation to match GLSL reference image
    }
  });
});
