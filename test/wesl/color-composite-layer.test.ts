import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("layerMultiplySourceOver4", async () => {
  const src = `
    import lygia::color::layer::multiplySourceOver::layerMultiplySourceOver4;

    @compute @workgroup_size(1)
    fn foo() {
      let src = vec4f(0.8, 0.6, 0.4, 0.75);
      let dst = vec4f(0.5, 0.7, 0.9, 0.5);
      let result = layerMultiplySourceOver4(src, dst);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Multiply blend with source-over compositing
  expectCloseTo([0.3625, 0.4025, 0.3825, 0.875], result);
});

test("layerScreenSourceOver4", async () => {
  const src = `
    import lygia::color::layer::screenSourceOver::layerScreenSourceOver4;

    @compute @workgroup_size(1)
    fn foo() {
      let src = vec4f(0.6, 0.4, 0.2, 0.5);
      let dst = vec4f(0.3, 0.5, 0.7, 0.6);
      let result = layerScreenSourceOver4(src, dst);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Screen blend with source-over compositing
  expectCloseTo([0.45, 0.5, 0.59, 0.8], result);
});

test("layerAddSourceOver4", async () => {
  const src = `
    import lygia::color::layer::addSourceOver::layerAddSourceOver4;

    @compute @workgroup_size(1)
    fn foo() {
      let src = vec4f(0.3, 0.4, 0.5, 0.6);
      let dst = vec4f(0.2, 0.3, 0.4, 0.5);
      let result = layerAddSourceOver4(src, dst);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Add blend with source-over compositing
  expectCloseTo([0.34, 0.48, 0.62, 0.8], result);
});

test("layerOverlaySourceOver4", async () => {
  const src = `
    import lygia::color::layer::overlaySourceOver::layerOverlaySourceOver4;

    @compute @workgroup_size(1)
    fn foo() {
      let src = vec4f(0.7, 0.3, 0.5, 0.8);
      let dst = vec4f(0.4, 0.6, 0.5, 0.4);
      let result = layerOverlaySourceOver4(src, dst);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Overlay blend with source-over compositing
  expectCloseTo([0.544, 0.336, 0.44, 0.88], result);
});

test("layerDarkenSourceOver4", async () => {
  const src = `
    import lygia::color::layer::darkenSourceOver::layerDarkenSourceOver4;

    @compute @workgroup_size(1)
    fn foo() {
      let src = vec4f(0.3, 0.7, 0.5, 0.5);
      let dst = vec4f(0.6, 0.4, 0.5, 0.5);
      let result = layerDarkenSourceOver4(src, dst);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Darken blend with source-over compositing
  expectCloseTo([0.3, 0.3, 0.375, 0.75], result);
});

test("layerLightenSourceOver4", async () => {
  const src = `
    import lygia::color::layer::lightenSourceOver::layerLightenSourceOver4;

    @compute @workgroup_size(1)
    fn foo() {
      let src = vec4f(0.3, 0.7, 0.5, 0.5);
      let dst = vec4f(0.6, 0.4, 0.5, 0.5);
      let result = layerLightenSourceOver4(src, dst);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Lighten blend with source-over compositing
  expectCloseTo([0.45, 0.45, 0.375, 0.75], result);
});

test("layerDifferenceSourceOver4", async () => {
  const src = `
    import lygia::color::layer::differenceSourceOver::layerDifferenceSourceOver4;

    @compute @workgroup_size(1)
    fn foo() {
      let src = vec4f(0.8, 0.3, 0.6, 0.7);
      let dst = vec4f(0.5, 0.7, 0.4, 0.6);
      let result = layerDifferenceSourceOver4(src, dst);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Difference blend with source-over compositing
  expectCloseTo([0.3, 0.406, 0.212, 0.88], result);
});

test("layerExclusionSourceOver4", async () => {
  const src = `
    import lygia::color::layer::exclusionSourceOver::layerExclusionSourceOver4;

    @compute @workgroup_size(1)
    fn foo() {
      let src = vec4f(0.6, 0.4, 0.8, 0.5);
      let dst = vec4f(0.3, 0.7, 0.2, 0.5);
      let result = layerExclusionSourceOver4(src, dst);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Exclusion blend with source-over compositing
  expectCloseTo([0.345, 0.445, 0.39, 0.75], result);
});

test("layerPhoenixSourceOver4", async () => {
  const src = `
    import lygia::color::layer::phoenixSourceOver::layerPhoenixSourceOver4;

    @compute @workgroup_size(1)
    fn foo() {
      let src = vec4f(0.7, 0.5, 0.3, 0.6);
      let dst = vec4f(0.4, 0.6, 0.8, 0.4);
      let result = layerPhoenixSourceOver4(src, dst);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Phoenix blend with source-over compositing
  expectCloseTo([0.484, 0.636, 0.428, 0.76], result);
});

test("layerSubtractSourceOver4", async () => {
  const src = `
    import lygia::color::layer::subtractSourceOver::layerSubtractSourceOver4;

    @compute @workgroup_size(1)
    fn foo() {
      let src = vec4f(0.8, 0.5, 0.3, 0.5);
      let dst = vec4f(0.4, 0.6, 0.7, 0.5);
      let result = layerSubtractSourceOver4(src, dst);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Subtract blend with source-over compositing
  expectCloseTo([0.2, 0.2, 0.175, 0.75], result);
});
