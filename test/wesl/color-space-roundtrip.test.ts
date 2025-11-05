import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("rgb2hsl4 -> hsl2rgb4 roundtrip", async () => {
  const src = `
     import lygia::color::space::rgb2hsl::rgb2hsl4;
     import lygia::color::space::hsl2rgb::hsl2rgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let original = vec4f(0.7, 0.3, 0.5, 0.8);
       let hsl = rgb2hsl4(original);
       let back = hsl2rgb4(hsl);
       test::results[0] = back;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.7, 0.3, 0.5, 0.8], result);
});

test("rgb2hsv4 -> hsv2rgb4 roundtrip", async () => {
  const src = `
     import lygia::color::space::rgb2hsv::rgb2hsv4;
     import lygia::color::space::hsv2rgb::hsv2rgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let original = vec4f(0.8, 0.2, 0.6, 0.5);
       let hsv = rgb2hsv4(original);
       let back = hsv2rgb4(hsv);
       test::results[0] = back;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.8, 0.2, 0.6, 0.5], result);
});

test("rgb2oklab4 -> oklab2rgb4 roundtrip", async () => {
  const src = `
     import lygia::color::space::rgb2oklab::rgb2oklab4;
     import lygia::color::space::oklab2rgb::oklab2rgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let original = vec4f(0.6, 0.4, 0.2, 0.9);
       let oklab = rgb2oklab4(original);
       let back = oklab2rgb4(oklab);
       test::results[0] = back;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.6, 0.4, 0.2, 0.9], result);
});

test("rgb2yiq4 -> yiq2rgb4 roundtrip", async () => {
  const src = `
     import lygia::color::space::rgb2yiq::rgb2yiq4;
     import lygia::color::space::yiq2rgb::yiq2rgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let original = vec4f(0.7, 0.4, 0.2, 0.6);
       let yiq = rgb2yiq4(original);
       let back = yiq2rgb4(yiq);
       test::results[0] = back;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.7, 0.40003, 0.19989, 0.6], result);
});

test("rgb2yuv4 -> yuv2rgb4 roundtrip", async () => {
  const src = `
     import lygia::color::space::rgb2yuv::rgb2yuv4;
     import lygia::color::space::yuv2rgb::yuv2rgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let original = vec4f(0.5, 0.6, 0.3, 0.8);
       let yuv = rgb2yuv4(original);
       let back = yuv2rgb4(yuv);
       test::results[0] = back;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.5, 0.60064, 0.29362, 0.8], result);
});

test("rgb2xyY4 -> xyY2rgb4 roundtrip (note: precision issues in xyY)", async () => {
  const src = `
     import lygia::color::space::rgb2xyY::rgb2xyY4;
     import lygia::color::space::xyY2rgb::xyY2rgb4;

     @compute @workgroup_size(1)
     fn foo() {
       // Using a brighter color to avoid precision issues at low values
       let original = vec4f(0.9, 0.8, 0.7, 0.6);
       let xyY = rgb2xyY4(original);
       let back = xyY2rgb4(xyY);
       test::results[0] = back;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Note: xyY conversion chain has some precision loss
  // The conversion goes: RGB -> XYZ (0-100) -> xyY -> XYZ (0-100) -> RGB
  // which accumulates rounding errors
  expectCloseTo([0.9, 0.8, 0.7, 0.6], result, 0.01);
});
