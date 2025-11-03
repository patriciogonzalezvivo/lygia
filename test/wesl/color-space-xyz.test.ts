import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("rgb2xyz", async () => {
  const src = `
	import lygia::color::space::rgb2xyz::rgb2xyz;

	@compute @workgroup_size(1)
	fn foo() {
		test::results[0] = rgb2xyz(vec3f(.8, .7, .5));
	}
	`;

  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  expectCloseTo([67.0487, 70.6832, 57.4054], result);

  const cie = { CIE_D50: true };
  const resultCie = await lygiaTestCompute(src, {
    elem: "vec3f",
    conditions: cie,
  });
  expectCloseTo([68.9945, 71.0127, 43.6206], resultCie);
});

test("srgb2xyz", async () => {
  const src = `
     import lygia::color::space::srgb2xyz::srgb2xyz;

     @compute @workgroup_size(1)
     fn foo() {
       let srgb = vec3f(1.0, 0.0, 0.0); // Red
       let result = srgb2xyz(srgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // WESL uses 0-100 scale for XYZ
  expectCloseTo([41.2456, 21.2673, 1.9334], result);
});

test("xyY2rgb", async () => {
  const src = `
     import lygia::color::space::xyY2rgb::xyY2rgb;

     @compute @workgroup_size(1)
     fn foo() {
       let xyY = vec3f(0.64, 0.33, 21.26); // Red (Y in 0-100 scale)
       let result = xyY2rgb(xyY);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // xyY -> RGB (roundtrip should restore original RGB values, 0.001 tolerance for accumulated error)
  expectCloseTo([1.0, 0.0, 0.0], result, 0.001);
});

test("rgb2xyY", async () => {
  const src = `
     import lygia::color::space::rgb2xyY::rgb2xyY;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec3f(1.0, 0.0, 0.0); // Red
       let result = rgb2xyY(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // WESL: x,y chromaticity 0-1, Y luminance 0-100 (matches XYZ scale)
  expectCloseTo([0.64, 0.33, 21.2673], result);
});

test("xyY2srgb", async () => {
  const src = `
     import lygia::color::space::xyY2srgb::xyY2srgb;

     @compute @workgroup_size(1)
     fn foo() {
       let xyY = vec3f(0.64, 0.33, 21.26); // Red (Y in 0-100 scale)
       let result = xyY2srgb(xyY);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // xyY -> XYZ (0-100 scale) -> RGB(1,0,0) -> sRGB(1,0,0)
  expectCloseTo([1.0, 0.0, 0.0], result, 0.001);
});

test("xyY2xyz", async () => {
  const src = `
     import lygia::color::space::xyY2xyz::xyY2xyz;

     @compute @workgroup_size(1)
     fn foo() {
       let xyY = vec3f(0.3127, 0.3290, 1.0); // D65 white point
       let result = xyY2xyz(xyY);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // xyY Y component is already 0-100 scale, so Y=1 stays as 1
  // x,y chromaticity coordinates scale proportionally with Y
  expectCloseTo([0.9505, 1.0, 1.089], result, 0.01);
});

test("xyz2xyY", async () => {
  const src = `
     import lygia::color::space::xyz2xyY::xyz2xyY;

     @compute @workgroup_size(1)
     fn foo() {
       let xyz = vec3f(0.9505, 1.0, 1.089); // D65 white
       let result = xyz2xyY(xyz);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // XYZ -> xyY
  expectCloseTo([0.3127, 0.329, 1.0], result);
});

test("xyz2srgb", async () => {
  const src = `
     import lygia::color::space::xyz2srgb::xyz2srgb;

     @compute @workgroup_size(1)
     fn foo() {
       let xyz = vec3f(41.24, 21.26, 1.93); // Red in XYZ (0-100 scale)
       let result = xyz2srgb(xyz);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // WESL uses 0-100 scale: XYZ(41.24, 21.26, 1.93) -> RGB(1,0,0) -> sRGB(1,0,0)
  expectCloseTo([1.0, 0.0, 0.0], result, 0.001);
});

test("xyz2rgb", async () => {
  const src = `
     import lygia::color::space::xyz2rgb::xyz2rgb;

     @compute @workgroup_size(1)
     fn foo() {
       let xyz = vec3f(41.24, 21.26, 1.93); // Red in XYZ (0-100 scale)
       let result = xyz2rgb(xyz);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // WESL uses 0-100 scale for XYZ (colorimetry standard)
  // XYZ(41.24, 21.26, 1.93) -> RGB(1, 0, 0)
  expectCloseTo([1.0, 0.0, 0.0], result);
});

test("rgb2xyz4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::rgb2xyz::rgb2xyz4;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec4f(0.8, 0.7, 0.5, 0.2);
       let result = rgb2xyz4(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // WESL uses 0-100 scale for XYZ
  expectCloseTo([67.0487, 70.6832, 57.4054, 0.2], result);
});

test("xyz2rgb4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::xyz2rgb::xyz2rgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let xyz = vec4f(41.24, 21.26, 1.93, 0.6); // Red with alpha
       let result = xyz2rgb4(xyz);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([1.0, 0.0, 0.0, 0.6], result);
});

test("rgb2xyY4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::rgb2xyY::rgb2xyY4;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec4f(1.0, 0.0, 0.0, 0.4); // Red with alpha
       let result = rgb2xyY4(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // WESL uses 0-100 scale for Y: RGB(1, 0, 0) -> xyY (x,y in 0-1, Y in 0-100)
  expectCloseTo([0.64, 0.33, 21.26, 0.4], result, 0.01);
});

test("srgb2xyz4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::srgb2xyz::srgb2xyz4;

     @compute @workgroup_size(1)
     fn foo() {
       let srgb = vec4f(1.0, 0.0, 0.0, 0.65); // Red with alpha
       let result = srgb2xyz4(srgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // WESL uses 0-100 scale: sRGB(1, 0, 0) -> XYZ with alpha
  expectCloseTo([41.24, 21.26, 1.93, 0.65], result, 0.01);
});

test("xyY2rgb4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::xyY2rgb::xyY2rgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let xyY = vec4f(0.64, 0.33, 21.26, 0.5); // Red with alpha (Y in 0-100 scale)
       let result = xyY2rgb4(xyY);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // xyY -> RGB (vec4 overload with alpha preservation, 0.001 tolerance for accumulated error)
  expectCloseTo([1.0, 0.0, 0.0, 0.5], result, 0.001);
});

test("xyY2srgb4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::xyY2srgb::xyY2srgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let xyY = vec4f(0.64, 0.33, 21.26, 0.85); // Red with alpha (Y in 0-100 scale)
       let result = xyY2srgb4(xyY);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // xyY -> XYZ (0-100 scale) -> RGB -> sRGB with alpha
  expectCloseTo([1.0, 0.0, 0.0, 0.85], result, 0.001);
});

test("xyY2xyz4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::xyY2xyz::xyY2xyz4;

     @compute @workgroup_size(1)
     fn foo() {
       let xyY = vec4f(0.3127, 0.3290, 1.0, 0.4); // D65 white with alpha
       let result = xyY2xyz4(xyY);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // xyY Y component is already 0-100 scale, so Y=1 stays as 1
  expectCloseTo([0.9505, 1.0, 1.089, 0.4], result, 0.01);
});

test("xyz2srgb4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::xyz2srgb::xyz2srgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let xyz = vec4f(41.24, 21.26, 1.93, 0.2); // Red with alpha (0-100 scale)
       let result = xyz2srgb4(xyz);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // WESL uses 0-100 scale: XYZ -> RGB -> sRGB with alpha
  expectCloseTo([1.0, 0.0, 0.0, 0.2], result, 0.001);
});

test("xyz2xyY4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::xyz2xyY::xyz2xyY4;

     @compute @workgroup_size(1)
     fn foo() {
       let xyz = vec4f(0.9505, 1.0, 1.089, 0.75); // D65 white with alpha
       let result = xyz2xyY4(xyz);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // XYZ -> xyY with alpha
  expectCloseTo([0.3127, 0.329, 1.0, 0.75], result);
});
