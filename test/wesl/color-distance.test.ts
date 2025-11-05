import { expect, test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("colorDistance", async () => {
  const src = `
     import lygia::color::distance::colorDistance;

     @compute @workgroup_size(1)
     fn foo() {
       let red = vec3f(1.0, 0.0, 0.0);
       let blue = vec3f(0.0, 0.0, 1.0);
       let distance = colorDistance(red, blue);
       test::results[0] = distance;
     }
   `;
  const result = await lygiaTestCompute(src);
  // Default is CIE94 distance between red and blue (0-100 scale)
  expectCloseTo([71.0491], result);
});

test("colorDistance4", async () => {
  const src = `
     import lygia::color::distance::colorDistance4;

     @compute @workgroup_size(1)
     fn foo() {
       let red = vec4f(1.0, 0.0, 0.0, 0.8);
       let blue = vec4f(0.0, 0.0, 1.0, 0.6);
       let distance = colorDistance4(red, blue);
       test::results[0] = distance;
     }
   `;
  const result = await lygiaTestCompute(src);
  // Alpha is ignored, should be same as colorDistance (0-100 scale)
  expectCloseTo([71.0491], result);
});

test("colorDistanceLAB", async () => {
  const src = `
     import lygia::color::distance::colorDistanceLAB;

     @compute @workgroup_size(1)
     fn foo() {
       let color1 = vec3f(1.0, 0.0, 0.0); // Red
       let color2 = vec3f(0.0, 0.0, 1.0); // Blue
       let result = colorDistanceLAB(color1, color2);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src);
  // LAB Euclidean distance between red and blue in LAB color space (0-100 scale)
  // This is a perceptual color distance metric
  expectCloseTo([176.314], result);
});

test("colorDistanceLABCIE94", async () => {
  const src = `
     import lygia::color::distance::colorDistanceLABCIE94;

     @compute @workgroup_size(1)
     fn foo() {
       let green = vec3f(0.0, 1.0, 0.0);
       let yellow = vec3f(1.0, 1.0, 0.0);
       let distance = colorDistanceLABCIE94(green, yellow);
       test::results[0] = distance;
     }
   `;
  const result = await lygiaTestCompute(src);
  // CIE94 distance between green and yellow (0-100 scale)
  // These are relatively close colors in perceptual space
  expectCloseTo([10.0626], result);
});

test("colorDistanceOKLAB", async () => {
  const src = `
     import lygia::color::distance::colorDistanceOKLAB;

     @compute @workgroup_size(1)
     fn foo() {
       let red = vec3f(1.0, 0.0, 0.0);
       let orange = vec3f(1.0, 0.5, 0.0);
       let blue = vec3f(0.0, 0.0, 1.0);

       // Perceptual distances
       let redToOrange = colorDistanceOKLAB(red, orange);
       let redToBlue = colorDistanceOKLAB(red, blue);

       test::results[0] = vec4f(redToOrange, redToBlue, 0.0, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  const redToOrange = result[0];
  const redToBlue = result[1];

  // Orange is perceptually closer to red than blue is
  expect(redToOrange).toBeLessThan(redToBlue);

  // Red-orange should be relatively small (similar hues)
  expect(redToOrange).toBeGreaterThan(0.05);
  expect(redToOrange).toBeLessThan(0.3);

  // Red-blue should be larger (opposite hues)
  expect(redToBlue).toBeGreaterThan(0.3);

  // Regression check - exact OKLAB distance values
  expectCloseTo([0.2917, 0.5371], [redToOrange, redToBlue]);
});

test("colorDistanceYCbCr", async () => {
  const src = `
     import lygia::color::distance::colorDistanceYCbCr;

     @compute @workgroup_size(1)
     fn foo() {
       // Test 1: Different chrominance (red vs blue)
       let red = vec3f(0.8, 0.2, 0.2);
       let blue = vec3f(0.2, 0.2, 0.8);
       let chromaDist = colorDistanceYCbCr(red, blue);

       // Test 2: Same chrominance, different luma (should be ~0)
       // Dark gray vs light gray (same neutral chroma)
       let darkGray = vec3f(0.3, 0.3, 0.3);
       let lightGray = vec3f(0.7, 0.7, 0.7);
       let lumaDist = colorDistanceYCbCr(darkGray, lightGray);

       test::results[0] = vec4f(chromaDist, lumaDist, 0.0, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  const chromaDist = result[0];
  const lumaDist = result[1];

  // Different chrominance should produce measurable distance
  expect(chromaDist).toBeGreaterThan(0.5);
  expect(chromaDist).toBeLessThan(1.5);

  // Same chrominance (grays) should have near-zero distance
  // (YCbCr distance ignores Y/luma)
  expectCloseTo([0.0], [lumaDist]);

  // Regression check - exact YCbCr chroma distance
  expectCloseTo([0.5316], [chromaDist]);
});

test("colorDistanceYPbPr", async () => {
  const src = `
     import lygia::color::distance::colorDistanceYPbPr;

     @compute @workgroup_size(1)
     fn foo() {
       // Complementary colors (magenta vs cyan)
       let magenta = vec3f(1.0, 0.0, 1.0);
       let cyan = vec3f(0.0, 1.0, 1.0);
       let complementaryDist = colorDistanceYPbPr(magenta, cyan);

       // Similar colors (cyan vs blue)
       let blue = vec3f(0.0, 0.0, 1.0);
       let similarDist = colorDistanceYPbPr(cyan, blue);

       // Test symmetry
       let dist1 = colorDistanceYPbPr(magenta, cyan);
       let dist2 = colorDistanceYPbPr(cyan, magenta);

       test::results[0] = vec4f(complementaryDist, similarDist, dist1, dist2);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  const complementaryDist = result[0];
  const similarDist = result[1];
  const dist1 = result[2];
  const dist2 = result[3];

  // Complementary colors should have larger distance than similar colors
  expect(complementaryDist).toBeGreaterThan(similarDist);

  // Distance should be symmetric
  expectCloseTo([dist1], [dist2]);

  // Complementary colors should have significant distance
  expect(complementaryDist).toBeGreaterThan(0.5);

  // Regression check - exact YPbPr distance values
  expectCloseTo([0.9919, 0.5957], [complementaryDist, similarDist]);
});

test("colorDistanceYUV", async () => {
  const src = `
     import lygia::color::distance::colorDistanceYUV;

     @compute @workgroup_size(1)
     fn foo() {
       let white = vec3f(1.0, 1.0, 1.0);
       let gray = vec3f(0.5, 0.5, 0.5);
       let distance = colorDistanceYUV(white, gray);
       test::results[0] = distance;
     }
   `;
  const result = await lygiaTestCompute(src);
  // YUV distance between white and gray (mainly Y difference)
  // Should be around 0.5 (difference in luminance)
  expectCloseTo([0.5], result);
});
