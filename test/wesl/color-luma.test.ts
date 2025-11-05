import { expect, test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("luminance", async () => {
  const src = `
     import lygia::color::luminance::luminance;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(1.0, 0.5, 0.0); // Orange color
       let result = luminance(color);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src);
  // Luminance calculation: 1*0.2125 + 0.5*0.7154 + 0*0.0721 = 0.2125 + 0.3577 = 0.5702
  expectCloseTo([0.5702], result);
});

test("luminance4", async () => {
  const src = `
     import lygia::color::luminance::luminance4;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(1.0, 0.5, 0.0, 0.8); // Orange color with alpha
       let result = luminance4(color);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src);
  // Should be same as luminance test (alpha ignored)
  expectCloseTo([0.5702], result);
});

test("luma", async () => {
  const src = `
     import lygia::color::luma::luma3;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec3f(1.0, 0.5, 0.0); // Orange
       let result = luma3(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src);
  // Luma using Rec709 coefficients: 1.0*0.2126 + 0.5*0.7152 + 0.0*0.0722 = 0.5702
  expectCloseTo([0.5702], result);
});

test("luma - grayscale consistency", async () => {
  const src = `
     import lygia::color::luma::luma;
     import lygia::color::luma::luma3;

     @compute @workgroup_size(1)
     fn foo() {
       let value = 0.75;
       let gray = vec3f(0.75, 0.75, 0.75);

       let scalarLuma = luma(value);
       let vectorLuma = luma3(gray);

       // For grayscale, both should give same result
       test::results[0] = vec4f(scalarLuma, vectorLuma, 0.0, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Grayscale color should have luma equal to its value
  expectCloseTo([0.75, 0.75], [result[0], result[1]]);
});

test("luma4", async () => {
  const src = `
     import lygia::color::luma::luma4;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.8, 0.3, 0.1, 0.6); // Orange-ish with alpha
       let result = luma4(color);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src);
  // Luma using Rec709: 0.8*0.2126 + 0.3*0.7152 + 0.1*0.0722
  // = 0.17008 + 0.21456 + 0.00722 = 0.39186
  expectCloseTo([0.39186], result);
});
