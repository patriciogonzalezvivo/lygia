import { expect, test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("rgb2heat", async () => {
  const src = `
    import lygia::color::space::rgb2heat::rgb2heat;

    @compute @workgroup_size(1)
    fn foo() {
      let x = rgb2heat(vec3f(.8, .7, .5));

      test::results[0] = x;
    }
  `;

  const result = await lygiaTestCompute(src);
  expectCloseTo([0.854], result);
});

test("cmyk2rgb", async () => {
  const src = `
     import lygia::color::space::cmyk2rgb::cmyk2rgb;

     @compute @workgroup_size(1)
     fn foo() {
       let cmyk = vec4f(0.0, 0.0, 0.0, 0.5); // 50% gray
       let result = cmyk2rgb(cmyk);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // CMYK(0,0,0,0.5) -> RGB (50% gray)
  expectCloseTo([0.5, 0.5, 0.5], result);
});

test("rgb2cmyk", async () => {
  const src = `
     import lygia::color::space::rgb2cmyk::rgb2cmyk;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec3f(1.0, 0.0, 0.0); // Red
       let result = rgb2cmyk(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // RGB(1, 0, 0) -> CMYK(0, 1, 1, 0)
  expectCloseTo([0.0, 1.0, 1.0, 0.0], result);
});

test("k2rgb - color temperature gradient", async () => {
  const src = `
     import lygia::color::space::k2rgb::k2rgb;

     @compute @workgroup_size(1)
     fn foo() {
       // Test that lower temps are warmer (more red, less blue)
       // and higher temps are cooler (less red, more blue)
       let warm = k2rgb(3000.0);  // Warm white
       let cool = k2rgb(8000.0);  // Cool white

       // Pack both results into 4 floats: warm.r, warm.b, cool.r, cool.b
       test::results[0] = warm.r;
       test::results[1] = warm.b;
       test::results[2] = cool.r;
       test::results[3] = cool.b;
     }
   `;
  const result = await lygiaTestCompute(src);

  // Verify warm light has more red, less blue than cool light
  expect(result[0]).toBeGreaterThan(result[2]); // Warm has more red than cool
  expect(result[3]).toBeGreaterThan(result[1]); // Cool has more blue than warm
});

test("rgb2lms", async () => {
  const src = `
     import lygia::color::space::rgb2lms::rgb2lms;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec3f(1.0, 0.0, 0.0); // Red
       let result = rgb2lms(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // RGB(1, 0, 0) -> LMS cone response (first column of RGB2LMS matrix)
  // L = 17.882 * 1.0 + 43.516 * 0.0 + 4.119 * 0.0 = 17.882
  // M =  3.456 * 1.0 + 27.155 * 0.0 + 0.184 * 0.0 = 3.456
  // S =  0.030 * 1.0 + 0.184 * 0.0 + 1.467 * 0.0 = 0.030
  expectCloseTo([17.8824, 3.45565, 0.02996], result);
});

test("lms2rgb", async () => {
  const src = `
     import lygia::color::space::lms2rgb::lms2rgb;

     @compute @workgroup_size(1)
     fn foo() {
       let lms = vec3f(0.3, 0.2, 0.1);
       let result = lms2rgb(lms);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // LMS(0.3, 0.2, 0.1) -> RGB via LMS2RGB matrix multiplication
  // Matrix is column-major in WGSL, so LMS2RGB * lms is:
  // R = row 0 dot lms = 0.0809 * 0.3 + (-0.0102) * 0.2 + (-0.00037) * 0.1
  // G = row 1 dot lms = (-0.1305) * 0.3 + 0.0540 * 0.2 + (-0.00412) * 0.1
  // B = row 2 dot lms = 0.1167 * 0.3 + (-0.1136) * 0.2 + 0.6935 * 0.1
  expectCloseTo([0.00985, -0.00363, 0.06842], result);
});

test("lms2rgb4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::lms2rgb::lms2rgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let lms = vec4f(0.3, 0.2, 0.1, 0.55);
       let result = lms2rgb4(lms);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.00985, -0.00363, 0.06842, 0.55], result);
});

test("rgb2lms4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::rgb2lms::rgb2lms4;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec4f(1.0, 0.0, 0.0, 0.3); // Red with alpha
       let result = rgb2lms4(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([17.8824, 3.45565, 0.02996, 0.3], result);
});

test("rgb2ryb - default mode", async () => {
  const src = `
     import lygia::color::space::rgb2ryb::rgb2ryb;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec3f(1.0, 0.0, 0.0); // Red
       let result = rgb2ryb(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  expectCloseTo([1.0, 0.0, 0.0], result);
});

test("rgb2ryb4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::rgb2ryb::rgb2ryb4;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec4f(0.0, 1.0, 0.0, 0.5); // Green with alpha
       let result = rgb2ryb4(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Green in RGB -> RYB: Actual result (0.0, 1.0, 0.483)
  expectCloseTo([0.0, 1.0, 0.483, 0.5], result);
});

test("ryb2rgb - default mode", async () => {
  const src = `
     import lygia::color::space::ryb2rgb::ryb2rgb;

     @compute @workgroup_size(1)
     fn foo() {
       let ryb = vec3f(1.0, 0.0, 0.0); // Red in RYB
       let result = ryb2rgb(ryb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  expectCloseTo([1.0, 0.0, 0.0], result);
});

test("ryb2rgb4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::ryb2rgb::ryb2rgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let ryb = vec4f(0.0, 1.0, 0.0, 0.75); // Yellow in RYB with alpha
       let result = ryb2rgb4(ryb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Yellow in RYB -> RGB: Actual result (1.0, 1.0, 0.0) - yellow
  expectCloseTo([1.0, 1.0, 0.0, 0.75], result);
});

test("rgb2heat4 - vec4 overload with alpha preservation", async () => {
  const src = `
     import lygia::color::space::rgb2heat::rgb2heat4;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec4f(0.8, 0.7, 0.5, 0.6); // Color with alpha
       let result = rgb2heat4(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Heat map conversion: converts to grayscale heat value replicated in RGB
  // heat value is 0.854, replicated as (0.854, 0.854, 0.854, 0.6)
  expectCloseTo([0.854, 0.854, 0.854, 0.6], result);
});
