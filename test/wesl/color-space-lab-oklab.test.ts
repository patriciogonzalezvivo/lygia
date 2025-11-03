import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("oklab2srgb", async () => {
  const src = `
     import lygia::color::space::oklab2srgb::oklab2srgb;

     @compute @workgroup_size(1)
     fn foo() {
       let oklab = vec3f(0.628, 0.225, 0.126); // Red in Oklab
       let result = oklab2srgb(oklab);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Oklab -> sRGB
  expectCloseTo([1.0, 0.0, 0.0], result);
});

test("srgb2oklab", async () => {
  const src = `
     import lygia::color::space::srgb2oklab::srgb2oklab;

     @compute @workgroup_size(1)
     fn foo() {
       let srgb = vec3f(1.0, 0.0, 0.0); // Red
       let result = srgb2oklab(srgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // sRGB(1, 0, 0) -> Oklab
  expectCloseTo([0.628, 0.2249, 0.1258], result);
});

test("oklab2rgb", async () => {
  const src = `
     import lygia::color::space::oklab2rgb::oklab2rgb;

     @compute @workgroup_size(1)
     fn foo() {
       let oklab = vec3f(0.628, 0.225, 0.126); // Red in Oklab
       let result = oklab2rgb(oklab);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Oklab(0.628, 0.225, 0.126) -> RGB(1, 0, 0)
  expectCloseTo([1.0008, -0.0002, -0.0002], result);
});

test("rgb2oklab", async () => {
  const src = `
     import lygia::color::space::rgb2oklab::rgb2oklab;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec3f(1.0, 0.0, 0.0); // Red
       let result = rgb2oklab(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // RGB(1, 0, 0) -> Oklab
  expectCloseTo([0.628, 0.2249, 0.1258], result);
});

test("oklab2rgb4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::oklab2rgb::oklab2rgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let oklab = vec4f(0.628, 0.225, 0.126, 0.45); // Red with alpha
       let result = oklab2rgb4(oklab);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([1.0008, -0.0002, -0.0002, 0.45], result);
});

test("oklab2srgb4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::oklab2srgb::oklab2srgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let oklab = vec4f(0.628, 0.225, 0.126, 0.35); // Red with alpha
       let result = oklab2srgb4(oklab);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([1.0, 0.0, 0.0, 0.35], result);
});

test("rgb2oklab4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::rgb2oklab::rgb2oklab4;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec4f(1.0, 0.0, 0.0, 0.6); // Red with alpha
       let result = rgb2oklab4(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.628, 0.2249, 0.1258, 0.6], result);
});

test("srgb2oklab4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::srgb2oklab::srgb2oklab4;

     @compute @workgroup_size(1)
     fn foo() {
       let srgb = vec4f(1.0, 0.0, 0.0, 0.85); // Red with alpha
       let result = srgb2oklab4(srgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.62796, 0.22486, 0.12585, 0.85], result);
});
