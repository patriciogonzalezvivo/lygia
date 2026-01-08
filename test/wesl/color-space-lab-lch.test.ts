import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("lab2srgb", async () => {
  const src = `
     import lygia::color::space::lab2srgb::lab2srgb;

     @compute @workgroup_size(1)
     fn foo() {
       let lab = vec3f(50.0, 25.0, -25.0);
       let result = lab2srgb(lab);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Lab(50, 25, -25) -> sRGB conversion
  expectCloseTo([0.5524, 0.413, 0.634], result);
});

test("lab2rgb", async () => {
  const src = `
     import lygia::color::space::lab2rgb::lab2rgb;

     @compute @workgroup_size(1)
     fn foo() {
       let lab = vec3f(53.24, 80.09, 67.20); // Red in LAB
       let result = lab2rgb(lab);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // LAB(53.24, 80.09, 67.20) -> RGB(1, 0, 0)
  expectCloseTo([1.0, 0.0, 0.0], result);
});

test("srgb2lab", async () => {
  const src = `
     import lygia::color::space::srgb2lab::srgb2lab;

     @compute @workgroup_size(1)
     fn foo() {
       let srgb = vec3f(1.0, 0.0, 0.0); // sRGB Red
       let result = srgb2lab(srgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // sRGB Red -> LAB (L* now in 0-100 scale, matching standard Lab convention)
  expectCloseTo([53.2408, 80.0925, 67.2032], result);
});

test("rgb2lab", async () => {
  const src = `
     import lygia::color::space::rgb2lab::rgb2lab;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec3f(0.5, 0.5, 0.5); // Gray
       let result = rgb2lab(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Gray in LAB (L* now in 0-100 scale)
  expectCloseTo([76.0693, 0, 0.00001], result);
});

test("lch2srgb3", async () => {
  const src = `
     import lygia::color::space::lch2srgb::lch2srgb;

     @compute @workgroup_size(1)
     fn foo() {
       let lch = vec3f(50.0, 30.0, 120.0);
       let result = lch2srgb(lch);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // LCh(50, 30, 120°) -> Lab -> sRGB conversion
  expectCloseTo([0.4277, 0.4903, 0.2895], result);
});

test("lch2rgb", async () => {
  const src = `
     import lygia::color::space::lch2rgb::lch2rgb;

     @compute @workgroup_size(1)
     fn foo() {
       let lch = vec3f(53.24, 104.55, 40.0); // Red in LCH
       let result = lch2rgb(lch);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // LCH -> RGB
  expectCloseTo([1.0, 0.0, 0.0], result);
});

test("srgb2lch", async () => {
  const src = `
     import lygia::color::space::srgb2lch::srgb2lch;

     @compute @workgroup_size(1)
     fn foo() {
       let srgb = vec3f(1.0, 0.0, 0.0); // sRGB Red
       let result = srgb2lch(srgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // sRGB Red -> LCH (L now in 0-100 scale, matching standard LCH convention)
  expectCloseTo([53.2408, 104.5518, 39.999], result);
});

test("rgb2lch", async () => {
  const src = `
     import lygia::color::space::rgb2lch::rgb2lch;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec3f(1.0, 0.0, 0.0); // Red
       let result = rgb2lch(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Red in LCH (L now in 0-100 scale)
  expectCloseTo([53.2408, 104.5518, 39.999], result);
});

test("lab2lch", async () => {
  const src = `
     import lygia::color::space::lab2lch::lab2lch;

     @compute @workgroup_size(1)
     fn foo() {
       let lab = vec3f(50.0, 25.0, 25.0);
       let result = lab2lch(lab);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // LAB(50, 25, 25) -> LCH(50, ~35.36, 45°)
  // L stays same, C = sqrt(a^2 + b^2), H = atan2(b, a)
  expectCloseTo([50.0, 35.3553, 45.0], result);
});

test("lch2lab", async () => {
  const src = `
     import lygia::color::space::lch2lab::lch2lab;

     @compute @workgroup_size(1)
     fn foo() {
       let lch = vec3f(50.0, 35.36, 45.0);
       let result = lch2lab(lch);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // LCH(50, 35.36, 45°) -> LAB(50, ~25, ~25)
  expectCloseTo([50.0, 25.0033, 25.0033], result);
});

test("lab2xyz", async () => {
  const src = `
     import lygia::color::space::lab2xyz::lab2xyz;

     @compute @workgroup_size(1)
     fn foo() {
       let lab = vec3f(50.0, 0.0, 0.0); // Neutral gray in LAB
       let result = lab2xyz(lab);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // WESL uses 0-100 scale for XYZ (colorimetry standard)
  // LAB(50, 0, 0) -> XYZ (uses D65 white point scaling)
  expectCloseTo([17.5061, 18.4186, 20.059], result);
});

test("xyz2lab", async () => {
  const src = `
     import lygia::color::space::xyz2lab::xyz2lab;

     @compute @workgroup_size(1)
     fn foo() {
       let xyz = vec3f(17.5, 18.4, 20.0); // Neutral gray from lab2xyz
       let result = xyz2lab(xyz);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // XYZ -> LAB (actual output)
  expectCloseTo([49.9777, 0.0615, 0.0653], result);
});

test("lab2lch4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::lab2lch::lab2lch4;

     @compute @workgroup_size(1)
     fn foo() {
       let lab = vec4f(50.0, 25.0, 25.0, 0.75);
       let result = lab2lch4(lab);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([50.0, 35.3553, 45.0, 0.75], result);
});

test("lab2rgb4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::lab2rgb::lab2rgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let lab = vec4f(53.24, 80.09, 67.20, 0.4); // Red with alpha
       let result = lab2rgb4(lab);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([1.0, 0.0, 0.0, 0.4], result);
});

test("lab2srgb4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::lab2srgb::lab2srgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let lab = vec4f(50.0, 25.0, -25.0, 0.85);
       let result = lab2srgb4(lab);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.5524, 0.413, 0.634, 0.85], result);
});

test("lab2xyz4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::lab2xyz::lab2xyz4;

     @compute @workgroup_size(1)
     fn foo() {
       let lab = vec4f(50.0, 0.0, 0.0, 0.3); // Neutral gray with alpha
       let result = lab2xyz4(lab);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // WESL uses 0-100 scale for XYZ
  expectCloseTo([17.5061, 18.4186, 20.059, 0.3], result);
});

test("lch2lab4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::lch2lab::lch2lab4;

     @compute @workgroup_size(1)
     fn foo() {
       let lch = vec4f(50.0, 35.36, 45.0, 0.95);
       let result = lch2lab4(lch);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([50.0, 25.0033, 25.0033, 0.95], result);
});

test("lch2rgb4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::lch2rgb::lch2rgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let lch = vec4f(53.24, 104.55, 40.0, 0.2); // Red with alpha
       let result = lch2rgb4(lch);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([1.0, 0.0, 0.0, 0.2], result);
});

test("lch2srgb4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::lch2srgb::lch2srgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let lch = vec4f(50.0, 30.0, 120.0, 0.65);
       let result = lch2srgb4(lch);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.4277, 0.4903, 0.2895, 0.65], result);
});

test("rgb2lab4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::rgb2lab::rgb2lab4;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec4f(0.5, 0.5, 0.5, 0.7); // Gray with alpha
       let result = rgb2lab4(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([76.0693, 0, 0.00001, 0.7], result);
});

test("rgb2lch4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::rgb2lch::rgb2lch4;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec4f(1.0, 0.0, 0.0, 0.8); // Red with alpha
       let result = rgb2lch4(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([53.2408, 104.5518, 39.999, 0.8], result);
});

test("srgb2lab4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::srgb2lab::srgb2lab4;

     @compute @workgroup_size(1)
     fn foo() {
       let srgb = vec4f(1.0, 0.0, 0.0, 0.75); // sRGB Red with alpha
       let result = srgb2lab4(srgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([53.2408, 80.0925, 67.2032, 0.75], result);
});

test("srgb2lch4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::srgb2lch::srgb2lch4;

     @compute @workgroup_size(1)
     fn foo() {
       let srgb = vec4f(1.0, 0.0, 0.0, 0.5); // sRGB Red with alpha
       let result = srgb2lch4(srgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([53.2408, 104.5518, 39.999, 0.5], result);
});

test("xyz2lab4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::xyz2lab::xyz2lab4;

     @compute @workgroup_size(1)
     fn foo() {
       let xyz = vec4f(17.5, 18.4, 20.0, 0.9); // Neutral gray with alpha
       let result = xyz2lab4(xyz);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([49.9777, 0.06151, 0.06528, 0.9], result);
});
