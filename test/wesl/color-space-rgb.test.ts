import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("hsl2rgb", async () => {
  const src = `
     import lygia::color::space::hsl2rgb::hsl2rgb;

     @compute @workgroup_size(1)
     fn foo() {
       let hsl = vec3f(0.5, 0.8, 0.5); // Cyan-ish color
       let result = hsl2rgb(hsl);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // HSL(180°, 80%, 50%) -> RGB
  expectCloseTo([0.1, 0.9, 0.9], result);
});

test("rgb2hsl", async () => {
  const src = `
     import lygia::color::space::rgb2hsl::rgb2hsl;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec3f(0.1, 0.9, 0.9); // Cyan-ish
       let result = rgb2hsl(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Should convert back to HSL(~0.5, ~0.8, 0.5)
  expectCloseTo([0.5, 0.8, 0.5], result);
});

test("hsv2rgb", async () => {
  const src = `
     import lygia::color::space::hsv2rgb::hsv2rgb;

     @compute @workgroup_size(1)
     fn foo() {
       let hsv = vec3f(0.6667, 1.0, 1.0); // Blue
       let result = hsv2rgb(hsv);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // HSV(240°, 100%, 100%) -> RGB(0, 0, 1)
  expectCloseTo([0.0002, 0.0, 1.0], result);
});

test("rgb2hsv", async () => {
  const src = `
     import lygia::color::space::rgb2hsv::rgb2hsv;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec3f(0.0, 0.0, 1.0); // Blue
       let result = rgb2hsv(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // RGB(0, 0, 1) -> HSV(240°/360 = 0.6667, 1, 1)
  expectCloseTo([0.6667, 1.0, 1.0], result);
});

test("hcy2rgb", async () => {
  const src = `
     import lygia::color::space::hcy2rgb::hcy2rgb;

     @compute @workgroup_size(1)
     fn foo() {
       let hcy = vec3f(0.0, 0.5, 0.5); // Red with chroma and luma
       let result = hcy2rgb(hcy);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // HCY(0°, 0.5, 0.5) -> RGB - matches GLSL output
  expectCloseTo([0.75, 0.3934, 0.3934], result);
});

test("rgb2hcy", async () => {
  const src = `
     import lygia::color::space::rgb2hcy::rgb2hcy;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec3f(1.0, 0.0, 0.0); // Pure red
       let result = rgb2hcy(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // RGB(1, 0, 0) -> HCY(0, 1, luma)
  expectCloseTo([0.0, 1.0, 0.2989], result);
});

test("hsv2ryb", async () => {
  const src = `
     import lygia::color::space::hsv2ryb::hsv2ryb;

     @compute @workgroup_size(1)
     fn foo() {
       let hsv = vec3f(0.0, 1.0, 1.0); // Red HSV
       let result = hsv2ryb(hsv);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // HSV(0°, 1, 1) -> RYB - needs investigation
  expectCloseTo([1.0, 0.0, 0.0], result);
});

// Color Utility Tests
test("rgb2luma", async () => {
  const src = `
     import lygia::color::space::rgb2luma::rgb2luma;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec3f(1.0, 0.5, 0.0); // Orange
       let result = rgb2luma(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src);
  // Luma using Rec709 coefficients: 1.0*0.2126 + 0.5*0.7152 + 0.0*0.0722
  expectCloseTo([0.5702], result);
});

test("srgb2luma", async () => {
  const src = `
     import lygia::color::space::srgb2luma::srgb2luma;

     @compute @workgroup_size(1)
     fn foo() {
       let srgb = vec3f(1.0, 0.5, 0.0); // Orange
       let result = srgb2luma(srgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src);
  // Uses Rec601 luma: dot(srgb, vec3(0.299, 0.587, 0.114))
  // 1.0*0.299 + 0.5*0.587 + 0.0*0.114 = 0.299 + 0.2935 = 0.5925
  expectCloseTo([0.5925], result);
});

test("rgb2hcv", async () => {
  const src = `
     import lygia::color::space::rgb2hcv::rgb2hcv;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec3f(1.0, 0.0, 0.0); // Red
       let result = rgb2hcv(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // RGB(1, 0, 0) -> HCV(0, 1, 1) - Hue, Chroma, Value
  expectCloseTo([0.0, 1.0, 1.0], result);
});

test("rgb2hue", async () => {
  const src = `
     import lygia::color::space::rgb2hue::rgb2hue;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec3f(0.0, 1.0, 0.0); // Green
       let result = rgb2hue(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src);
  // Green is at 120° = 1/3 in normalized hue
  expectCloseTo([0.3333], result);
});

test("hue2rgb", async () => {
  const src = `
     import lygia::color::space::hue2rgb::hue2rgb;

     @compute @workgroup_size(1)
     fn foo() {
       let hue = 0.3333; // Green
       let result = hue2rgb(hue);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Hue 0.3333 (120°) -> RGB(0, 1, 0) Green
  expectCloseTo([0.0002, 1.0, 0.0], result);
});

test("hcy2rgb4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::hcy2rgb::hcy2rgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let hcy = vec4f(0.0, 0.5, 0.5, 0.6); // Red with alpha
       let result = hcy2rgb4(hcy);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.75, 0.3934, 0.3934, 0.6], result);
});

test("hsl2rgb4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::hsl2rgb::hsl2rgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let hsl = vec4f(0.5, 0.8, 0.5, 0.9); // Cyan-ish with alpha
       let result = hsl2rgb4(hsl);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.1, 0.9, 0.9, 0.9], result);
});

test("hsv2rgb4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::hsv2rgb::hsv2rgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let hsv = vec4f(0.6667, 1.0, 1.0, 0.5); // Blue with alpha
       let result = hsv2rgb4(hsv);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.0002, 0.0, 1.0, 0.5], result);
});

test("rgb2hcy4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::rgb2hcy::rgb2hcy4;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec4f(1.0, 0.0, 0.0, 0.1); // Pure red with alpha
       let result = rgb2hcy4(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.0, 1.0, 0.2989, 0.1], result);
});

test("rgb2hsl4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::rgb2hsl::rgb2hsl4;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec4f(0.1, 0.9, 0.9, 0.5); // Cyan-ish with alpha
       let result = rgb2hsl4(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.5, 0.8, 0.5, 0.5], result);
});

test("rgb2hsv4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::rgb2hsv::rgb2hsv4;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec4f(0.0, 0.0, 1.0, 0.9); // Blue with alpha
       let result = rgb2hsv4(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.6667, 1.0, 1.0, 0.9], result);
});

test("hsv2ryb - default mode", async () => {
  const src = `
     import lygia::color::space::hsv2ryb::hsv2ryb;

     @compute @workgroup_size(1)
     fn foo() {
       let hsv = vec3f(0.0, 1.0, 1.0); // Red HSV
       let result = hsv2ryb(hsv);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // HSV(0°, 1, 1) Red -> RYB
  expectCloseTo([1.0, 0.0, 0.0], result);
});

test("hsv2ryb - FAST mode", async () => {
  const src = `
     import lygia::color::space::hsv2ryb::hsv2ryb;

     @compute @workgroup_size(1)
     fn foo() {
       let hsv = vec3f(0.3333, 0.8, 0.9); // Greenish HSV
       let result = hsv2ryb(hsv);
       test::results[0] = result;
     }
   `;
  const defines = { HSV2RYB_FAST: true };
  const result = await lygiaTestCompute(src, {
    elem: "vec3f",
    conditions: defines,
  });
  // HSV -> RYB using fast CMY bias version
  // Actual result: (0.9, 0.9, 0.18) - yellowish-green
  expectCloseTo([0.9, 0.9, 0.18], result);
});

test("rgb2hcv4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::rgb2hcv::rgb2hcv4;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec4f(1.0, 0.5, 0.0, 0.6); // Orange with alpha
       let result = rgb2hcv4(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // RGB(1, 0.5, 0) -> HCV (Hue, Chroma, Value) with alpha
  expectCloseTo([0.0833, 1.0, 1.0, 0.6], result);
});

test("rgb2hue4 - vec4 overload with alpha preservation (FIXED)", async () => {
  const src = `
     import lygia::color::space::rgb2hue::rgb2hue4;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec4f(0.0, 1.0, 0.0, 0.75); // Green with alpha
       let result = rgb2hue4(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Hue is 0.3333 (120°/360°), replicated as (0.3333, 0.3333, 0.3333, 0.75)
  expectCloseTo([0.3333, 0.3333, 0.3333, 0.75], result);
});

test("rgb2luma4 - vec4 overload with alpha preservation (FIXED)", async () => {
  const src = `
     import lygia::color::space::rgb2luma::rgb2luma4;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec4f(1.0, 0.5, 0.0, 0.85); // Orange with alpha
       let result = rgb2luma4(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Luma using Rec709: 1.0*0.2126 + 0.5*0.7152 + 0.0*0.0722 = 0.5702
  // Replicated as (0.5702, 0.5702, 0.5702, 0.85)
  expectCloseTo([0.5702, 0.5702, 0.5702, 0.85], result);
});

test("srgb2luma4 - vec4 overload with alpha preservation (FIXED)", async () => {
  const src = `
     import lygia::color::space::srgb2luma::srgb2luma4;

     @compute @workgroup_size(1)
     fn foo() {
       let srgb = vec4f(1.0, 0.5, 0.0, 0.95); // Orange sRGB with alpha
       let result = srgb2luma4(srgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Rec601 luma: 1.0*0.299 + 0.5*0.587 + 0.0*0.114 = 0.5925
  // Replicated as (0.5925, 0.5925, 0.5925, 0.95)
  expectCloseTo([0.5925, 0.5925, 0.5925, 0.95], result);
});

test("rgb2srgb_mono - f32 function", async () => {
  const src = `
     import lygia::color::space::rgb2srgb::rgb2srgb_mono;

     @compute @workgroup_size(1)
     fn foo() {
       // Test both branches of the function
       let low = rgb2srgb_mono(0.002); // < 0.0031308 branch
       let high = rgb2srgb_mono(0.5);  // >= 0.0031308 branch
       test::results[0] = vec4f(low, high, 0.0, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // low: 12.92 * 0.002 = 0.02584
  // high: 1.055 * pow(0.5, 0.41667) - 0.055 ≈ 0.735
  expectCloseTo([0.02584, 0.73536, 0.0, 0.0], result);
});

test("srgb2rgb_mono - f32 function", async () => {
  const src = `
     import lygia::color::space::srgb2rgb::srgb2rgb_mono;

     @compute @workgroup_size(1)
     fn foo() {
       // Test both branches
       let low = srgb2rgb_mono(0.03);  // < 0.04045 branch
       let high = srgb2rgb_mono(0.735); // >= 0.04045 branch
       test::results[0] = vec4f(low, high, 0.0, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // low: 0.03 * 0.0773993808 ≈ 0.00232
  // high: pow((0.735 + 0.055) * 0.9478673, 2.4) ≈ 0.5
  expectCloseTo([0.00232, 0.49946, 0.0, 0.0], result);
});

test("rgb2srgb", async () => {
  const src = `
     import lygia::color::space::rgb2srgb::rgb2srgb;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec3f(0.5, 0.3, 0.1); // Linear RGB
       let result = rgb2srgb(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Linear RGB -> sRGB (gamma correction)
  expectCloseTo([0.7354, 0.5838, 0.3492], result);
});

test("srgb2rgb", async () => {
  const src = `
     import lygia::color::space::srgb2rgb::srgb2rgb;

     @compute @workgroup_size(1)
     fn foo() {
       let srgb = vec3f(0.735, 0.584, 0.349);
       let result = srgb2rgb(srgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // sRGB -> Linear RGB
  expectCloseTo([0.4995, 0.3002, 0.0999], result);
});

test("rgb2srgb4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::rgb2srgb::rgb2srgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec4f(0.5, 0.3, 0.1, 0.4); // Linear RGB with alpha
       let result = rgb2srgb4(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.7354, 0.5838, 0.3492, 0.4], result);
});

test("srgb2rgb4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::srgb2rgb::srgb2rgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let srgb = vec4f(0.735, 0.584, 0.349, 0.3);
       let result = srgb2rgb4(srgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.49946, 0.30019, 0.09989, 0.3], result);
});
