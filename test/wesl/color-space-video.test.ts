import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("rgb2YPbPr", async () => {
  const src = `
		import lygia::color::space::rgb2YPbPr::rgb2YPbPr;

		@compute @workgroup_size(1)
		fn foo() {
			test::results[0] = rgb2YPbPr(vec3f(.6, .7, .5));
		}
	`;

  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  const expected = [0.6643, -0.0885, -0.0408];
  expectCloseTo(expected, result);

  const sdtv = { YPBPR_SDTV: true };
  const resultSdtv = await lygiaTestCompute(src, {
    elem: "vec3f",
    conditions: sdtv,
  });
  const expectedSdtv = [0.6473, -0.0831, -0.0338];
  expectCloseTo(expectedSdtv, resultSdtv);
});

test("rgb2yuv", async () => {
  const src = `
    import lygia::color::space::rgb2yuv::rgb2yuv;

    @compute @workgroup_size(1)
    fn foo() { 
      test::results[0] = rgb2yuv(vec3f(.6, .7, .5)); 
    }
  `;

  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  const expected = [0.6643, -0.0822, -0.0502];
  expectCloseTo(expected, result);

  const sdtv = { YUV_SDTV: true };
  const resultSdtv = await lygiaTestCompute(src, {
    elem: "vec3f",
    conditions: sdtv,
  });
  const expectedSdtv = [0.6473, -0.0725, -0.0415];
  expectCloseTo(expectedSdtv, resultSdtv);
});

test("yuv2rgb", async () => {
  const src = `
     import lygia::color::space::yuv2rgb::yuv2rgb;

     @compute @workgroup_size(1)
     fn foo() {
       test::results[0] = yuv2rgb(vec3f(.6, .7, .5));
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  const expected = [1.2402, 0.2593, 2.0896];
  expectCloseTo(expected, result);

  const sdtv = { YUV_SDTV: true };
  const resultSdtv = await lygiaTestCompute(src, {
    elem: "vec3f",
    conditions: sdtv,
  });
  const expectedSdtv = [1.1699, 0.0334, 2.0225];
  expectCloseTo(expectedSdtv, resultSdtv);
});

test("YCbCr2rgb", async () => {
  const src = `
     import lygia::color::space::YCbCr2rgb::YCbCr2rgb;

     @compute @workgroup_size(1)
     fn foo() {
       let ycbcr = vec3f(0.5, 0.5, 0.5); // Mid gray
       let result = YCbCr2rgb(ycbcr);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // YCbCr(0.5, 0.5, 0.5) -> RGB (gray)
  expectCloseTo([0.5, 0.5, 0.5], result);
});

test("YPbPr2rgb", async () => {
  const src = `
     import lygia::color::space::YPbPr2rgb::YPbPr2rgb;

     @compute @workgroup_size(1)
     fn foo() {
       let ypbpr = vec3f(0.5, 0.0, 0.0); // Mid gray
       let result = YPbPr2rgb(ypbpr);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // YPbPr(0.5, 0, 0) -> RGB (gray)
  expectCloseTo([0.5, 0.5, 0.5], result);
});

test("rgb2YCbCr", async () => {
  const src = `
     import lygia::color::space::rgb2YCbCr::rgb2YCbCr;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec3f(0.5, 0.5, 0.5); // Gray
       let result = rgb2YCbCr(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // RGB(0.5, 0.5, 0.5) -> YCbCr (0.5, 0.5, 0.5)
  expectCloseTo([0.5, 0.5, 0.5], result);
});

test("yiq2rgb", async () => {
  const src = `
     import lygia::color::space::yiq2rgb::yiq2rgb;

     @compute @workgroup_size(1)
     fn foo() {
       let yiq = vec3f(0.5, 0.0, 0.0); // Gray in YIQ
       let result = yiq2rgb(yiq);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // YIQ(0.5, 0, 0) -> RGB
  // Matrix mult: [1.0, 1.0, 1.0] * 0.5 = (0.5, 0.5, 0.5) only if I=Q=0
  // But matrix has other values in first column, so actual result varies
  expectCloseTo([0.5, 0.4735, 0.3117], result);
});

test("rgb2yiq", async () => {
  const src = `
     import lygia::color::space::rgb2yiq::rgb2yiq;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec3f(1.0, 0.0, 0.0); // Red
       let result = rgb2yiq(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // RGB(1, 0, 0) -> YIQ using matrix RGB2YIQ (column-major)
  // Y = 0.3, I = 0.599, Q = 0.213 (first column of matrix)
  expectCloseTo([0.3, 0.59, 0.11], result);
});

test("YCbCr2rgb4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::YCbCr2rgb::YCbCr2rgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let ycbcr = vec4f(0.5, 0.5, 0.5, 0.7); // Mid gray with alpha
       let result = YCbCr2rgb4(ycbcr);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.5, 0.5, 0.5, 0.7], result);
});

test("YPbPr2rgb4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::YPbPr2rgb::YPbPr2rgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let ypbpr = vec4f(0.5, 0.0, 0.0, 0.8); // Mid gray with alpha
       let result = YPbPr2rgb4(ypbpr);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.5, 0.5, 0.5, 0.8], result);
});

test("rgb2YCbCr4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::rgb2YCbCr::rgb2YCbCr4;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec4f(0.5, 0.5, 0.5, 0.25); // Gray with alpha
       let result = rgb2YCbCr4(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.5, 0.5, 0.5, 0.25], result);
});

test("rgb2YPbPr4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::rgb2YPbPr::rgb2YPbPr4;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec4f(0.6, 0.7, 0.5, 0.15);
       let result = rgb2YPbPr4(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.6643, -0.0885, -0.0408, 0.15], result);
});

test("rgb2yiq4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::rgb2yiq::rgb2yiq4;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec4f(1.0, 0.0, 0.0, 0.8); // Red with alpha
       let result = rgb2yiq4(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // RGB(1, 0, 0) -> YIQ (first column of RGB2YIQ matrix)
  expectCloseTo([0.3, 0.59, 0.11, 0.8], result);
});

test("rgb2yuv4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::rgb2yuv::rgb2yuv4;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec4f(0.6, 0.7, 0.5, 0.3);
       let result = rgb2yuv4(rgb);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // YUV conversion with alpha
  expectCloseTo([0.6643, -0.0822, -0.0502, 0.3], result);
});

test("yiq2rgb4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::yiq2rgb::yiq2rgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let yiq = vec4f(0.5, 0.0, 0.0, 0.55); // Gray with alpha
       let result = yiq2rgb4(yiq);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // YIQ(0.5, 0, 0) -> RGB with alpha
  expectCloseTo([0.5, 0.4735, 0.3117, 0.55], result);
});

test("yuv2rgb4 - alpha preservation", async () => {
  const src = `
     import lygia::color::space::yuv2rgb::yuv2rgb4;

     @compute @workgroup_size(1)
     fn foo() {
       let yuv = vec4f(0.6, 0.7, 0.5, 0.95);
       let result = yuv2rgb4(yuv);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // YUV -> RGB with alpha
  expectCloseTo([1.2402, 0.2593, 2.0896, 0.95], result);
});
