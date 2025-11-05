import { afterAll, beforeAll, test } from "vitest";
import { imageMatcher } from "vitest-image-snapshot";
import { destroySharedDevice, getGPUDevice } from "wesl-test";
import { expectDither, lygiaTestCompute, expectCloseTo } from "./testUtil.ts";

imageMatcher();

beforeAll(async () => {
  await getGPUDevice();
});

afterAll(() => {
  destroySharedDevice();
});

test("ditherBayer3 - gradient with Bayer pattern", async () => {
  await expectDither(
    `
    import lygia::color::dither::bayer::ditherBayer3Precision;
    import lygia::testing::sampleQuantized::{sampleQuantized3, sampleOriginal3};

    @group(0) @binding(0) var<uniform> u: test::Uniforms;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let levels = 8;
      let original = sampleOriginal3(pos);

      // Left: posterized without dithering (hard banding)
      let posterized = sampleQuantized3(pos, levels);

      // Right: dithered to same levels (smooth with Bayer pattern)
      let dithered = ditherBayer3Precision(original, pos.xy, levels);

      let result = select(posterized, dithered, pos.x > u.resolution.x / 2);
      return vec4f(result, 1.0);
    }
  `,
    "dither-bayer3-gradient",
  );
});

test("ditherVlachos3 - gradient with Vlachos noise", async () => {
  await expectDither(
    `
    import lygia::color::dither::vlachos::ditherVlachos3Precision;
    import lygia::testing::sampleQuantized::{sampleQuantized3, sampleOriginal3};

    @group(0) @binding(0) var<uniform> u: test::Uniforms;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let levels = 8;
      let original = sampleOriginal3(pos);

      // Left: posterized without dithering (hard banding)
      let posterized = sampleQuantized3(pos, levels);

      // Right: dithered to same levels (smooth with Vlachos noise)
      let dithered = ditherVlachos3Precision(original, pos.xy, levels);

      let result = select(posterized, dithered, pos.x > u.resolution.x / 2);
      return vec4f(result, 1.0);
    }
  `,
    "dither-vlachos3-gradient",
  );
});

test("ditherBlueNoise3 - gradient with blue noise pattern", async () => {
  await expectDither(
    `
    import lygia::color::dither::blueNoise::ditherBlueNoise3Precision;
    import lygia::testing::sampleQuantized::{sampleQuantized3, sampleOriginal3};

    @group(0) @binding(0) var<uniform> u: test::Uniforms;

    @fragment
    fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
      let levels = 8;
      let original = sampleOriginal3(pos);

      // Left: posterized without dithering (hard banding)
      let posterized = sampleQuantized3(pos, levels);

      // Right: dithered to same levels (smooth with blue noise)
      let dithered = ditherBlueNoise3Precision(original, pos.xy, levels);

      let result = select(posterized, dithered, pos.x > u.resolution.x / 2);
      return vec4f(result, 1.0);
    }
  `,
    "dither-bluenoise3-gradient",
  );
});

// =============================================================================
// Unit Tests
// =============================================================================

test("ditherBayer - all wrapper functions", async () => {
  const src = `
     import lygia::color::dither::bayer::{
       ditherBayer,
       ditherBayerPrecision,
       ditherBayer3,
       ditherBayer3Precision,
       ditherBayer4
     };

     @compute @workgroup_size(1)
     fn foo() {
       // Test all wrapper functions with non-trivial values
       let value = 0.53;
       let color3 = vec3f(0.53, 0.62, 0.47);
       let color4 = vec4f(0.53, 0.62, 0.47, 0.85);
       let xy = vec2f(2.0, 3.0);

       // Test scalar wrapper with precision
       let ditheredScalar = ditherBayerPrecision(value, xy, 16);

       // Test vec3 wrapper with default precision (256)
       let dithered3Default = ditherBayer3(color3, xy);

       // Test vec3 wrapper with custom precision
       let dithered3Custom = ditherBayer3Precision(color3, xy, 16);

       // Test vec4 wrapper (should preserve alpha)
       let dithered4 = ditherBayer4(color4, xy);

       // Pack results for validation
       test::results[0] = vec4f(ditheredScalar, 0.0, 0.0, 0.0);
       test::results[1] = vec4f(dithered3Default, 0.0);
       test::results[2] = vec4f(dithered3Custom, 0.0);
       test::results[3] = dithered4;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f", size: 4 });

  // Scalar with precision=16: 0.53 quantized to 16 levels
  expectCloseTo([0.5, 0.0, 0.0, 0.0], result.slice(0, 4));

  // Vec3 with default precision=256 (rounds down for these values)
  expectCloseTo([0.5273, 0.6172, 0.4688, 0.0], result.slice(4, 8));

  // Vec3 with precision=16
  expectCloseTo([0.5, 0.625, 0.4375, 0.0], result.slice(8, 12));

  // Vec4 with precision=256 (should preserve alpha=0.85)
  expectCloseTo([0.5273, 0.6172, 0.4688, 0.85], result.slice(12, 16));
});

test("ditherVlachos - all wrapper functions", async () => {
  const src = `
     import lygia::color::dither::vlachos::{
       ditherVlachos3,
       ditherVlachos3Precision,
       ditherVlachos4
     };

     @compute @workgroup_size(1)
     fn foo() {
       // Test all wrapper functions with non-trivial values
       let color3 = vec3f(0.53, 0.62, 0.47);
       let color4 = vec4f(0.53, 0.62, 0.47, 0.85);
       let xy = vec2f(2.0, 3.0);

       // Test vec3 wrapper with default precision (256)
       let dithered3Default = ditherVlachos3(color3, xy);

       // Test vec3 wrapper with custom precision (16)
       let dithered3Custom = ditherVlachos3Precision(color3, xy, 16);

       // Test vec4 wrapper (should preserve alpha)
       let dithered4 = ditherVlachos4(color4, xy);

       // Pack results for validation
       test::results[0] = vec4f(dithered3Default, 0.0);
       test::results[1] = vec4f(dithered3Custom, 0.0);
       test::results[2] = dithered4;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f", size: 3 });

  // Vec3 with default precision=256 (adds noise then quantizes)
  // Values should be close to input (within ±1/256 due to noise)
  expectCloseTo([0.5273, 0.6211, 0.4688, 0.0], result.slice(0, 4), 0.004);

  // Vec3 with precision=16 (larger quantization steps)
  expectCloseTo([0.5, 0.625, 0.4375, 0.0], result.slice(4, 8), 0.07);

  // Vec4 with precision=256 (should preserve alpha=0.85)
  expectCloseTo([0.5273, 0.6211, 0.4688], result.slice(8, 11), 0.004);
  expectCloseTo([0.85], [result[11]]);
});

test("ditherBlueNoise - all wrapper functions", async () => {
  const src = `
     import lygia::color::dither::blueNoise::{
       ditherBlueNoise1,
       ditherBlueNoise3,
       ditherBlueNoise3Precision,
       ditherBlueNoise4
     };

     @compute @workgroup_size(1)
     fn foo() {
       // Test all wrapper functions with non-trivial values
       let value = 0.53;
       let color3 = vec3f(0.53, 0.62, 0.47);
       let color4 = vec4f(0.53, 0.62, 0.47, 0.85);
       let xy = vec2f(2.0, 3.0);

       // Test scalar wrapper with default precision (256)
       let dithered1 = ditherBlueNoise1(value, xy);

       // Test vec3 wrapper with default precision (256)
       let dithered3Default = ditherBlueNoise3(color3, xy);

       // Test vec3 wrapper with custom precision (16)
       let dithered3Custom = ditherBlueNoise3Precision(color3, xy, 16);

       // Test vec4 wrapper (should preserve alpha)
       let dithered4 = ditherBlueNoise4(color4, xy);

       // Pack results for validation
       test::results[0] = vec4f(dithered1, 0.0, 0.0, 0.0);
       test::results[1] = vec4f(dithered3Default, 0.0);
       test::results[2] = vec4f(dithered3Custom, 0.0);
       test::results[3] = dithered4;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f", size: 4 });

  // Scalar with precision=256: 0.53 quantized
  expectCloseTo([0.5312], result.slice(0, 1));

  // Vec3 with default precision=256
  expectCloseTo([0.5312, 0.6211, 0.4727, 0.0], result.slice(4, 8));

  // Vec3 with precision=16
  expectCloseTo([0.5625, 0.625, 0.5, 0.0], result.slice(8, 12));

  // Vec4 with precision=256 (should preserve alpha=0.85)
  expectCloseTo([0.5312, 0.6211, 0.4727, 0.85], result.slice(12, 16));
});
