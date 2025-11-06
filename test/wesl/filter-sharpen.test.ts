import { expect, test } from "vitest";
import { imageMatcher } from "vitest-image-snapshot";
import {
  createSampler,
  getGPUDevice,
  lemurTexture,
} from "wesl-test";
import {
  lygiaExampleImage,
  lygiaTestCompute,
} from "./testUtil.ts";

imageMatcher();

test("sharpenAdaptive - visual", async () => {
  const device = await getGPUDevice();
  const inputTex = await lemurTexture(device, 256);
  const sampler = createSampler(device);

  await lygiaExampleImage(device, "filter-sharpen-adaptive", {
    textures: [inputTex],
    samplers: [sampler],
    shader: `
      import lygia::filter::sharpen::adaptive::sharpenAdaptive;

      @group(0) @binding(0) var<uniform> uniforms: test::Uniforms;
      @group(0) @binding(1) var input_tex: texture_2d<f32>;
      @group(0) @binding(2) var input_samp: sampler;

      @fragment
      fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
        let uv = pos.xy / uniforms.resolution;
        let pixel_size = 1.0 / uniforms.resolution;
        return sharpenAdaptive(input_tex, input_samp, uv, pixel_size);
      }
    `,
  });
});

test("sharpenAdaptive4 - visual", async () => {
  const device = await getGPUDevice();
  const inputTex = await lemurTexture(device, 256);
  const sampler = createSampler(device);

  await lygiaExampleImage(device, "filter-sharpen-adaptive4", {
    textures: [inputTex],
    samplers: [sampler],
    shader: `
      import lygia::filter::sharpen::adaptive::sharpenAdaptive4;

      @group(0) @binding(0) var<uniform> uniforms: test::Uniforms;
      @group(0) @binding(1) var input_tex: texture_2d<f32>;
      @group(0) @binding(2) var input_samp: sampler;

      @fragment
      fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
        let uv = pos.xy / uniforms.resolution;
        let pixel_size = 1.0 / uniforms.resolution;
        return sharpenAdaptive4(input_tex, input_samp, uv, pixel_size, 1.0);
      }
    `,
  });
});

test("sharpenContrastAdaptive - visual", async () => {
  const device = await getGPUDevice();
  const inputTex = await lemurTexture(device, 256);
  const sampler = createSampler(device);

  await lygiaExampleImage(device, "filter-sharpen-contrast-adaptive", {
    textures: [inputTex],
    samplers: [sampler],
    shader: `
      import lygia::filter::sharpen::adaptive::sharpenContrastAdaptive;

      @group(0) @binding(0) var<uniform> uniforms: test::Uniforms;
      @group(0) @binding(1) var input_tex: texture_2d<f32>;
      @group(0) @binding(2) var input_samp: sampler;

      @fragment
      fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
        let uv = pos.xy / uniforms.resolution;
        let pixel_size = 1.0 / uniforms.resolution;
        let sharpened = sharpenContrastAdaptive(input_tex, input_samp, uv, pixel_size, 1.0);
        return vec4f(sharpened, 1.0);
      }
    `,
  });
});

test("sharpenFast - visual", async () => {
  const device = await getGPUDevice();
  const inputTex = await lemurTexture(device, 256);
  const sampler = createSampler(device);

  await lygiaExampleImage(device, "filter-sharpen-fast", {
    textures: [inputTex],
    samplers: [sampler],
    shader: `
      import lygia::filter::sharpen::fast::sharpenFast;

      @group(0) @binding(0) var<uniform> uniforms: test::Uniforms;
      @group(0) @binding(1) var input_tex: texture_2d<f32>;
      @group(0) @binding(2) var input_samp: sampler;

      @fragment
      fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
        let uv = pos.xy / uniforms.resolution;
        let pixel_size = 1.0 / uniforms.resolution;
        return sharpenFast(input_tex, input_samp, uv, pixel_size);
      }
    `,
  });
});

test("sharpenFast4 - visual", async () => {
  const device = await getGPUDevice();
  const inputTex = await lemurTexture(device, 256);
  const sampler = createSampler(device);

  await lygiaExampleImage(device, "filter-sharpen-fast4", {
    textures: [inputTex],
    samplers: [sampler],
    shader: `
      import lygia::filter::sharpen::fast::sharpenFast4;

      @group(0) @binding(0) var<uniform> uniforms: test::Uniforms;
      @group(0) @binding(1) var input_tex: texture_2d<f32>;
      @group(0) @binding(2) var input_samp: sampler;

      @fragment
      fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
        let uv = pos.xy / uniforms.resolution;
        let pixel_size = 1.0 / uniforms.resolution;
        return sharpenFast4(input_tex, input_samp, uv, pixel_size, 1.0);
      }
    `,
  });
});

test("sharpendAdaptiveControl4", async () => {
  const src = `
     import lygia::filter::sharpen::adaptive::sharpendAdaptiveControl4;

     @compute @workgroup_size(1)
     fn foo() {
       // sharpendAdaptiveControl4 computes perceptual luma: dot(rgba*rgba, vec4(0.21266, 0.71516, 0.07219, 0.0))
       // Test with gray color
       let gray = vec4f(0.5, 0.5, 0.5, 1.0);
       let result1 = sharpendAdaptiveControl4(gray);

       // Test with colored input
       let orange = vec4f(0.8, 0.5, 0.2, 1.0);
       let result2 = sharpendAdaptiveControl4(orange);

       // Test with black (should be 0)
       let black = vec4f(0.0, 0.0, 0.0, 1.0);
       let result3 = sharpendAdaptiveControl4(black);

       test::results[0] = vec3f(result1, result2, result3);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });

  // Gray: (0.5^2) * (0.21266 + 0.71516 + 0.07219) = 0.25 * 1.0 = 0.25
  expect(result[0]).toBeCloseTo(0.25, 2);

  // Orange: (0.8^2)*0.21266 + (0.5^2)*0.71516 + (0.2^2)*0.07219
  //       = 0.64*0.21266 + 0.25*0.71516 + 0.04*0.07219
  //       = 0.13610 + 0.17879 + 0.00289 = 0.31778
  expect(result[1]).toBeCloseTo(0.318, 2);

  // Black should be 0
  expect(result[2]).toBeCloseTo(0.0, 3);
});
