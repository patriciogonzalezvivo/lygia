import { test } from "vitest";
import { imageMatcher } from "vitest-image-snapshot";
import { createSampler, getGPUDevice, lemurTexture } from "wgsl-test";
import { lygiaExampleImage } from "./testUtil.ts";

imageMatcher();

test("edgePrewitt - visual", async () => {
  const device = await getGPUDevice();
  const inputTex = await lemurTexture(device, 256);
  const sampler = createSampler(device);

  await lygiaExampleImage(device, "filter-edge-prewitt", {
    textures: [inputTex],
    samplers: [sampler],
    shader: `
      import lygia::filter::edge::prewitt::edgePrewitt;

      @group(0) @binding(0) var<uniform> uniforms: test::Uniforms;
      @group(0) @binding(1) var input_tex: texture_2d<f32>;
      @group(0) @binding(2) var input_samp: sampler;

      @fragment
      fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
        let uv = pos.xy / uniforms.resolution;
        let pixel_size = 1.0 / uniforms.resolution;
        let edge = edgePrewitt(input_tex, input_samp, uv, pixel_size);
        return vec4f(edge, 1.0);
      }
    `,
  });
});
