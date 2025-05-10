import { beforeAll, expect, test } from "vitest";
import { testComputeShader } from "wesl-tooling";

let gpu: GPU;

beforeAll(async () => {
  const webgpu = await import("webgpu");
  Object.assign(globalThis, webgpu.globals);

  gpu = webgpu.create([]);
});

test("rgb2heat", async () => {
  const src = `
    import lygia::color::space::rgb2heat::rgb2heat;

    @compute @workgroup_size(1)
    fn foo() { 
      let x = rgb2heat(vec3f(.8, .7, .5)); 

      test::results[0] = x;
    }
  `;

  const result = await testComputeShader(import.meta.url, gpu, src, "f32");
  expect(result[0]).approximately(0.854, 0.001);
});
