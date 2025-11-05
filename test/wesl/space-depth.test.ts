import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("linearizeDepth", async () => {
  const src = `
    import lygia::space::linearizeDepth::linearizeDepth;
    @compute @workgroup_size(1)
    fn foo() {
      let result = linearizeDepth(0.5, 0.1, 100.0);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src);
  // Linearize depth with near=0.1, far=100.0, depth=0.5
  // d = 2*0.5 - 1 = 0
  // result = (2 * 0.1 * 100) / (100 + 0.1 - 0 * (100 - 0.1)) = 20 / 100.1 = 0.1998...
  expectCloseTo([0.1998], result);
});

test("depth2viewZ perspective", async () => {
  const src = `
    import lygia::space::depth2viewZ::depth2viewZ;
    @compute @workgroup_size(1)
    fn foo() {
      let result = depth2viewZ(0.5, 1.0, 100.0);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src);
  // Perspective: (near * far) / ((far - near) * depth - far)
  // = (1 * 100) / ((100 - 1) * 0.5 - 100) = 100 / (49.5 - 100) = 100 / -50.5
  expectCloseTo([-1.9802], result);
});

test("viewZ2depth perspective", async () => {
  const src = `
    import lygia::space::viewZ2depth::viewZ2depth;
    @compute @workgroup_size(1)
    fn foo() {
      let result = viewZ2depth(-1.9802, 1.0, 100.0);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src);
  // Should reverse the depth2viewZ operation
  expectCloseTo([0.5], result);
});
