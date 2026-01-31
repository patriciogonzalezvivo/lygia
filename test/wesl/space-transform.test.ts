import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute, lygiaTestWesl } from "./testUtil.ts";

await lygiaTestWesl("test/wesl/shaders/space_transform_test");

// decimateNormal kept inline - uses > and < comparisons not available in wgsl_test
test("decimateNormal", async () => {
  const src = `
    import lygia::math::consts::INV_SQRT2;
    import lygia::space::decimateNormal::decimateNormal;
    @compute @workgroup_size(1)
    fn foo() {
      // Test quantization with precision 4.0 (0.25 steps)
      let prec = 4.0;

      // Test 1: Known case - 45° normal
      let n1 = normalize(vec3f(INV_SQRT2, INV_SQRT2, 0.0));
      let d1 = decimateNormal(n1, prec);

      // Test 2: Nearby normal (should quantize similarly)
      let n2 = normalize(vec3f(0.710, 0.690, 0.0));
      let d2 = decimateNormal(n2, prec);

      // Test 3: Distant normal (should quantize differently)
      let n3 = normalize(vec3f(1.0, 0.0, 0.0));
      let d3 = decimateNormal(n3, prec);

      // Compute metrics: unit length and differences
      let len1 = length(d1);
      let diff_nearby = abs(d1.x - d2.x) + abs(d1.y - d2.y) + abs(d1.z - d2.z);
      let diff_distant = abs(d1.x - d3.x) + abs(d1.y - d3.y) + abs(d1.z - d3.z);

      test::results[0] = d1.x;  // Expected: ~0.730
      test::results[1] = len1;  // Expected: 1.0 (unit length)
      test::results[2] = diff_nearby;  // Expected: <0.2 (nearby normals quantize similarly)
      test::results[3] = diff_distant;  // Expected: >0.3 (distant normals differ)
    }
  `;
  const result = await lygiaTestCompute(src);
  const r = result as number[];

  // Test 1: Known expected output for 45° normal with precision 4.0
  expectCloseTo([0.72986], [r[0]]);

  // Verify unit length
  expectCloseTo([1.0], [r[1]]);

  // d1 and d2 should be close (nearby normals quantize similarly)
  if (r[2] > 0.2) {
    throw new Error(
      `Expected nearby normals to quantize similarly, but difference was ${r[2]}`,
    );
  }

  // d3 should differ significantly (distant normal, different quantization bin)
  if (r[3] < 0.3) {
    throw new Error(
      `Expected distant normal to quantize differently, but difference was only ${r[3]}`,
    );
  }
});
