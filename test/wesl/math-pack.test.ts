import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("pack/unpack roundtrip", async () => {
  const src = `
    import lygia::math::pack::pack;
    import lygia::math::unpack::unpack4;
    @compute @workgroup_size(1)
    fn foo() {
      let original = 0.12346;
      let packed = pack(original);
      let unpacked = unpack4(packed);
      test::results[0] = vec4f(original, unpacked, 0.0, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Pack/unpack roundtrip should preserve value within default tolerance despite 8-bit RGBA encoding
  expectCloseTo([0.12346, 0.12346], result.slice(0, 2));
});

test("pack/unpack roundtrip - multiple values", async () => {
  const src = `
    import lygia::math::pack::pack;
    import lygia::math::unpack::unpack4;
    @compute @workgroup_size(1)
    fn foo() {
      // Test that pack/unpack are inverse operations with multiple values
      let v1 = 0.0;
      let v2 = 0.25;
      let v3 = 0.5;
      let v4 = 0.75;

      let r1 = unpack4(pack(v1));
      let r2 = unpack4(pack(v2));
      let r3 = unpack4(pack(v3));
      let r4 = unpack4(pack(v4));

      test::results[0] = vec4f(r1, r2, r3, r4);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // Verify roundtrip accuracy - values should roundtrip within default tolerance
  expectCloseTo([0.0], [result[0]]);
  expectCloseTo([0.25], [result[1]]);
  expectCloseTo([0.5], [result[2]]);
  expectCloseTo([0.75], [result[3]]);
});

test("unpack256 - default base 256", async () => {
  const src = `
    import lygia::math::unpack::unpack256;
    @compute @workgroup_size(1)
    fn foo() {
      // Test unpacking with base 256
      // unpack256 uses dot(v, vec3(256, 256^2, 256^3)) / 16581375
      // Note: divisor is 16581375 = 256^3 * (255/256), not just 256^3
      let v1 = vec3f(1.0, 0.0, 0.0);  // 256 / 16581375
      let v2 = vec3f(0.5, 0.5, 0.5);  // (128 + 32768 + 8388608) / 16581375
      let v3 = vec3f(1.0, 1.0, 1.0);  // (256 + 65536 + 16777216) / 16581375

      let r1 = unpack256(v1);
      let r2 = unpack256(v2);
      let r3 = unpack256(v3);

      test::results[0] = vec4f(r1, r2, r3, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // v1 = (1,0,0)  256 / 16581375 ≈ 0.000015
  expectCloseTo([0.000015], [result[0]]); // Very small value
  // v2 = (0.5,0.5,0.5)  (128 + 32768 + 8388608) / 16581375 ≈ 0.50787
  expectCloseTo([0.50787], [result[1]]);
  // v3 = (1,1,1)  (256 + 65536 + 16777216) / 16581375 ≈ 1.01578
  expectCloseTo([1.01578], [result[2]]);
});

test("unpack - alias for unpack256", async () => {
  const src = `
    import lygia::math::unpack::unpack;
    import lygia::math::unpack::unpack256;
    @compute @workgroup_size(1)
    fn foo() {
      let v = vec3f(0.5, 0.5, 0.5);
      let r1 = unpack(v);
      let r2 = unpack256(v);
      test::results[0] = vec4f(r1, r2, 0.0, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // unpack should be identical to unpack256
  expectCloseTo([result[0]], [result[1]]);
});

test("unpack8 - base 8", async () => {
  const src = `
    import lygia::math::unpack::unpack8;
    @compute @workgroup_size(1)
    fn foo() {
      // unpack8 uses dot(v, vec3(8, 64, 512)) / 512
      let v1 = vec3f(1.0, 0.0, 0.0);   // 8 / 512 = 0.015625
      let v2 = vec3f(0.0, 1.0, 0.0);   // 64 / 512 = 0.125
      let v3 = vec3f(0.0, 0.0, 1.0);   // 512 / 512 = 1.0
      let v4 = vec3f(0.5, 0.5, 0.5);   // (4 + 32 + 256) / 512 = 0.5703125

      test::results[0] = vec4f(unpack8(v1), unpack8(v2), unpack8(v3), unpack8(v4));
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.015625, 0.125, 1.0, 0.5703125], result);
});

test("unpack16 - base 16", async () => {
  const src = `
    import lygia::math::unpack::unpack16;
    @compute @workgroup_size(1)
    fn foo() {
      // unpack16 uses dot(v, vec3(16, 256, 4096)) / 4096
      let v1 = vec3f(1.0, 0.0, 0.0);   // 16 / 4096 = 0.00390625
      let v2 = vec3f(0.0, 1.0, 0.0);   // 256 / 4096 = 0.0625
      let v3 = vec3f(0.0, 0.0, 1.0);   // 4096 / 4096 = 1.0
      let v4 = vec3f(0.5, 0.5, 0.5);   // (8 + 128 + 2048) / 4096 = 0.533203125

      test::results[0] = vec4f(unpack16(v1), unpack16(v2), unpack16(v3), unpack16(v4));
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.00390625, 0.0625, 1.0, 0.533203125], result);
});

test("unpack32 - base 32", async () => {
  const src = `
    import lygia::math::unpack::unpack32;
    @compute @workgroup_size(1)
    fn foo() {
      // unpack32 uses dot(v, vec3(32, 1024, 32768)) / 32768
      let v1 = vec3f(1.0, 0.0, 0.0);   // 32 / 32768 = 0.00098
      let v2 = vec3f(0.0, 1.0, 0.0);   // 1024 / 32768 = 0.03125
      let v3 = vec3f(0.0, 0.0, 1.0);   // 32768 / 32768 = 1.0
      let v4 = vec3f(0.5, 0.5, 0.5);   // (16 + 512 + 16384) / 32768 = 0.5161

      test::results[0] = vec4f(unpack32(v1), unpack32(v2), unpack32(v3), unpack32(v4));
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.00098, 0.03125, 1.0, 0.5161], result);
});

test("unpack64 - base 64", async () => {
  const src = `
    import lygia::math::unpack::unpack64;
    @compute @workgroup_size(1)
    fn foo() {
      // unpack64 uses dot(v, vec3(64, 4096, 262144)) / 262144
      let v1 = vec3f(1.0, 0.0, 0.0);   // 64 / 262144 H 0.000244
      let v2 = vec3f(0.0, 1.0, 0.0);   // 4096 / 262144 H 0.015625
      let v3 = vec3f(0.0, 0.0, 1.0);   // 262144 / 262144 = 1.0
      let v4 = vec3f(0.5, 0.5, 0.5);   // (32 + 2048 + 131072) / 262144 H 0.507935

      test::results[0] = vec4f(unpack64(v1), unpack64(v2), unpack64(v3), unpack64(v4));
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.000244, 0.015625, 1.0, 0.507935], result);
});

test("unpack128 - base 128", async () => {
  const src = `
    import lygia::math::unpack::unpack128;
    @compute @workgroup_size(1)
    fn foo() {
      // unpack128 uses dot(v, vec3(128, 16384, 2097152)) / 2097152
      let v1 = vec3f(1.0, 0.0, 0.0);   // 128 / 2097152 H 0.000061
      let v2 = vec3f(0.0, 1.0, 0.0);   // 16384 / 2097152 H 0.0078125
      let v3 = vec3f(0.0, 0.0, 1.0);   // 2097152 / 2097152 = 1.0
      let v4 = vec3f(0.5, 0.5, 0.5);   // (64 + 8192 + 1048576) / 2097152 H 0.503967

      test::results[0] = vec4f(unpack128(v1), unpack128(v2), unpack128(v3), unpack128(v4));
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.000061, 0.0078125, 1.0, 0.503967], result);
});

test("unpackBase - custom base", async () => {
  const src = `
    import lygia::math::unpack::unpackBase;
    @compute @workgroup_size(1)
    fn foo() {
      // Test with base 10: dot(v, vec3(10, 100, 1000)) / 1000
      let base = 10.0;
      let v1 = vec3f(1.0, 0.0, 0.0);   // 10 / 1000 = 0.01
      let v2 = vec3f(0.0, 1.0, 0.0);   // 100 / 1000 = 0.1
      let v3 = vec3f(0.0, 0.0, 1.0);   // 1000 / 1000 = 1.0
      let v4 = vec3f(0.5, 0.5, 0.5);   // (5 + 50 + 500) / 1000 = 0.555

      test::results[0] = vec4f(
        unpackBase(v1, base),
        unpackBase(v2, base),
        unpackBase(v3, base),
        unpackBase(v4, base)
      );
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.01, 0.1, 1.0, 0.555], result);
});

test("unpack4 - vec4 unpacking (ThreeJS style)", async () => {
  const src = `
    import lygia::math::unpack::unpack4;
    @compute @workgroup_size(1)
    fn foo() {
      // unpack4 uses ThreeJS packing: dot(v, UnpackFactors)
      // UnpackFactors = (255/256) / vec4(256^3, 256^2, 256, 1)
      // = vec4(5.960464e-8, 1.5258789e-5, 0.00390625, 0.99609375)
      // Sum H 1.0
      let v1 = vec4f(1.0, 0.0, 0.0, 0.0);  // r only: 5.960464e-8
      let v2 = vec4f(0.0, 0.0, 0.0, 1.0);  // a only: 0.99609375
      let v3 = vec4f(0.5, 0.5, 0.5, 0.5);  // uniform: 0.5
      let v4 = vec4f(1.0, 1.0, 1.0, 1.0);  // all: 1.0

      test::results[0] = vec4f(unpack4(v1), unpack4(v2), unpack4(v3), unpack4(v4));
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // Test with specific expected values based on UnpackFactors formula
  // UnpackFactors = (255/256) / vec4f(256^3, 256^2, 256, 1)
  expectCloseTo([5.960464e-8], [result[0]]); // r component: very small value, use default precision
  expectCloseTo([0.99609375], [result[1]]); // a component: 255/256
  expectCloseTo([0.5], [result[2]]); // uniform 0.5 across all components
  expectCloseTo([1.0], [result[3]]); // sum of all UnpackFactors ≈ 1.0
});
