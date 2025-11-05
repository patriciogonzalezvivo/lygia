import { expect, test } from "vitest";
import {
  expectCloseTo,
  expectDistribution,
  lygiaTestCompute,
  testDistribution,
} from "./testUtil.ts";

test("random", async () => {
  const src = `
     import lygia::generative::random::random;

     @compute @workgroup_size(1)
     fn foo() {
       let r1 = random(1.0);
       let r2 = random(1.0); // Same input
       let r3 = random(2.0); // Different input

       test::results[0] = vec4f(r1, r2, r3, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different inputs produce different outputs
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Regression: exact output value
  expectCloseTo([0.763], [result[0]]);
});

test("random - distribution", async () => {
  const sampleCount = 1024;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::random::random;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        test::results[i] = random(f32(i));
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount,
  });
  expectDistribution(samples, [0.0, 1.0]);
});

test("random2", async () => {
  const src = `
     import lygia::generative::random::random2;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec2f(1.0, 2.0);
       let r1 = random2(p1);
       let r2 = random2(p1); // Same input
       let r3 = random2(vec2f(3.0, 4.0)); // Different input

       test::results[0] = vec4f(r1, r2, r3, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different inputs produce different outputs
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Regression: exact output value
  expectCloseTo([0.6153], [result[0]]);
});

test("random2 - distribution", async () => {
  const sampleCount = 512;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::random::random2;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        let x = f32(i % 32u);
        let y = f32(i / 32u);
        test::results[i] = random2(vec2f(x, y));
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount,
  });
  expectDistribution(samples, [0.0, 1.0]);
});

test("random3", async () => {
  const src = `
     import lygia::generative::random::random3;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec3f(1.0, 2.0, 3.0);
       let r1 = random3(p1);
       let r2 = random3(p1); // Same input
       let r3 = random3(vec3f(4.0, 5.0, 6.0)); // Different input

       test::results[0] = vec4f(r1, r2, r3, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different inputs produce different outputs
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Regression: exact output value
  expectCloseTo([0.372], [result[0]]);
});

test("random3 - distribution", async () => {
  const sampleCount = 1024;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::random::random3;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        let x = f32(i % 16u);
        let y = f32((i / 16u) % 16u);
        let z = f32(i / 256u);
        test::results[i] = random3(vec3f(x, y, z));
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount,
  });
  expectDistribution(samples, [0.0, 1.0]);
});

test("random4", async () => {
  const src = `
     import lygia::generative::random::random4;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec4f(1.0, 2.0, 3.0, 4.0);
       let r1 = random4(p1);
       let r2 = random4(p1); // Same input
       let r3 = random4(vec4f(5.0, 6.0, 7.0, 8.0)); // Different input

       test::results[0] = vec4f(r1, r2, r3, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different inputs produce different outputs
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Regression: exact output value
  expectCloseTo([0.5181], [result[0]]);
});

test("random21 - basic output", async () => {
  const src = `
     import lygia::generative::random::random21;

     @compute @workgroup_size(1)
     fn foo() {
       let r1 = random21(1.0);
       let r2 = random21(1.0); // Same input
       let r3 = random21(2.0); // Different input

       // Test determinism and range
       test::results[0] = vec4f(r1.x, r1.y, r2.x, r2.y);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same input produces same output
  expectCloseTo([result[0], result[1]], [result[2], result[3]]);
  // Regression: exact output value
  expectCloseTo([0.8786], [result[0]]);
});

test("random22 - basic output", async () => {
  const src = `
     import lygia::generative::random::random22;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec2f(1.0, 2.0);
       let r1 = random22(p1);
       let r2 = random22(p1); // Same input

       // Test determinism and range
       test::results[0] = vec4f(r1.x, r1.y, r2.x, r2.y);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same input produces same output
  expectCloseTo([result[0], result[1]], [result[2], result[3]]);
  // Regression: exact output value
  expectCloseTo([0.2333], [result[0]]);
});

test("random22 - distribution (x component)", async () => {
  const sampleCount = 512;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::random::random22;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        let x = f32(i % 32u);
        let y = f32(i / 32u);
        let sample = random22(vec2f(x, y));
        test::results[i] = sample.x;
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount,
  });
  expectDistribution(samples, [0.0, 1.0]);
});

test("random23 - basic output", async () => {
  const src = `
     import lygia::generative::random::random23;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec3f(1.0, 2.0, 3.0);
       let r1 = random23(p1);
       let r2 = random23(p1); // Same input

       // Test determinism and range
       test::results[0] = vec4f(r1.x, r1.y, r2.x, r2.y);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same input produces same output
  expectCloseTo([result[0], result[1]], [result[2], result[3]]);
  // Regression: exact output value
  expectCloseTo([0.6837], [result[0]]);
});

test("random31 - basic output", async () => {
  const src = `
     import lygia::generative::random::random31;

     @compute @workgroup_size(1)
     fn foo() {
       let r1 = random31(1.0);
       let r2 = random31(1.0); // Same input

       // Test determinism and range (can only fit 3 components, test first 3)
       test::results[0] = vec4f(r1.x, r1.y, r1.z, r2.x);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same input produces same output (test first component)
  expectCloseTo([result[0]], [result[3]]);
  // Regression: exact output value
  expectCloseTo([0.8786], [result[0]]);
});

test("random32 - basic output", async () => {
  const src = `
     import lygia::generative::random::random32;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec2f(1.0, 2.0);
       let r1 = random32(p1);
       let r2 = random32(p1); // Same input

       // Test determinism and range (can only fit 3 components, test first 3)
       test::results[0] = vec4f(r1.x, r1.y, r1.z, r2.x);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same input produces same output (test first component)
  expectCloseTo([result[0]], [result[3]]);
  // Regression: exact output value
  expectCloseTo([0.2534], [result[0]]);
});

test("random33 - basic output", async () => {
  const src = `
     import lygia::generative::random::random33;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec3f(1.0, 2.0, 3.0);
       let r1 = random33(p1);
       let r2 = random33(p1); // Same input

       // Test determinism and range (can only fit 3 components, test first 3)
       test::results[0] = vec4f(r1.x, r1.y, r1.z, r2.x);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same input produces same output (test first component)
  expectCloseTo([result[0]], [result[3]]);
  // Regression: exact output value
  expectCloseTo([0.4542], [result[0]]);
});

test("random33 - distribution (x component)", async () => {
  const sampleCount = 1024;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::random::random33;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        let x = f32(i % 16u);
        let y = f32((i / 16u) % 16u);
        let z = f32(i / 256u);
        let sample = random33(vec3f(x, y, z));
        test::results[i] = sample.x;
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount,
  });
  expectDistribution(samples, [0.0, 1.0]);
});

test("random41 - determinism and range", async () => {
  const src = `
     import lygia::generative::random::random41;

     @compute @workgroup_size(1)
     fn foo() {
       test::results[0] = random41(1.0);
     }
   `;
  const result1 = await lygiaTestCompute(src, { elem: "vec4f", size: 1 });
  const result2 = await lygiaTestCompute(src, { elem: "vec4f", size: 1 });

  // Test determinism: same input produces same output across runs
  expectCloseTo(result1, result2);

  // Range check: all components in [0, 1]
  result1.forEach((v) => {
    expect(v).toBeGreaterThanOrEqual(0.0);
    expect(v).toBeLessThanOrEqual(1.0);
  });

  // Regression: exact output value
  expectCloseTo([0.3824, 0.4284, 0.539, 0.4849], result1);
});
test("random42 - hash properties", async () => {
  // Test determinism
  const src1 = `
     import lygia::generative::random::random42;
     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec2f(1.0, 2.0);
       let p2 = vec2f(1.0, 2.0);  // Same input
       test::results[0] = random42(p1) - random42(p2);
     }
   `;
  const determinism = await lygiaTestCompute(src1, { elem: "vec4f", size: 1 });
  // Hash functions are deterministic - same input produces exactly same output
  expectCloseTo([0.0, 0.0, 0.0, 0.0], determinism); // Should be exactly zero

  // Test range and independence
  const src2 = `
     import lygia::generative::random::random42;
     @compute @workgroup_size(1)
     fn foo() {
       test::results[0] = random42(vec2f(1.0, 2.0));
     }
   `;
  const result = await lygiaTestCompute(src2, { elem: "vec4f", size: 1 });

  // All components in [0, 1]
  result.forEach((v) => {
    expect(v).toBeGreaterThanOrEqual(0.0);
    expect(v).toBeLessThanOrEqual(1.0);
  });

  // Components should differ (not all identical)
  const allSame = result.every((v) => Math.abs(v - result[0]) < 0.001);
  expect(allSame).toBe(false);

  // Test avalanche effect: small input change causes significant output change
  const src3 = `
     import lygia::generative::random::random42;
     @compute @workgroup_size(1)
     fn foo() {
       let r1 = random42(vec2f(1.0, 2.0));
       let r3 = random42(vec2f(1.01, 2.0));  // Tiny 1% change in input
       test::results[0] = abs(r1 - r3);  // Difference magnitude
     }
   `;
  const avalanche = await lygiaTestCompute(src3, { elem: "vec4f", size: 1 });
  const avgDiff = avalanche.reduce((a, b) => a + b) / avalanche.length;
  expect(avgDiff).toBeGreaterThan(0.03); // Hash property: small input → significant output change

  // Regression: exact output value
  expectCloseTo([0.6688, 0.9968, 0.6032, 0.9088], result);
});

test("random43 - hash properties", async () => {
  // Test determinism
  const src1 = `
     import lygia::generative::random::random43;
     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec3f(1.0, 2.0, 3.0);
       let p2 = vec3f(1.0, 2.0, 3.0);  // Same input
       test::results[0] = random43(p1) - random43(p2);
     }
   `;
  const determinism = await lygiaTestCompute(src1, { elem: "vec4f", size: 1 });
  // Hash functions are deterministic - same input produces exactly same output
  expectCloseTo([0.0, 0.0, 0.0, 0.0], determinism); // Should be exactly zero

  // Test range and independence
  const src2 = `
     import lygia::generative::random::random43;
     @compute @workgroup_size(1)
     fn foo() {
       test::results[0] = random43(vec3f(1.0, 2.0, 3.0));
     }
   `;
  const result = await lygiaTestCompute(src2, { elem: "vec4f", size: 1 });

  // All components in [0, 1]
  result.forEach((v) => {
    expect(v).toBeGreaterThanOrEqual(0.0);
    expect(v).toBeLessThanOrEqual(1.0);
  });

  // Components should differ (not all identical)
  const allSame = result.every((v) => Math.abs(v - result[0]) < 0.001);
  expect(allSame).toBe(false);

  // Test avalanche effect
  const src3 = `
     import lygia::generative::random::random43;
     @compute @workgroup_size(1)
     fn foo() {
       let r1 = random43(vec3f(1.0, 2.0, 3.0));
       let r3 = random43(vec3f(1.01, 2.0, 3.0));
       test::results[0] = abs(r1 - r3);
     }
   `;
  const avalanche = await lygiaTestCompute(src3, { elem: "vec4f", size: 1 });
  const avgDiff = avalanche.reduce((a, b) => a + b) / avalanche.length;
  expect(avgDiff).toBeGreaterThan(0.1);

  // Regression: exact output value
  expectCloseTo([0.408, 0.2166, 0.9606, 0.4371], result);
});

test("random44 - hash properties", async () => {
  // Test determinism
  const src1 = `
     import lygia::generative::random::random44;
     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec4f(1.0, 2.0, 3.0, 4.0);
       let p2 = vec4f(1.0, 2.0, 3.0, 4.0);  // Same input
       test::results[0] = random44(p1) - random44(p2);
     }
   `;
  const determinism = await lygiaTestCompute(src1, { elem: "vec4f", size: 1 });
  // Hash functions are deterministic - same input produces exactly same output
  expectCloseTo([0.0, 0.0, 0.0, 0.0], determinism); // Should be exactly zero

  // Test range and independence
  const src2 = `
     import lygia::generative::random::random44;
     @compute @workgroup_size(1)
     fn foo() {
       test::results[0] = random44(vec4f(1.0, 2.0, 3.0, 4.0));
     }
   `;
  const result = await lygiaTestCompute(src2, { elem: "vec4f", size: 1 });

  // All components in [0, 1]
  result.forEach((v) => {
    expect(v).toBeGreaterThanOrEqual(0.0);
    expect(v).toBeLessThanOrEqual(1.0);
  });

  // Components should differ (not all identical)
  const allSame = result.every((v) => Math.abs(v - result[0]) < 0.001);
  expect(allSame).toBe(false);

  // Test avalanche effect
  const src3 = `
     import lygia::generative::random::random44;
     @compute @workgroup_size(1)
     fn foo() {
       let r1 = random44(vec4f(1.0, 2.0, 3.0, 4.0));
       let r3 = random44(vec4f(1.01, 2.0, 3.0, 4.0));
       test::results[0] = abs(r1 - r3);
     }
   `;
  const avalanche = await lygiaTestCompute(src3, { elem: "vec4f", size: 1 });
  const avgDiff = avalanche.reduce((a, b) => a + b) / avalanche.length;
  expect(avgDiff).toBeGreaterThan(0.1);

  // Regression: exact output value
  expectCloseTo([0.8164, 0.0728, 0.7236, 0.7064], result);
});
test("srandom2", async () => {
  const src = `
     import lygia::generative::srandom::srandom2;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec2f(1.0, 2.0);
       let p2 = vec2f(1.0, 2.0); // Same point
       let p3 = vec2f(3.0, 4.0); // Different point

       let r1 = srandom2(p1);
       let r2 = srandom2(p2);
       let r3 = srandom2(p3);

       test::results[0] = vec4f(r1, r2, r3, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different inputs produce different outputs
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Regression: exact output value
  expectCloseTo([-91 / 128], [result[0]]);
});

test("srandom2 - distribution", async () => {
  const sampleCount = 512;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::srandom::srandom2;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        let x = f32(i % 32u);
        let y = f32(i / 32u);
        test::results[i] = srandom2(vec2f(x, y));
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount,
  });
  expectDistribution(samples, [-1.0, 1.0]);
});

test("srandom", async () => {
  const src = `
     import lygia::generative::srandom::srandom;

     @compute @workgroup_size(1)
     fn foo() {
       let r1 = srandom(12.34);
       let r2 = srandom(12.34); // Same input
       let r3 = srandom(98.76); // Different input

       test::results[0] = vec4f(r1, r2, r3, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different inputs produce different outputs
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Regression: exact output value
  expectCloseTo([233 / 512], [result[0]]);
});

test("srandom - distribution", async () => {
  const sampleCount = 1024;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::srandom::srandom;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        // Vary inputs more to avoid patterns
        test::results[i] = srandom(f32(i) * 1.234 + 0.567);
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount,
  });
  expectDistribution(samples, [-1.0, 1.0]);
});

test("srandom22", async () => {
  const src = `
     import lygia::generative::srandom::srandom22;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec2f(1.0, 2.0);
       let p2 = vec2f(1.0, 2.0); // Same point
       let p3 = vec2f(3.0, 4.0); // Different point

       let r1 = srandom22(p1);
       let r2 = srandom22(p2);
       let r3 = srandom22(p3);

       test::results[0] = vec4f(r1.x, r1.y, r2.x, r2.y);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same input produces same output
  expectCloseTo([result[0], result[1]], [result[2], result[3]]);
  // Regression: exact output value
  expectCloseTo([-0.3648], [result[0]]);
});

test("srandom22 - distribution (x component)", async () => {
  const sampleCount = 1024;
  const src = `
    import constants::SAMPLE_COUNT;
    import lygia::generative::srandom::srandom22;

    @compute @workgroup_size(1)
    fn main() {
      for (var i = 0u; i < SAMPLE_COUNT; i++) {
        // Vary inputs more to avoid patterns
        let x = f32(i % 32u) * 1.1 + 0.3;
        let y = f32(i / 32u) * 1.3 + 0.7;
        let sample = srandom22(vec2f(x, y));
        test::results[i] = sample.x;
      }
    }
  `;
  const samples = await testDistribution(src, sampleCount, "f32", {
    SAMPLE_COUNT: sampleCount,
  });
  expectDistribution(samples, [-1.0, 1.0]);
});

test("srandom3", async () => {
  const src = `
     import lygia::generative::srandom::srandom3;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec3f(1.0, 2.0, 3.0);
       let p2 = vec3f(1.0, 2.0, 3.0); // Same point
       let p3 = vec3f(4.0, 5.0, 6.0); // Different point

       let r1 = srandom3(p1);
       let r2 = srandom3(p2);
       let r3 = srandom3(p3);

       test::results[0] = vec4f(r1, r2, r3, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different inputs produce different outputs
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Regression: exact output value
  expectCloseTo([17 / 128], [result[0]]);
});

test("srandom33", async () => {
  const src = `
     import lygia::generative::srandom::srandom33;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec3f(1.0, 2.0, 3.0);
       let p2 = vec3f(1.0, 2.0, 3.0); // Same point

       let r1 = srandom33(p1);
       let r2 = srandom33(p2);

       test::results[0] = vec4f(r1.x, r1.y, r1.z, r2.x);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same input produces same output (check first component)
  expectCloseTo([result[0]], [result[3]]);
  // Regression: exact output value
  expectCloseTo([-151 / 256], [result[0]]);
});

test("srandom4", async () => {
  const src = `
     import lygia::generative::srandom::srandom4;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec4f(1.0, 2.0, 3.0, 4.0);
       let p2 = vec4f(1.0, 2.0, 3.0, 4.0); // Same point
       let p3 = vec4f(5.0, 6.0, 7.0, 8.0); // Different point

       let r1 = srandom4(p1);
       let r2 = srandom4(p2);
       let r3 = srandom4(p3);

       test::results[0] = vec4f(r1, r2, r3, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different inputs produce different outputs
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Regression: exact output value
  expectCloseTo([-87 / 128], [result[0]]);
});

test("srandom_tile22", async () => {
  const src = `
     import lygia::generative::srandom::srandom_tile22;

     @compute @workgroup_size(1)
     fn foo() {
       let tileLength = 4.0;
       let p1 = vec2f(1.0, 2.0);
       let p2 = vec2f(5.0, 6.0); // Should tile with period=4.0

       // p2 = p1 + (4.0, 4.0), so after tiling they should be the same
       let r1 = srandom_tile22(p1, tileLength);
       let r2 = srandom_tile22(p2, tileLength);

       test::results[0] = vec4f(r1.x, r1.y, r2.x, r2.y);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test tiling: points separated by tileLength should produce same output
  expectCloseTo([result[0], result[1]], [result[2], result[3]]);
  // Regression: exact output value
  expectCloseTo([-0.3648], [result[0]]);
});

test("srandom_tile33", async () => {
  const src = `
     import lygia::generative::srandom::srandom_tile33;

     @compute @workgroup_size(1)
     fn foo() {
       let tileLength = 4.0;
       let p1 = vec3f(1.0, 2.0, 3.0);
       let p2 = vec3f(5.0, 6.0, 7.0); // Should tile with period=4.0

       // p2 = p1 + (4.0, 4.0, 4.0), so after tiling they should be the same
       let r1 = srandom_tile33(p1, tileLength);
       let r2 = srandom_tile33(p2, tileLength);

       test::results[0] = vec4f(r1.x, r1.y, r1.z, r2.x);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test tiling: points separated by tileLength should produce same output (check first component)
  expectCloseTo([result[0]], [result[3]]);
  // Regression: exact output value
  expectCloseTo([-151 / 256], [result[0]]);
});
