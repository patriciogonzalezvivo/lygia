import { expect, test } from "vitest";
import {
  expectCloseTo,
  lygiaTestCompute,
  testDistribution,
} from "./testUtil.ts";

test("cnoise2", async () => {
  const pairCount = 256;
  const sampleCount = pairCount * 2;
  const src = `
     import constants::PAIR_COUNT;
     import lygia::generative::cnoise::cnoise2;

     @compute @workgroup_size(1)
     fn foo() {
       for (var i = 0u; i < PAIR_COUNT; i++) {
         // Offset by 0.5 so first point is at (0.5, 0.5) - our regression point
         let x = f32(i % 16u) * 0.2 + 0.5;
         let y = f32(i / 16u) * 0.2 + 0.5;

         test::results[i * 2] = cnoise2(vec2f(x, y));
         test::results[i * 2 + 1] = cnoise2(vec2f(x + 0.01, y + 0.01));
       }
     }
   `;
  const result = await testDistribution(src, sampleCount, {
    constants: { PAIR_COUNT: pairCount },
  });

  // Continuity: check all 256 pairs
  let maxDiff = 0;
  for (let i = 0; i < pairCount; i++) {
    const diff = Math.abs(result[i * 2 + 1] - result[i * 2]);
    maxDiff = Math.max(maxDiff, diff);
  }
  expect(maxDiff).toBeLessThan(0.1);

  // Range: all values in [-1, 1]
  expect(Math.min(...result)).toBeGreaterThanOrEqual(-1.0);
  expect(Math.max(...result)).toBeLessThanOrEqual(1.0);

  // Regression: result[0] is naturally at (0.5, 0.5)
  expectCloseTo([-0.4915], [result[0]]);
});

test("cnoise3", async () => {
  const pairCount = 256;
  const sampleCount = pairCount * 2;
  const src = `
     import constants::PAIR_COUNT;
     import lygia::generative::cnoise::cnoise3;

     @compute @workgroup_size(1)
     fn foo() {
       for (var i = 0u; i < PAIR_COUNT; i++) {
         // Offset by 0.5 so first point is at (0.5, 0.5, 0.5) - our regression point
         let x = f32(i % 8u) * 0.2 + 0.5;
         let y = f32((i / 8u) % 8u) * 0.2 + 0.5;
         let z = f32(i / 64u) * 0.2 + 0.5;

         test::results[i * 2] = cnoise3(vec3f(x, y, z));
         test::results[i * 2 + 1] = cnoise3(vec3f(x + 0.01, y + 0.01, z + 0.01));
       }
     }
   `;
  const result = await testDistribution(src, sampleCount, {
    constants: { PAIR_COUNT: pairCount },
  });

  // Continuity: check all 256 pairs
  let maxDiff = 0;
  for (let i = 0; i < pairCount; i++) {
    const diff = Math.abs(result[i * 2 + 1] - result[i * 2]);
    maxDiff = Math.max(maxDiff, diff);
  }
  expect(maxDiff).toBeLessThan(0.1);

  // Range: all values in [-1, 1]
  expect(Math.min(...result)).toBeGreaterThanOrEqual(-1.0);
  expect(Math.max(...result)).toBeLessThanOrEqual(1.0);

  // Regression: result[0] is naturally at (0.5, 0.5, 0.5)
  expectCloseTo([-0.3962], [result[0]]);
});

test("cnoise4", async () => {
  const pairCount = 256;
  const sampleCount = pairCount * 2;
  const src = `
     import constants::PAIR_COUNT;
     import lygia::generative::cnoise::cnoise4;

     @compute @workgroup_size(1)
     fn foo() {
       for (var i = 0u; i < PAIR_COUNT; i++) {
         // Offset by 0.5 so first point is at (0.5, 0.5, 0.5, 0.5) - our regression point
         let x = f32(i % 4u) * 0.2 + 0.5;
         let y = f32((i / 4u) % 4u) * 0.2 + 0.5;
         let z = f32((i / 16u) % 4u) * 0.2 + 0.5;
         let w = f32(i / 64u) * 0.2 + 0.5;

         test::results[i * 2] = cnoise4(vec4f(x, y, z, w));
         test::results[i * 2 + 1] = cnoise4(vec4f(x + 0.01, y + 0.01, z + 0.01, w + 0.01));
       }
     }
   `;
  const result = await testDistribution(src, sampleCount, {
    constants: { PAIR_COUNT: pairCount },
  });

  // Continuity: check all 256 pairs
  let maxDiff = 0;
  for (let i = 0; i < pairCount; i++) {
    const diff = Math.abs(result[i * 2 + 1] - result[i * 2]);
    maxDiff = Math.max(maxDiff, diff);
  }
  expect(maxDiff).toBeLessThan(0.1);

  // Range: all values in [-1, 1]
  expect(Math.min(...result)).toBeGreaterThanOrEqual(-1.0);
  expect(Math.max(...result)).toBeLessThanOrEqual(1.0);

  // Regression: result[0] is naturally at (0.5, 0.5, 0.5, 0.5)
  expectCloseTo([0.0203], [result[0]]);
});

test("snoise2", async () => {
  const pairCount = 256;
  const sampleCount = pairCount * 2;
  const src = `
     import constants::PAIR_COUNT;
     import lygia::generative::snoise::snoise2;

     @compute @workgroup_size(1)
     fn foo() {
       for (var i = 0u; i < PAIR_COUNT; i++) {
         // Offset by 1.0 so first point is at (1.0, 2.0) - our regression point
         let x = f32(i % 16u) * 0.2 + 1.0;
         let y = f32(i / 16u) * 0.2 + 2.0;

         test::results[i * 2] = snoise2(vec2f(x, y));
         test::results[i * 2 + 1] = snoise2(vec2f(x + 0.01, y + 0.01));
       }
     }
   `;
  const result = await testDistribution(src, sampleCount, {
    constants: { PAIR_COUNT: pairCount },
  });

  // Continuity: check all 256 pairs
  let maxDiff = 0;
  for (let i = 0; i < pairCount; i++) {
    const diff = Math.abs(result[i * 2 + 1] - result[i * 2]);
    maxDiff = Math.max(maxDiff, diff);
  }
  expect(maxDiff).toBeLessThan(0.2);

  // Range: all values in approximately [-1, 1] (allow small overshoot for simplex noise)
  expect(Math.min(...result)).toBeGreaterThanOrEqual(-1.1);
  expect(Math.max(...result)).toBeLessThanOrEqual(1.1);

  // Regression: result[0] is naturally at (1.0, 2.0)
  expectCloseTo([0.3683], [result[0]]);
});

test("snoise3", async () => {
  const pairCount = 256;
  const sampleCount = pairCount * 2;
  const src = `
     import constants::PAIR_COUNT;
     import lygia::generative::snoise::snoise3;

     @compute @workgroup_size(1)
     fn foo() {
       for (var i = 0u; i < PAIR_COUNT; i++) {
         // Offset so first point is at (1.0, 2.0, 3.0) - our regression point
         let x = f32(i % 8u) * 0.2 + 1.0;
         let y = f32((i / 8u) % 8u) * 0.2 + 2.0;
         let z = f32(i / 64u) * 0.2 + 3.0;

         test::results[i * 2] = snoise3(vec3f(x, y, z));
         test::results[i * 2 + 1] = snoise3(vec3f(x + 0.01, y + 0.01, z + 0.01));
       }
     }
   `;
  const result = await testDistribution(src, sampleCount, {
    constants: { PAIR_COUNT: pairCount },
  });

  // Continuity: check all 256 pairs
  let maxDiff = 0;
  for (let i = 0; i < pairCount; i++) {
    const diff = Math.abs(result[i * 2 + 1] - result[i * 2]);
    maxDiff = Math.max(maxDiff, diff);
  }
  expect(maxDiff).toBeLessThan(0.35);

  // Range: all values in approximately [-1, 1] (allow small overshoot for simplex noise)
  expect(Math.min(...result)).toBeGreaterThanOrEqual(-1.1);
  expect(Math.max(...result)).toBeLessThanOrEqual(1.1);

  // Regression: result[0] is naturally at (1.0, 2.0, 3.0)
  expectCloseTo([0.7335], [result[0]]);
});

test("snoise4", async () => {
  const pairCount = 256;
  const sampleCount = pairCount * 2;
  const src = `
     import constants::PAIR_COUNT;
     import lygia::generative::snoise::snoise4;

     @compute @workgroup_size(1)
     fn foo() {
       for (var i = 0u; i < PAIR_COUNT; i++) {
         // Offset so first point is at (1.0, 2.0, 3.0, 4.0) - our regression point
         let x = f32(i % 4u) * 0.2 + 1.0;
         let y = f32((i / 4u) % 4u) * 0.2 + 2.0;
         let z = f32((i / 16u) % 4u) * 0.2 + 3.0;
         let w = f32(i / 64u) * 0.2 + 4.0;

         test::results[i * 2] = snoise4(vec4f(x, y, z, w));
         test::results[i * 2 + 1] = snoise4(vec4f(x + 0.01, y + 0.01, z + 0.01, w + 0.01));
       }
     }
   `;
  const result = await testDistribution(src, sampleCount, {
    constants: { PAIR_COUNT: pairCount },
  });

  // Continuity: check all 256 pairs
  let maxDiff = 0;
  for (let i = 0; i < pairCount; i++) {
    const diff = Math.abs(result[i * 2 + 1] - result[i * 2]);
    maxDiff = Math.max(maxDiff, diff);
  }
  expect(maxDiff).toBeLessThan(0.2);

  // Range: all values in approximately [-1, 1] (allow small overshoot for simplex noise)
  expect(Math.min(...result)).toBeGreaterThanOrEqual(-1.1);
  expect(Math.max(...result)).toBeLessThanOrEqual(1.1);

  // Regression: result[0] is naturally at (1.0, 2.0, 3.0, 4.0)
  expectCloseTo([-0.3748], [result[0]]);
});

test("snoise22", async () => {
  const src = `
     import lygia::generative::snoise::snoise22;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec2f(1.0, 2.0);
       let p2 = vec2f(1.0, 2.0); // Same point

       let n1 = snoise22(p1);
       let n2 = snoise22(p2);

       test::results[0] = vec4f(n1.x, n1.y, n2.x, n2.y);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same input produces same output
  expectCloseTo([result[0], result[1]], [result[2], result[3]]);
});

test("snoise33", async () => {
  const src = `
     import lygia::generative::snoise::snoise33;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec3f(1.0, 2.0, 3.0);
       let p2 = vec3f(1.0, 2.0, 3.0); // Same point
       let p3 = vec3f(1.01, 2.01, 3.01); // Nearby point

       let n1 = snoise33(p1);
       let n2 = snoise33(p2);
       let n3 = snoise33(p3);

       // Test continuity by computing difference
       test::results[0] = vec4f(n1.x, n1.y, n1.z, length(n3 - n1));
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test continuity: nearby points should have similar values
  expect(result[3]).toBeLessThan(0.2);
  // Regression: exact output value
  expectCloseTo([0.7335], [result[0]]);
});

test("snoise34", async () => {
  const src = `
     import lygia::generative::snoise::snoise34;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec4f(1.0, 2.0, 3.0, 4.0);
       let p2 = vec4f(1.0, 2.0, 3.0, 4.0); // Same point
       let p3 = vec4f(1.01, 2.01, 3.01, 4.01); // Nearby point

       let n1 = snoise34(p1);
       let n2 = snoise34(p2);
       let n3 = snoise34(p3);

       // Test continuity by computing difference
       test::results[0] = vec4f(n1.x, n1.y, n1.z, length(n3 - n1));
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test continuity: nearby points should have similar values
  expect(result[3]).toBeLessThan(0.2);
  // Regression: exact output value
  expectCloseTo([-0.3748], [result[0]]);
});

test("pnoise2", async () => {
  const pairCount = 256;
  const sampleCount = pairCount * 2;
  const src = `
     import constants::PAIR_COUNT;
     import lygia::generative::pnoise::pnoise2;

     @compute @workgroup_size(1)
     fn foo() {
       let period = vec2f(4.0, 4.0);
       for (var i = 0u; i < PAIR_COUNT; i++) {
         // Offset by 0.5 so first point is at (0.5, 0.5) - our regression point
         let x = f32(i % 16u) * 0.2 + 0.5;
         let y = f32(i / 16u) * 0.2 + 0.5;

         test::results[i * 2] = pnoise2(vec2f(x, y), period);
         test::results[i * 2 + 1] = pnoise2(vec2f(x + 0.01, y + 0.01), period);
       }
     }
   `;
  const result = await testDistribution(src, sampleCount, {
    constants: { PAIR_COUNT: pairCount },
  });

  // Continuity: check all 256 pairs
  let maxDiff = 0;
  for (let i = 0; i < pairCount; i++) {
    const diff = Math.abs(result[i * 2 + 1] - result[i * 2]);
    maxDiff = Math.max(maxDiff, diff);
  }
  expect(maxDiff).toBeLessThan(0.1);

  // Range: all values in [-1, 1]
  expect(Math.min(...result)).toBeGreaterThanOrEqual(-1.0);
  expect(Math.max(...result)).toBeLessThanOrEqual(1.0);

  // Regression: result[0] is naturally at (0.5, 0.5)
  expectCloseTo([-0.4915], [result[0]]);
});

test("pnoise2 - periodicity", async () => {
  const src = `
     import lygia::generative::pnoise::pnoise2;

     @compute @workgroup_size(1)
     fn foo() {
       let period = vec2f(4.0, 4.0);
       let p = vec2f(0.5, 0.5);

       // Test periodicity: pnoise(p, period) == pnoise(p + period, period)
       let n1 = pnoise2(p, period);
       let n2 = pnoise2(p + period, period);
       let n3 = pnoise2(p + period * 2.0, period);

       test::results[0] = vec4f(n1, n2, n3, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test periodicity property: noise repeats exactly after one period
  expectCloseTo([result[0], result[0]], [result[1], result[2]]);
});

test("pnoise3", async () => {
  const pairCount = 256;
  const sampleCount = pairCount * 2;
  const src = `
     import constants::PAIR_COUNT;
     import lygia::generative::pnoise::pnoise3;

     @compute @workgroup_size(1)
     fn foo() {
       let period = vec3f(4.0, 4.0, 4.0);
       for (var i = 0u; i < PAIR_COUNT; i++) {
         // Offset by 0.5 so first point is at (0.5, 0.5, 0.5) - our regression point
         let x = f32(i % 8u) * 0.2 + 0.5;
         let y = f32((i / 8u) % 8u) * 0.2 + 0.5;
         let z = f32(i / 64u) * 0.2 + 0.5;

         test::results[i * 2] = pnoise3(vec3f(x, y, z), period);
         test::results[i * 2 + 1] = pnoise3(vec3f(x + 0.01, y + 0.01, z + 0.01), period);
       }
     }
   `;
  const result = await testDistribution(src, sampleCount, {
    constants: { PAIR_COUNT: pairCount },
  });

  // Continuity: check all 256 pairs
  let maxDiff = 0;
  for (let i = 0; i < pairCount; i++) {
    const diff = Math.abs(result[i * 2 + 1] - result[i * 2]);
    maxDiff = Math.max(maxDiff, diff);
  }
  expect(maxDiff).toBeLessThan(0.1);

  // Range: all values in [-1, 1]
  expect(Math.min(...result)).toBeGreaterThanOrEqual(-1.0);
  expect(Math.max(...result)).toBeLessThanOrEqual(1.0);

  // Regression: result[0] is naturally at (0.5, 0.5, 0.5)
  expectCloseTo([-0.3962], [result[0]]);
});

test("pnoise3 - periodicity", async () => {
  const src = `
     import lygia::generative::pnoise::pnoise3;

     @compute @workgroup_size(1)
     fn foo() {
       let period = vec3f(4.0, 4.0, 4.0);
       let p = vec3f(0.5, 0.5, 0.5);

       // Test periodicity: pnoise(p, period) == pnoise(p + period, period)
       let n1 = pnoise3(p, period);
       let n2 = pnoise3(p + period, period);
       let n3 = pnoise3(p + period * 2.0, period);

       test::results[0] = vec4f(n1, n2, n3, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test periodicity property: noise repeats exactly after one period
  expectCloseTo([result[0], result[0]], [result[1], result[2]]);
});

test("pnoise4", async () => {
  const pairCount = 256;
  const sampleCount = pairCount * 2;
  const src = `
     import constants::PAIR_COUNT;
     import lygia::generative::pnoise::pnoise4;

     @compute @workgroup_size(1)
     fn foo() {
       let period = vec4f(4.0, 4.0, 4.0, 4.0);
       for (var i = 0u; i < PAIR_COUNT; i++) {
         // Offset by 0.5 so first point is at (0.5, 0.5, 0.5, 0.5) - our regression point
         let x = f32(i % 4u) * 0.2 + 0.5;
         let y = f32((i / 4u) % 4u) * 0.2 + 0.5;
         let z = f32((i / 16u) % 4u) * 0.2 + 0.5;
         let w = f32(i / 64u) * 0.2 + 0.5;

         test::results[i * 2] = pnoise4(vec4f(x, y, z, w), period);
         test::results[i * 2 + 1] = pnoise4(vec4f(x + 0.01, y + 0.01, z + 0.01, w + 0.01), period);
       }
     }
   `;
  const result = await testDistribution(src, sampleCount, {
    constants: { PAIR_COUNT: pairCount },
  });

  // Continuity: check all 256 pairs
  let maxDiff = 0;
  for (let i = 0; i < pairCount; i++) {
    const diff = Math.abs(result[i * 2 + 1] - result[i * 2]);
    maxDiff = Math.max(maxDiff, diff);
  }
  expect(maxDiff).toBeLessThan(0.1);

  // Range: all values in [-1, 1]
  expect(Math.min(...result)).toBeGreaterThanOrEqual(-1.0);
  expect(Math.max(...result)).toBeLessThanOrEqual(1.0);

  // Regression: result[0] is naturally at (0.5, 0.5, 0.5, 0.5)
  expectCloseTo([0.0203], [result[0]]);
});

test("pnoise4 - periodicity", async () => {
  const period = "vec4f(4.0, 4.0, 4.0, 4.0)";
  const p = "vec4f(0.5, 0.5, 0.5, 0.5)";

  // Test periodicity by running three separate shader invocations
  const src1 = `
     import lygia::generative::pnoise::pnoise4;
     @compute @workgroup_size(1)
     fn foo() {
       test::results[0] = pnoise4(${p}, ${period});
     }
   `;
  const src2 = `
     import lygia::generative::pnoise::pnoise4;
     @compute @workgroup_size(1)
     fn foo() {
       test::results[0] = pnoise4(${p} + ${period}, ${period});
     }
   `;
  const src3 = `
     import lygia::generative::pnoise::pnoise4;
     @compute @workgroup_size(1)
     fn foo() {
       test::results[0] = pnoise4(${p} + ${period} * 2.0, ${period});
     }
   `;

  const n1 = await lygiaTestCompute(src1, { elem: "f32", size: 1 });
  const n2 = await lygiaTestCompute(src2, { elem: "f32", size: 1 });
  const n3 = await lygiaTestCompute(src3, { elem: "f32", size: 1 });

  // Test periodicity property: noise repeats exactly after one period
  expectCloseTo([n1[0], n1[0]], [n2[0], n3[0]]);
});
