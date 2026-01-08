import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("levelsInputRange3", async () => {
  const src = `
     import lygia::color::levels::inputRange::levelsInputRange3;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(0.3, 0.5, 0.7);
       let result = levelsInputRange3(color, vec3f(0.2), vec3f(0.8));
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // (v - iMin) / (iMax - iMin) clamped to [0, 1]
  // (0.3 - 0.2) / (0.8 - 0.2) = 0.1 / 0.6 = 0.1667
  // (0.5 - 0.2) / 0.6 = 0.5
  // (0.7 - 0.2) / 0.6 = 0.8333
  expectCloseTo([0.1667, 0.5, 0.8333], result);
});

test("levelsGamma3", async () => {
  const src = `
     import lygia::color::levels::gamma::levelsGamma3;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(0.25, 0.5, 0.75);
       let result = levelsGamma3(color, vec3f(2.0));
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // pow(v, 1/gamma) = pow(v, 0.5) = sqrt(v)
  expectCloseTo([0.5, Math.SQRT1_2, 0.866], result);
});

test("levels3Float", async () => {
  const src = `
     import lygia::color::levels::levels3Float;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(0.3, 0.5, 0.7);
       // Remap input [0.2, 0.8] to output [0.1, 0.9] with gamma 2.0
       let result = levels3Float(color, 0.2, 2.0, 0.8, 0.1, 0.9);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Step 1: inputRange: (v - 0.2) / (0.8 - 0.2) = (v - 0.2) / 0.6
  //   r: (0.3 - 0.2) / 0.6 = 0.1667
  //   g: (0.5 - 0.2) / 0.6 = 0.5
  //   b: (0.7 - 0.2) / 0.6 = 0.8333
  // Step 2: gamma: pow(v, 1/2.0) = sqrt(v)
  //   r: sqrt(0.1667) = 0.4082
  //   g: sqrt(0.5) = INV_SQRT2 (≈ 0.7071)
  //   b: sqrt(0.8333) = 0.9129
  // Step 3: outputRange: mix(0.1, 0.9, v) = 0.1 + v * 0.8
  //   r: 0.1 + 0.4082 * 0.8 = 0.4266
  //   g: 0.1 + INV_SQRT2 * 0.8 = 0.6657
  //   b: 0.1 + 0.9129 * 0.8 = 0.8303
  expectCloseTo([0.4266, 0.6657, 0.8303], result);
});

test("levels3", async () => {
  const src = `
     import lygia::color::levels::levels3;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(0.4, 0.6, 0.8);
       // Input range [0.2, 0.9], gamma 2.0, output range [0.1, 0.8]
       let result = levels3(color, vec3f(0.2), vec3f(2.0), vec3f(0.9), vec3f(0.1), vec3f(0.8));
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Step 1: inputRange: (v - 0.2) / (0.9 - 0.2)
  //   r: (0.4 - 0.2) / 0.7 = 0.2857
  //   g: (0.6 - 0.2) / 0.7 = 0.5714
  //   b: (0.8 - 0.2) / 0.7 = 0.8571
  // Step 2: gamma: pow(v, 0.5) = sqrt(v)
  //   r: sqrt(0.2857) = 0.5345
  //   g: sqrt(0.5714) = 0.7560
  //   b: sqrt(0.8571) = 0.9258
  // Step 3: outputRange: mix(0.1, 0.8, v) = 0.1 + v * 0.7
  //   r: 0.1 + 0.5345 * 0.7 = 0.4742
  //   g: 0.1 + 0.7560 * 0.7 = 0.6292
  //   b: 0.1 + 0.9258 * 0.7 = 0.7481
  expectCloseTo([0.4742, 0.6292, 0.7481], result);
});

test("levels4", async () => {
  const src = `
     import lygia::color::levels::levels4;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.4, 0.6, 0.8, 0.75);
       // Same settings as levels3 test
       let result = levels4(color, vec3f(0.2), vec3f(2.0), vec3f(0.9), vec3f(0.1), vec3f(0.8));
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // RGB should match levels3 test, alpha preserved
  expectCloseTo([0.4742, 0.6292, 0.7481, 0.75], result);
});

test("levels4Float", async () => {
  const src = `
     import lygia::color::levels::levels4Float;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.5, 0.7, 0.3, 0.85);
       // Input [0.3, 0.8], gamma 1.5, output [0.2, 0.9]
       let result = levels4Float(color, 0.3, 1.5, 0.8, 0.2, 0.9);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Step 1: inputRange: (v - 0.3) / (0.8 - 0.3) = (v - 0.3) / 0.5
  //   r: (0.5 - 0.3) / 0.5 = 0.4
  //   g: (0.7 - 0.3) / 0.5 = 0.8
  //   b: (0.3 - 0.3) / 0.5 = 0.0
  // Step 2: gamma: pow(v, 1/1.5) = pow(v, 0.6667)
  //   r: pow(0.4, 0.6667) = 0.5429
  //   g: pow(0.8, 0.6667) = 0.8618
  //   b: pow(0.0, 0.6667) = 0.0
  // Step 3: outputRange: mix(0.2, 0.9, v) = 0.2 + v * 0.7
  //   r: 0.2 + 0.5429 * 0.7 = 0.5800
  //   g: 0.2 + 0.8618 * 0.7 = 0.8032
  //   b: 0.2 + 0.0 * 0.7 = 0.2
  //   a: preserved at 0.85
  expectCloseTo([0.58, 0.8032, 0.2, 0.85], result);
});

// Gamma function tests
test("levelsGamma3Float", async () => {
  const src = `
     import lygia::color::levels::gamma::levelsGamma3Float;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(0.16, 0.36, 0.64);
       let result = levelsGamma3Float(color, 2.0);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // pow(v, 1/2.0) = sqrt(v)
  // sqrt(0.16) = 0.4, sqrt(0.36) = 0.6, sqrt(0.64) = 0.8
  expectCloseTo([0.4, 0.6, 0.8], result);
});

test("levelsGamma4", async () => {
  const src = `
     import lygia::color::levels::gamma::levelsGamma4;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.25, 0.5, 0.75, 0.9);
       let result = levelsGamma4(color, vec3f(2.0, 1.5, 3.0));
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // pow(v, 1/gamma)
  // r: pow(0.25, 0.5) = 0.5
  // g: pow(0.5, 1/1.5) = pow(0.5, 0.6667) = 0.6300
  // b: pow(0.75, 1/3.0) = 0.9086
  // a: preserved at 0.9
  expectCloseTo([0.5, 0.63, 0.9086, 0.9], result);
});

test("levelsGamma4Float", async () => {
  const src = `
     import lygia::color::levels::gamma::levelsGamma4Float;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.09, 0.25, 0.49, 0.85);
       let result = levelsGamma4Float(color, 2.0);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // pow(v, 1/2.0) = sqrt(v)
  // sqrt(0.09) = 0.3, sqrt(0.25) = 0.5, sqrt(0.49) = 0.7
  // a: preserved at 0.85
  expectCloseTo([0.3, 0.5, 0.7, 0.85], result);
});

// Input Range function tests
test("levelsInputRange3Float", async () => {
  const src = `
     import lygia::color::levels::inputRange::levelsInputRange3Float;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(0.2, 0.5, 0.8);
       let result = levelsInputRange3Float(color, 0.1, 0.9);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // (v - iMin) / (iMax - iMin) clamped to [0, 1]
  // (0.2 - 0.1) / (0.9 - 0.1) = 0.1 / 0.8 = 0.125
  // (0.5 - 0.1) / 0.8 = 0.5
  // (0.8 - 0.1) / 0.8 = 0.875
  expectCloseTo([0.125, 0.5, 0.875], result);
});

test("levelsInputRange4", async () => {
  const src = `
     import lygia::color::levels::inputRange::levelsInputRange4;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.3, 0.6, 0.9, 0.75);
       let result = levelsInputRange4(color, vec3f(0.2, 0.4, 0.5), vec3f(0.8, 0.9, 1.0));
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Per-channel input range mapping:
  // r: (0.3 - 0.2) / (0.8 - 0.2) = 0.1 / 0.6 = 0.1667
  // g: (0.6 - 0.4) / (0.9 - 0.4) = 0.2 / 0.5 = 0.4
  // b: (0.9 - 0.5) / (1.0 - 0.5) = 0.4 / 0.5 = 0.8
  // a: preserved at 0.75
  expectCloseTo([0.1667, 0.4, 0.8, 0.75], result);
});

test("levelsInputRange4Float", async () => {
  const src = `
     import lygia::color::levels::inputRange::levelsInputRange4Float;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.15, 0.45, 0.75, 0.95);
       let result = levelsInputRange4Float(color, 0.1, 0.8);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // (v - 0.1) / (0.8 - 0.1) = (v - 0.1) / 0.7
  // r: (0.15 - 0.1) / 0.7 = 0.0714
  // g: (0.45 - 0.1) / 0.7 = 0.5
  // b: (0.75 - 0.1) / 0.7 = 0.9286
  // a: preserved at 0.95
  expectCloseTo([0.0714, 0.5, 0.9286, 0.95], result);
});

// Output Range function tests
test("levelsOutputRange3Float", async () => {
  const src = `
     import lygia::color::levels::outputRange::levelsOutputRange3Float;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(0.0, 0.5, 1.0);
       let result = levelsOutputRange3Float(color, 0.2, 0.9);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // mix(0.2, 0.9, v) = 0.2 + v * (0.9 - 0.2) = 0.2 + v * 0.7
  // r: 0.2 + 0.0 * 0.7 = 0.2
  // g: 0.2 + 0.5 * 0.7 = 0.55
  // b: 0.2 + 1.0 * 0.7 = 0.9
  expectCloseTo([0.2, 0.55, 0.9], result);
});

test("levelsOutputRange4", async () => {
  const src = `
     import lygia::color::levels::outputRange::levelsOutputRange4;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.25, 0.5, 0.75, 0.8);
       let result = levelsOutputRange4(color, vec3f(0.1, 0.2, 0.3), vec3f(0.8, 0.9, 1.0));
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Per-channel output range mapping: mix(oMin, oMax, v)
  // r: mix(0.1, 0.8, 0.25) = 0.1 + 0.25 * 0.7 = 0.275
  // g: mix(0.2, 0.9, 0.5) = 0.2 + 0.5 * 0.7 = 0.55
  // b: mix(0.3, 1.0, 0.75) = 0.3 + 0.75 * 0.7 = 0.825
  // a: preserved at 0.8
  expectCloseTo([0.275, 0.55, 0.825, 0.8], result);
});

test("levelsOutputRange4Float", async () => {
  const src = `
     import lygia::color::levels::outputRange::levelsOutputRange4Float;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.2, 0.6, 0.8, 0.7);
       let result = levelsOutputRange4Float(color, 0.3, 0.95);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // mix(0.3, 0.95, v) = 0.3 + v * (0.95 - 0.3) = 0.3 + v * 0.65
  // r: 0.3 + 0.2 * 0.65 = 0.43
  // g: 0.3 + 0.6 * 0.65 = 0.69
  // b: 0.3 + 0.8 * 0.65 = 0.82
  // a: preserved at 0.7
  expectCloseTo([0.43, 0.69, 0.82, 0.7], result);
});
