import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("tonemapACES3", async () => {
  const src = `
     import lygia::color::tonemap::aces::tonemapACES3;

     @compute @workgroup_size(1)
     fn foo() {
       let hdr = vec3f(2.0, 1.5, 1.0); // HDR color
       let result = tonemapACES3(hdr);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // ACES formula: saturate((v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14))
  // For HDR input [2.0, 1.5, 1.0], the ACES curve maps to LDR:
  // R: 0.9149, G: 0.8768, B: 0.8038
  expectCloseTo([0.9149, 0.8768, 0.8038], result);
});

test("tonemapACES4", async () => {
  const src = `
     import lygia::color::tonemap::aces::tonemapACES4;

     @compute @workgroup_size(1)
     fn foo() {
       let hdr = vec4f(2.0, 1.5, 1.0, 0.8);
       let result = tonemapACES4(hdr);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // ACES tonemap on RGB + alpha preserved
  // Same computation as tonemapACES3 for RGB, alpha unchanged
  expectCloseTo([0.9149, 0.8768, 0.8038, 0.8], result);
});

test("tonemapDebug3", async () => {
  const src = `
     import lygia::color::tonemap::debug::tonemapDebug3;

     @compute @workgroup_size(1)
     fn foo() {
       let hdr = vec3f(1.5, 1.0, 0.5);
       let result = tonemapDebug3(hdr);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Debug tonemap converts based on luminance relative to 18% gray
  // luma = 1.5 * 0.2125 + 1.0 * 0.7154 + 0.5 * 0.0721 ≈ 1.0702
  // stops = log2(1.0702 / 0.18) ≈ 2.57
  // clamped to [0, 15]: 2.57 + 5.0 = 7.57
  // index 7 (green) mixed with index 8 (yellow)
  // green = [0.0, 0.7843, 0.0], yellow = [1.0, 1.0, 0.0]
  // mix with t = 0.57: [0.5718, 0.9076, 0.0]
  expectCloseTo([0.5718, 0.9076, 0.0], result);
});

test("tonemapFilmic3", async () => {
  const src = `
     import lygia::color::tonemap::filmic::tonemapFilmic3;

     @compute @workgroup_size(1)
     fn foo() {
       let hdr = vec3f(2.0, 1.5, 1.0);
       let result = tonemapFilmic3(hdr);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Filmic tonemap: v = max(v - 0.004, 0), then (v * (6.2 * v + 0.5)) / (v * (6.2 * v + 1.7) + 0.06)
  // This is a complex curve that produces high values for HDR inputs
  // For [2.0, 1.5, 1.0]: approximately [0.9128, 0.8874, 0.8412]
  expectCloseTo([0.9128, 0.8874, 0.8412], result);
});

test("tonemapLinear3", async () => {
  const src = `
     import lygia::color::tonemap::linear::tonemapLinear3;

     @compute @workgroup_size(1)
     fn foo() {
       let hdr = vec3f(2.0, 1.5, 1.0);
       let result = tonemapLinear3(hdr);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Linear tonemap is identity (no modification)
  expectCloseTo([2.0, 1.5, 1.0], result);
});

test("tonemapReinhard3", async () => {
  const src = `
     import lygia::color::tonemap::reinhard::tonemapReinhard3;

     @compute @workgroup_size(1)
     fn foo() {
       let hdr = vec3f(2.0, 1.5, 1.0);
       let result = tonemapReinhard3(hdr);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Reinhard: v / (1 + luma), where luma = dot(v, [0.2125, 0.7154, 0.0721])
  // luma = 2.0 * 0.2125 + 1.5 * 0.7154 + 1.0 * 0.0721 ≈ 1.5706
  // result = [2.0, 1.5, 1.0] / (1 + 1.5706) = [2.0, 1.5, 1.0] / 2.5706
  // = [0.7782, 0.5836, 0.3891]
  expectCloseTo([0.7782, 0.5836, 0.3891], result);
});

test("tonemapReinhardJodie3", async () => {
  const src = `
     import lygia::color::tonemap::reinhardJodie::tonemapReinhardJodie3;

     @compute @workgroup_size(1)
     fn foo() {
       let hdr = vec3f(2.0, 1.5, 1.0);
       let result = tonemapReinhardJodie3(hdr);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Reinhard-Jodie: luma = 1.5706, tc = x/(x+1) = [0.6667, 0.6, 0.5]
  // mix(x/(l+1), tc, tc) where x/(l+1) = [0.7782, 0.5836, 0.3891]
  // R: mix(0.7782, 0.6667, 0.6667) = 0.7782 + (0.6667 - 0.7782) * 0.6667 ≈ 0.7038
  // G: mix(0.5836, 0.6, 0.6) = 0.5836 + (0.6 - 0.5836) * 0.6 ≈ 0.5935
  // B: mix(0.3891, 0.5, 0.5) = 0.3891 + (0.5 - 0.3891) * 0.5 ≈ 0.4445
  expectCloseTo([0.7038, 0.5935, 0.4445], result);
});

test("tonemapUncharted3", async () => {
  const src = `
     import lygia::color::tonemap::uncharted::tonemapUncharted3;

     @compute @workgroup_size(1)
     fn foo() {
       let hdr = vec3f(2.0, 1.5, 1.0);
       let result = tonemapUncharted3(hdr);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Uncharted uses John Hable's curve with exposure bias 2.0 and white point 11.2
  // The curve formula: ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F
  // Applied with exposure bias then divided by whiteScale
  // For [2.0, 1.5, 1.0]: approximately [0.7132, 0.6208, 0.4929]
  expectCloseTo([0.7132, 0.6208, 0.4929], result);
});

test("tonemapUncharted23", async () => {
  const src = `
     import lygia::color::tonemap::uncharted2::tonemapUncharted23;

     @compute @workgroup_size(1)
     fn foo() {
       let hdr = vec3f(2.0, 1.5, 1.0);
       let result = tonemapUncharted23(hdr);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Uncharted2 applies John Hable's curve to vec4(v, W) then divides xyz by w
  // This normalizes by white point W=11.2 in the same curve calculation
  // For [2.0, 1.5, 1.0]: approximately [0.4929, 0.4086, 0.3043]
  expectCloseTo([0.4929, 0.4086, 0.3043], result);
});

test("tonemapUnreal3", async () => {
  const src = `
     import lygia::color::tonemap::unreal::tonemapUnreal3;

     @compute @workgroup_size(1)
     fn foo() {
       let hdr = vec3f(2.0, 1.5, 1.0);
       let result = tonemapUnreal3(hdr);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Unreal tonemap: x / (x + 0.155) * 1.019
  // R: 2.0 / (2.0 + 0.155) * 1.019 = 2.0 / 2.155 * 1.019 ≈ 0.9457
  // G: 1.5 / (1.5 + 0.155) * 1.019 = 1.5 / 1.655 * 1.019 ≈ 0.9236
  // B: 1.0 / (1.0 + 0.155) * 1.019 = 1.0 / 1.155 * 1.019 ≈ 0.8823
  expectCloseTo([0.9457, 0.9236, 0.8823], result);
});

test("tonemapDebug4", async () => {
  const src = `
     import lygia::color::tonemap::debug::tonemapDebug4;

     @compute @workgroup_size(1)
     fn foo() {
       let hdr = vec4f(1.5, 1.0, 0.5, 0.7);
       let result = tonemapDebug4(hdr);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Debug tonemap on RGB, alpha preserved
  // Same as tonemapDebug3: [0.5718, 0.9076, 0.0], alpha = 0.7
  expectCloseTo([0.5718, 0.9076, 0.0, 0.7], result);
});

test("tonemapFilmic4", async () => {
  const src = `
     import lygia::color::tonemap::filmic::tonemapFilmic4;

     @compute @workgroup_size(1)
     fn foo() {
       let hdr = vec4f(2.0, 1.5, 1.0, 0.8);
       let result = tonemapFilmic4(hdr);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Filmic tonemap on RGB + alpha preserved
  // Same as tonemapFilmic3: [0.9128, 0.8874, 0.8412], alpha = 0.8
  expectCloseTo([0.9128, 0.8874, 0.8412, 0.8], result);
});

test("tonemapLinear4", async () => {
  const src = `
     import lygia::color::tonemap::linear::tonemapLinear4;

     @compute @workgroup_size(1)
     fn foo() {
       let hdr = vec4f(2.0, 1.5, 1.0, 0.5);
       let result = tonemapLinear4(hdr);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Linear tonemap is identity - no modification, alpha included
  expectCloseTo([2.0, 1.5, 1.0, 0.5], result);
});

test("tonemapReinhard4", async () => {
  const src = `
     import lygia::color::tonemap::reinhard::tonemapReinhard4;

     @compute @workgroup_size(1)
     fn foo() {
       let hdr = vec4f(2.0, 1.5, 1.0, 0.6);
       let result = tonemapReinhard4(hdr);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Reinhard tonemap on RGB + alpha preserved
  // Same as tonemapReinhard3: [0.7782, 0.5836, 0.3891], alpha = 0.6
  expectCloseTo([0.7782, 0.5836, 0.3891, 0.6], result);
});

test("tonemapReinhardJodie4", async () => {
  const src = `
     import lygia::color::tonemap::reinhardJodie::tonemapReinhardJodie4;

     @compute @workgroup_size(1)
     fn foo() {
       let hdr = vec4f(2.0, 1.5, 1.0, 0.75);
       let result = tonemapReinhardJodie4(hdr);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Reinhard-Jodie tonemap on RGB + alpha preserved
  // Same as tonemapReinhardJodie3: [0.7038, 0.5935, 0.4445], alpha = 0.75
  expectCloseTo([0.7038, 0.5935, 0.4445, 0.75], result);
});

test("tonemapUncharted4", async () => {
  const src = `
     import lygia::color::tonemap::uncharted::tonemapUncharted4;

     @compute @workgroup_size(1)
     fn foo() {
       let hdr = vec4f(2.0, 1.5, 1.0, 0.9);
       let result = tonemapUncharted4(hdr);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Uncharted tonemap on RGB + alpha preserved
  // Same as tonemapUncharted3: [0.7132, 0.6208, 0.4929], alpha = 0.9
  expectCloseTo([0.7132, 0.6208, 0.4929, 0.9], result);
});

test("uncharted2Tonemap", async () => {
  const src = `
     import lygia::color::tonemap::uncharted::uncharted2Tonemap;

     @compute @workgroup_size(1)
     fn foo() {
       let hdr = vec3f(2.0, 1.5, 1.0);
       let result = uncharted2Tonemap(hdr);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Helper function: ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
  // A=0.15, B=0.50, C=0.10, D=0.20, E=0.02, F=0.30
  // This is the raw curve without exposure bias or white point normalization
  // For [2.0, 1.5, 1.0]: [0.3574, 0.2963, 0.2207]
  expectCloseTo([0.3574, 0.2963, 0.2207], result);
});

test("tonemapUncharted24", async () => {
  const src = `
     import lygia::color::tonemap::uncharted2::tonemapUncharted24;

     @compute @workgroup_size(1)
     fn foo() {
       let hdr = vec4f(2.0, 1.5, 1.0, 0.85);
       let result = tonemapUncharted24(hdr);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Uncharted2 tonemap on RGB + alpha preserved
  // Same as tonemapUncharted23: [0.4929, 0.4086, 0.3043], alpha = 0.85
  expectCloseTo([0.4929, 0.4086, 0.3043, 0.85], result);
});

test("tonemapUnreal4", async () => {
  const src = `
     import lygia::color::tonemap::unreal::tonemapUnreal4;

     @compute @workgroup_size(1)
     fn foo() {
       let hdr = vec4f(2.0, 1.5, 1.0, 0.65);
       let result = tonemapUnreal4(hdr);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Unreal tonemap on RGB + alpha preserved
  // Same as tonemapUnreal3: [0.9457, 0.9236, 0.8823], alpha = 0.65
  expectCloseTo([0.9457, 0.9236, 0.8823, 0.65], result);
});
