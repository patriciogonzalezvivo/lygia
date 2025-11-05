import { expect, test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("diffuseOrenNayar", async () => {
  const src = `
    import lygia::lighting::diffuse::orenNayar::diffuseOrenNayar;

    @compute @workgroup_size(1)
    fn foo() {
      // Test 1: Roughness=0 should equal Lambert diffuse (NoL)
      // L at 45° to surface normal
      let L = normalize(vec3f(1.0, 1.0, 1.0));
      let N = vec3f(0.0, 0.0, 1.0);
      let V = vec3f(0.0, 0.0, 1.0);
      let NoV = dot(N, V);  // = 1.0
      let NoL = dot(N, L);  // = 1/sqrt(3) ≈ 0.5774

      // At roughness=0: sigma2=0, A=1.0, B=0.0
      // Result = NoL * (1.0 + 0) = NoL ≈ 0.5774
      let smoothResult = diffuseOrenNayar(L, N, V, NoV, NoL, 0.0);

      // Test 2: Roughness=1.0 with perpendicular view
      // sigma2 = 1.0
      // A = 1.0 + 1.0 * (1.0/(1.0+0.13) + 0.5/(1.0+0.33))
      //   = 1.0 + 1.0 * (0.88496 + 0.37594)
      //   = 1.0 + 1.26090 = 2.26090
      // LoV = dot(L,V) = NoL = 1/sqrt(3)
      // s = LoV - NoL*NoV = 1/sqrt(3) - 1/sqrt(3)*1.0 = 0
      // t = mix(1.0, max(NoL,NoV), step(0.0,s)) = mix(1.0, 1.0, 0.0) = 1.0
      // B = 0.45 * 1.0 / (1.0 + 0.09) = 0.41284
      // Result = NoL * (A + B * 0 / 1.0) = NoL * 2.26090 ≈ 1.30536
      let roughResult = diffuseOrenNayar(L, N, V, NoV, NoL, 1.0);

      // Test 3: Retroreflection (V = L, roughness=1.0)
      // NoV = NoL = 1/sqrt(3)
      // LoV = dot(L,L) = 1.0
      // s = 1.0 - (1/sqrt(3))*(1/sqrt(3)) = 1.0 - 1/3 = 2/3 ≈ 0.6667
      // t = mix(1.0, max(NoL,NoV), step(0.0,s))
      //   = mix(1.0, 1/sqrt(3), 1.0) = 1/sqrt(3) ≈ 0.5774
      // A = 2.2609 (same as test 2)
      // B = 0.4128 (same as test 2)
      // Result = NoL * (A + B * s / t)
      //        = (1/sqrt(3)) * (2.2609 + 0.4128 * (2/3) / (1/sqrt(3)))
      //        = 0.5774 * (2.2609 + 0.4128 * 1.1547)
      //        = 0.5774 * (2.2609 + 0.4767)
      //        = 0.5774 * 2.7376
      //        ≈ 1.5806
      let V2 = L;
      let NoV2 = dot(N, V2);
      let retroResult = diffuseOrenNayar(L, N, V2, NoV2, NoL, 1.0);

      test::results[0] = vec4f(smoothResult, roughResult, retroResult, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // Expected values calculated manually from the Oren-Nayar formula:
  const NoL = 1.0 / Math.sqrt(3); // ≈ 0.57735

  // Test 1: roughness=0 → result = NoL
  // Using 1e-5: GPU and CPU compute identical formula, should match to high precision
  expectCloseTo([NoL], [result[0]], 1e-5);

  // Test 2: roughness=1.0, perpendicular view
  // A = 1.0 + 1.0 * (1.0/(1.0+0.13) + 0.5/(1.0+0.33)) ≈ 2.26090
  // s = 0, so Result = NoL * A
  const A = 1.0 + 1.0 * (1.0 / (1.0 + 0.13) + 0.5 / (1.0 + 0.33));
  const expectedRoughResult = NoL * A;
  // Using 1e-5: GPU and CPU compute identical formula, should match to high precision
  expectCloseTo([expectedRoughResult], [result[1]], 1e-5);

  // Test 3: retroreflection (V=L, roughness=1.0)
  // s = 1.0 - 1/3 = 2/3, t = 1/sqrt(3)
  // B = 0.45 * 1.0 / (1.0 + 0.09)
  // Result = NoL * (A + B * s / t)
  const B = (0.45 * 1.0) / (1.0 + 0.09);
  const s = 1.0 - NoL * NoL;
  const t = NoL;
  const expectedRetroResult = NoL * (A + (B * s) / t);
  // Using 1e-5: GPU and CPU compute identical formula, should match to high precision
  expectCloseTo([expectedRetroResult], [result[2]], 1e-5);

  // Verify relationships still hold as sanity check
  expect(result[1]).toBeGreaterThan(result[0]); // Rough > smooth
  expect(result[2]).toBeGreaterThan(result[1]); // Retro > rough
});
