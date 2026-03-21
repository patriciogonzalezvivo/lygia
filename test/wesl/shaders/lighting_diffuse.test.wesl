import wgsl_test::{expect, expectWithin};
import lygia::lighting::diffuse::orenNayar::diffuseOrenNayar;

@test fn testDiffuseOrenNayarSmooth() {
  // Roughness=0 should equal Lambert diffuse (NoL)
  // L at 45 degrees to surface normal
  let L = normalize(vec3f(1.0, 1.0, 1.0));
  let N = vec3f(0.0, 0.0, 1.0);
  let V = vec3f(0.0, 0.0, 1.0);
  let NoV = dot(N, V);  // = 1.0
  let NoL = dot(N, L);  // = 1/sqrt(3) ~ 0.5774

  let smoothResult = diffuseOrenNayar(L, N, V, NoV, NoL, 0.0);
  // At roughness=0: result = NoL ~ 0.57735
  expectWithin(smoothResult, 0.57735, 1e-5, 1e-6);
}

@test fn testDiffuseOrenNayarRough() {
  // Roughness=1.0 with perpendicular view
  let L = normalize(vec3f(1.0, 1.0, 1.0));
  let N = vec3f(0.0, 0.0, 1.0);
  let V = vec3f(0.0, 0.0, 1.0);
  let NoV = dot(N, V);
  let NoL = dot(N, L);

  let roughResult = diffuseOrenNayar(L, N, V, NoV, NoL, 1.0);
  // A = 1.0 + 1.0 * (1.0/(1.0+0.13) + 0.5/(1.0+0.33)) ~ 2.26090
  // s = 0, so Result = NoL * A
  let A = 1.0 + 1.0 * (1.0 / (1.0 + 0.13) + 0.5 / (1.0 + 0.33));
  let expectedRough = NoL * A;
  expectWithin(roughResult, expectedRough, 1e-5, 1e-6);
}

@test fn testDiffuseOrenNayarRetro() {
  // Retroreflection (V = L, roughness=1.0)
  let L = normalize(vec3f(1.0, 1.0, 1.0));
  let N = vec3f(0.0, 0.0, 1.0);
  let NoL = dot(N, L);
  let V2 = L;
  let NoV2 = dot(N, V2);

  let retroResult = diffuseOrenNayar(L, N, V2, NoV2, NoL, 1.0);
  let A = 1.0 + 1.0 * (1.0 / (1.0 + 0.13) + 0.5 / (1.0 + 0.33));
  let B = 0.45 * 1.0 / (1.0 + 0.09);
  let s = 1.0 - NoL * NoL;
  let t = NoL;
  let expectedRetro = NoL * (A + B * s / t);
  expectWithin(retroResult, expectedRetro, 1e-5, 1e-6);
}

@test fn testDiffuseOrenNayarRelationships() {
  // Verify relationships: rough > smooth, retro > rough
  let L = normalize(vec3f(1.0, 1.0, 1.0));
  let N = vec3f(0.0, 0.0, 1.0);
  let V = vec3f(0.0, 0.0, 1.0);
  let NoV = dot(N, V);
  let NoL = dot(N, L);

  let smoothResult = diffuseOrenNayar(L, N, V, NoV, NoL, 0.0);
  let roughResult = diffuseOrenNayar(L, N, V, NoV, NoL, 1.0);

  let V2 = L;
  let NoV2 = dot(N, V2);
  let retroResult = diffuseOrenNayar(L, N, V2, NoV2, NoL, 1.0);

  expect(roughResult > smoothResult);
  expect(retroResult > roughResult);
}
