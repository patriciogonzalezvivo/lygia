import wgsl_test::{expect, expectNear, expectNearVec4, expectWithin, expectWithinVec4};
import lygia::math::cubic::cubic;
import lygia::math::quartic::quartic;
import lygia::math::quintic::quintic;
import lygia::math::invCubic::invCubic;
import lygia::math::invQuartic::invQuartic;
import lygia::math::gain::gain;
import lygia::math::parabola::parabola;
import lygia::math::gaussian::gaussian;

@test fn testCubic() {
  // cubic(0) = 0, cubic(0.5) = 0.5, cubic(1) = 1
  expectNearVec4(
    vec4f(cubic(0.0), cubic(0.5), cubic(1.0), 0.0),
    vec4f(0.0, 0.5, 1.0, 0.0)
  );
}

@test fn testQuartic() {
  // quartic(v) = v*v*(2-v*v), quartic(0.5) = 0.25 * 1.75 = 0.4375
  expectNearVec4(
    vec4f(quartic(0.0), quartic(0.5), quartic(1.0), 0.0),
    vec4f(0.0, 0.4375, 1.0, 0.0)
  );
}

@test fn testQuintic() {
  // quintic(0) = 0, quintic(0.5) = 0.5, quintic(1) = 1
  expectNearVec4(
    vec4f(quintic(0.0), quintic(0.5), quintic(1.0), 0.0),
    vec4f(0.0, 0.5, 1.0, 0.0)
  );
}

@test fn testInvCubic() {
  // invCubic should be the inverse of cubic
  let x = 0.3;
  let y = cubic(x);
  let xRecovered = invCubic(y);
  expectWithin(xRecovered, x, 0.01, 1e-6);
}

@test fn testInvQuartic() {
  // invQuartic should be the inverse of quartic
  let x = 0.7;
  let y = quartic(x);
  let xRecovered = invQuartic(y);
  expectWithin(xRecovered, x, 0.01, 1e-6);
}

@test fn testGain() {
  // gain(0.5, k) = 0.5 always (midpoint invariant)
  expectWithin(gain(0.5, 2.0), 0.5, 0.01, 1e-6);
  // Exact values for regression detection
  expectNearVec4(
    vec4f(gain(0.5, 2.0), gain(0.25, 2.0), gain(0.75, 2.0), 0.0),
    vec4f(0.5, 0.125, 0.875, 0.0)
  );
}

@test fn testParabolaK1() {
  // k=1.0: edges at 0, peak at 0.5, symmetric at 0.25/0.75
  expectNearVec4(
    vec4f(parabola(0.0, 1.0), parabola(0.25, 1.0), parabola(0.5, 1.0), parabola(0.75, 1.0)),
    vec4f(0.0, 0.75, 1.0, 0.75)
  );
}

@test fn testParabolaK2() {
  // k=2.0: sharper peak - values at 0.25/0.75 drop to 0.5625 (0.75^2)
  let p25 = parabola(0.25, 2.0);
  let p50 = parabola(0.5, 2.0);
  let p75 = parabola(0.75, 2.0);
  expectWithin(p25, 0.5625, 0.01, 1e-6);
  expectWithin(p50, 1.0, 0.01, 1e-6);
  // k=2 gives lower values than k=1 at 0.25
  expect(p25 < parabola(0.25, 1.0));
  // Exact values for regression
  expectNearVec4(
    vec4f(p25, p50, p75, 0.0),
    vec4f(0.5625, 1.0, 0.5625, 0.0)
  );
}

@test fn testGaussian() {
  // gaussian(0, sigma) should be 1.0
  // gaussian(sigma, sigma) should be exp(-0.5) ~ 0.6065
  expectNear(gaussian(0.0, 1.0), 1.0);
  expectNear(gaussian(1.0, 1.0), 0.6065);
}
