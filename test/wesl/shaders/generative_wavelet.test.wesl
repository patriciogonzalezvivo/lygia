import wgsl_test::{expect, expectNear, expectWithin};
import lygia::generative::noised::{noised2, noised3};
import lygia::generative::wavelet::{wavelet, wavelet2, wavelet3, waveletScaled2, waveletScaled3};

@test fn testNoised2() {
  let h = 0.001;
  let p = vec2f(1.0, 2.0);

  // Get noise and analytical derivatives
  let nd = noised2(p);
  let dx_analytical = nd.y;
  let dy_analytical = nd.z;

  // Compute numerical derivatives via finite differences
  let dx_numerical = (noised2(p + vec2f(h, 0.0)).x - noised2(p - vec2f(h, 0.0)).x) / (2.0 * h);
  let dy_numerical = (noised2(p + vec2f(0.0, h)).x - noised2(p - vec2f(0.0, h)).x) / (2.0 * h);

  // Analytical derivatives should match numerical derivatives (precision=2 -> 0.01 tolerance)
  expectWithin(dx_analytical, dx_numerical, 0.01, 1e-6);
  expectWithin(dy_analytical, dy_numerical, 0.01, 1e-6);
  // Derivatives should be in reasonable range
  expect(abs(dx_analytical) < 5.0);
  expect(abs(dy_analytical) < 5.0);
  // Regression: exact output value
  expectNear(dx_analytical, -0.3648);
}

@test fn testNoised3() {
  let h = 0.001;
  let p = vec3f(1.0, 2.0, 3.0);

  // Get noise and analytical derivatives
  let nd = noised3(p);
  let dx_analytical = nd.y;
  let dy_analytical = nd.z;

  // Compute numerical derivatives via finite differences
  let dx_numerical = (noised3(p + vec3f(h, 0.0, 0.0)).x - noised3(p - vec3f(h, 0.0, 0.0)).x) / (2.0 * h);
  let dy_numerical = (noised3(p + vec3f(0.0, h, 0.0)).x - noised3(p - vec3f(0.0, h, 0.0)).x) / (2.0 * h);

  // Analytical derivatives should match numerical derivatives
  expectWithin(dx_analytical, dx_numerical, 0.01, 1e-6);
  expectWithin(dy_analytical, dy_numerical, 0.01, 1e-6);
  // Derivatives should be in reasonable range
  expect(abs(dx_analytical) < 5.0);
  expect(abs(dy_analytical) < 5.0);
}

@test fn testWavelet2() {
  let p1 = vec2f(1.0, 2.0);
  let p2 = vec2f(1.0, 2.0); // Same point
  let p3 = vec2f(3.0, 4.0); // Different point

  let w1 = wavelet2(p1);
  let w2 = wavelet2(p2);
  let w3 = wavelet2(p3);

  // Determinism: same input produces same output
  expectNear(w1, w2);
  // Different inputs produce different outputs
  expect(abs(w1 - w3) > 0.01);
  // Regression: exact output value
  expectNear(w1, -0.1946);
}

@test fn testWavelet3() {
  // Third component is the phase parameter
  let p = vec2f(1.0, 2.0);
  let phase1 = 0.0;
  let phase2 = 1.0;

  let w1 = wavelet3(vec3f(p, phase1));
  let w2 = wavelet3(vec3f(p, phase2));
  let w3 = wavelet3(vec3f(p, phase1)); // Same as w1

  // Determinism: same position and phase produce same output
  expectNear(w1, w3);
  // Different phases produce different outputs
  expect(abs(w1 - w2) > 0.01);
  // Regression: exact output value
  expectNear(w1, -0.1946);
}

@test fn testWaveletScaled2() {
  let p = vec2f(1.0, 2.0);
  let phase = 0.5;

  let w1 = waveletScaled2(p, phase);
  let w2 = waveletScaled2(p, phase); // Same
  let w3 = waveletScaled2(p * 2.0, phase); // Different position

  // Determinism: same input produces same output
  expectNear(w1, w2);
  // Scaling position changes output
  expect(abs(w1 - w3) > 0.01);
  // Regression: exact output value
  expectNear(w1, -0.1114);
}

@test fn testWaveletScaled3() {
  let p = vec2f(1.0, 2.0);
  let phase = 0.5;
  let scale1 = 1.0;
  let scale2 = 2.0;

  let w1 = waveletScaled3(vec3f(p, phase), scale1);
  let w2 = waveletScaled3(vec3f(p, phase), scale1); // Same
  let w3 = waveletScaled3(vec3f(p, phase), scale2); // Different scale

  // Determinism: same input produces same output
  expectNear(w1, w2);
  // Different scales produce different outputs
  expect(abs(w1 - w3) > 0.01);
  // Regression: exact output value
  expectNear(w1, 0.0945);
}

@test fn testWaveletBase() {
  let p = vec2f(1.0, 2.0);
  let phase = 0.5;
  let scale = 1.5;

  let w1 = wavelet(p, phase, scale);
  let w2 = wavelet(p, phase, scale); // Same
  let w3 = wavelet(p, 1.0, scale); // Different phase
  let w4 = wavelet(p, phase, 2.0); // Different scale

  // Determinism: same input produces same output
  expectNear(w1, w2);
  // Different phase produces different output
  expect(abs(w1 - w3) > 0.01);
  // Different scale produces different output
  expect(abs(w1 - w4) > 0.01);
  // Regression: exact output value
  expectNear(w1, 0.1884);
}
