import wgsl_test::{expect, expectNear};
import lygia::generative::worley::{worley2, worley22, worley3, worley32};

@test fn testWorley2() {
  let p1 = vec2f(1.0, 2.0);
  let p2 = vec2f(1.0, 2.0); // Same point
  let p3 = vec2f(1.5, 2.5); // Point in same cell

  let w1 = worley2(p1);
  let w2 = worley2(p2);
  let w3 = worley2(p3);

  // Determinism: same input produces same output
  expectNear(w1, w2);
  // Different positions have different distances
  expect(abs(w1 - w3) > 0.01);
  // Regression: exact output value
  expectNear(w1, 0.7471);
}

@test fn testWorley22() {
  let p1 = vec2f(1.0, 2.0);
  let p2 = vec2f(1.0, 2.0); // Same point

  let w1 = worley22(p1);
  let w2 = worley22(p2);

  // Determinism: same input produces same output
  expectNear(w1.x, w2.x);
  expectNear(w1.y, w2.y);
  // Worley noise returns distances (F1, F2) in reasonable range
  expect(w1.x >= 0.0);
  expect(w1.x <= 1.5);
  expect(w1.y >= 0.0);
  expect(w1.y <= 1.5);
  // F1 should be less than or equal to F2
  expect(w1.x <= w1.y + 0.001);
  // Regression: exact output value
  expectNear(w1.x, 0.2529);
}

@test fn testWorley3() {
  let p1 = vec3f(1.0, 2.0, 3.0);
  let p2 = vec3f(1.0, 2.0, 3.0); // Same point
  let p3 = vec3f(4.0, 5.0, 6.0); // Different point

  let w1 = worley3(p1);
  let w2 = worley3(p2);
  let w3 = worley3(p3);

  // Determinism: same input produces same output
  expectNear(w1, w2);
  // Different inputs produce different outputs
  expect(abs(w1 - w3) > 0.01);
  // Regression: exact output value
  expectNear(w1, 0.3876);
}

@test fn testWorley32() {
  let p1 = vec3f(1.0, 2.0, 3.0);
  let p2 = vec3f(1.0, 2.0, 3.0); // Same point

  let w1 = worley32(p1);
  let w2 = worley32(p2);

  // Determinism: same input produces same output
  expectNear(w1.x, w2.x);
  expectNear(w1.y, w2.y);
  // Worley noise returns distances (F1, F2) in reasonable range
  expect(w1.x >= 0.0);
  expect(w1.x <= 2.0);
  expect(w1.y >= 0.0);
  expect(w1.y <= 2.0);
  // F1 should be less than or equal to F2
  expect(w1.x <= w1.y + 0.001);
  // Regression: exact output value
  expectNear(w1.x, 0.6124);
}
