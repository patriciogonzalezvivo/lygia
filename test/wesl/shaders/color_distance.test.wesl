import wgsl_test::{expect, expectNear, expectNearVec3, expectWithin};
import lygia::color::distance::{colorDistance, colorDistance4, colorDistanceLAB, colorDistanceLABCIE94, colorDistanceOKLAB, colorDistanceYCbCr, colorDistanceYPbPr, colorDistanceYUV};

@test fn testColorDistance() {
  let red = vec3f(1.0, 0.0, 0.0);
  let blue = vec3f(0.0, 0.0, 1.0);
  let distance = colorDistance(red, blue);
  // Default is CIE94 distance between red and blue (0-100 scale)
  expectNear(distance, 71.0491);
}

@test fn testColorDistance4() {
  let red = vec4f(1.0, 0.0, 0.0, 0.8);
  let blue = vec4f(0.0, 0.0, 1.0, 0.6);
  let distance = colorDistance4(red, blue);
  // Alpha is ignored, should be same as colorDistance (0-100 scale)
  expectNear(distance, 71.0491);
}

@test fn testColorDistanceLAB() {
  let color1 = vec3f(1.0, 0.0, 0.0); // Red
  let color2 = vec3f(0.0, 0.0, 1.0); // Blue
  let result = colorDistanceLAB(color1, color2);
  // LAB Euclidean distance between red and blue in LAB color space (0-100 scale)
  expectNear(result, 176.314);
}

@test fn testColorDistanceLABCIE94() {
  let green = vec3f(0.0, 1.0, 0.0);
  let yellow = vec3f(1.0, 1.0, 0.0);
  let distance = colorDistanceLABCIE94(green, yellow);
  // CIE94 distance between green and yellow (0-100 scale)
  expectNear(distance, 10.0626);
}

@test fn testColorDistanceOKLAB() {
  let red = vec3f(1.0, 0.0, 0.0);
  let orange = vec3f(1.0, 0.5, 0.0);
  let blue = vec3f(0.0, 0.0, 1.0);

  let redToOrange = colorDistanceOKLAB(red, orange);
  let redToBlue = colorDistanceOKLAB(red, blue);

  // Orange is perceptually closer to red than blue is
  expect(redToOrange < redToBlue);

  // Red-orange should be relatively small (similar hues)
  expect(redToOrange > 0.05);
  expect(redToOrange < 0.3);

  // Red-blue should be larger (opposite hues)
  expect(redToBlue > 0.3);

  // Regression check - exact OKLAB distance values
  expectNear(redToOrange, 0.2917);
  expectNear(redToBlue, 0.5371);
}

@test fn testColorDistanceYCbCr() {
  // Test 1: Different chrominance (red vs blue)
  let red = vec3f(0.8, 0.2, 0.2);
  let blue = vec3f(0.2, 0.2, 0.8);
  let chromaDist = colorDistanceYCbCr(red, blue);

  // Test 2: Same chrominance, different luma (should be ~0)
  let darkGray = vec3f(0.3, 0.3, 0.3);
  let lightGray = vec3f(0.7, 0.7, 0.7);
  let lumaDist = colorDistanceYCbCr(darkGray, lightGray);

  // Different chrominance should produce measurable distance
  expect(chromaDist > 0.5);
  expect(chromaDist < 1.5);

  // Same chrominance (grays) should have near-zero distance
  expectWithin(lumaDist, 0.0, 1e-4, 1e-4);

  // Regression check - exact YCbCr chroma distance
  expectNear(chromaDist, 0.5316);
}

@test fn testColorDistanceYPbPr() {
  // Complementary colors (magenta vs cyan)
  let magenta = vec3f(1.0, 0.0, 1.0);
  let cyan = vec3f(0.0, 1.0, 1.0);
  let complementaryDist = colorDistanceYPbPr(magenta, cyan);

  // Similar colors (cyan vs blue)
  let blue = vec3f(0.0, 0.0, 1.0);
  let similarDist = colorDistanceYPbPr(cyan, blue);

  // Test symmetry
  let dist1 = colorDistanceYPbPr(magenta, cyan);
  let dist2 = colorDistanceYPbPr(cyan, magenta);

  // Complementary colors should have larger distance than similar colors
  expect(complementaryDist > similarDist);

  // Distance should be symmetric
  expectNear(dist1, dist2);

  // Complementary colors should have significant distance
  expect(complementaryDist > 0.5);

  // Regression check - exact YPbPr distance values
  expectNear(complementaryDist, 0.9919);
  expectNear(similarDist, 0.5957);
}

@test fn testColorDistanceYUV() {
  let white = vec3f(1.0, 1.0, 1.0);
  let gray = vec3f(0.5, 0.5, 0.5);
  let distance = colorDistanceYUV(white, gray);
  // YUV distance between white and gray (mainly Y difference)
  expectNear(distance, 0.5);
}
