// Test helper for dither visual tests
// Provides texture input and quantization to show banding

@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;

/// Sample input texture without quantization (original values)
fn sampleOriginal(pos: vec4f) -> f32 {
  let resolution = vec2f(textureDimensions(inputTex));
  let uv = pos.xy / resolution;
  return textureSample(inputTex, smp, uv).r;
}

/// Sample input texture as vec3 without quantization
fn sampleOriginal3(pos: vec4f) -> vec3f {
  let resolution = vec2f(textureDimensions(inputTex));
  let uv = pos.xy / resolution;
  return textureSample(inputTex, smp, uv).rgb;
}

/// Sample input texture as vec4 without quantization
fn sampleOriginal4(pos: vec4f) -> vec4f {
  let resolution = vec2f(textureDimensions(inputTex));
  let uv = pos.xy / resolution;
  return textureSample(inputTex, smp, uv);
}

/// Sample input texture and quantize to create visible banding
/// Quantization shows the problem that dithering solves
fn sampleQuantized(pos: vec4f, quantizationLevels: i32) -> f32 {
  let value = sampleOriginal(pos);
  let levels = f32(quantizationLevels);
  return floor(value * levels) / levels;
}

/// Sample input texture as vec3 and quantize
fn sampleQuantized3(pos: vec4f, quantizationLevels: i32) -> vec3f {
  let value = sampleOriginal3(pos);
  let levels = f32(quantizationLevels);
  return floor(value * levels) / levels;
}

/// Sample input texture as vec4 and quantize (preserves alpha)
fn sampleQuantized4(pos: vec4f, quantizationLevels: i32) -> vec4f {
  let value = sampleOriginal4(pos);
  let levels = f32(quantizationLevels);
  let quantized = floor(value.rgb * levels) / levels;
  return vec4f(quantized, value.a);
}
