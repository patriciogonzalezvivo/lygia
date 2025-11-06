// Test helper for layer blend mode visual tests
// Provides texture bindings and sampling for blend operations

@group(0) @binding(1) var srcTex: texture_2d<f32>;
@group(0) @binding(2) var dstTex: texture_2d<f32>;
@group(0) @binding(3) var smp: sampler;

struct BlendInputs {
  src: vec4f,
  dst: vec4f,
}

/// Sample src and dst textures at the given position
/// Automatically detects texture resolution
fn blendInputs(pos: vec4f) -> BlendInputs {
  let resolution = vec2f(textureDimensions(srcTex));
  let uv = pos.xy / resolution;
  let src = textureSample(srcTex, smp, uv);
  let dst = textureSample(dstTex, smp, uv);
  return BlendInputs(src, dst);
}
