import lygia::space::kaleidoscope::kaleidoscope_seg;
import test::Uniforms;

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var input_samp: sampler;

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let st = pos.xy / uniforms.resolution;

  // Create 2x2 grid with different segment counts
  // Top-left: 4 segments, Top-right: 6 segments
  // Bottom-left: 8 segments, Bottom-right: 12 segments

  // Map each quadrant to [0, 1] space
  let quadrantX = fract(st.x * 2.0);
  let quadrantY = fract(st.y * 2.0);
  let localUV = vec2f(quadrantX, quadrantY);

  // Determine which quadrant and segment count
  let isLeft = st.x < 0.5;
  let isTop = st.y < 0.5;

  var segmentCount: f32;
  if (isTop && isLeft) {
    segmentCount = 4.0;  // Top-left
  } else if (isTop && !isLeft) {
    segmentCount = 6.0;  // Top-right
  } else if (!isTop && isLeft) {
    segmentCount = 8.0;  // Bottom-left
  } else {
    segmentCount = 12.0; // Bottom-right
  }

  // Apply kaleidoscope transformation
  let transformedUV = kaleidoscope_seg(localUV, segmentCount);

  // Sample texture
  let color = textureSample(input_tex, input_samp, transformedUV);

  return color;
}
