import lygia::generative::pnoise::pnoise2;

@group(0) @binding(0) var<uniform> uniforms: test::Uniforms;

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let st = pos.xy / uniforms.resolution;

  // Periodic noise with 5-octave FBM, period = 1.0 for seamless tiling
  var value = 0.0;
  var amplitude = 0.5;
  var frequency = 5.0;

  for (var i = 0; i < 5; i++) {
    value += amplitude * pnoise2(st * frequency, vec2f(frequency));
    frequency *= 2.0;
    amplitude *= 0.5;
  }

  // Map [-1, 1] to [0, 1] for visualization
  let color = vec3f(value * 0.5 + 0.5);
  return vec4f(color, 1.0);
}
