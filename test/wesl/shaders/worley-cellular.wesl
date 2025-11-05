import lygia::generative::worley::worley2;

@group(0) @binding(0) var<uniform> uniforms: test::Uniforms;

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let st = pos.xy / uniforms.resolution;

  // Worley/Voronoi cellular pattern
  let n = worley2(st * 8.0);

  // Worley returns distance to nearest cell, create pattern
  let color = vec3f(n);
  return vec4f(color, 1.0);
}
