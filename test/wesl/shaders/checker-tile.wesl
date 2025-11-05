import lygia::space::checkerTile::checkerTile2;
import test::Uniforms;

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let st = pos.xy / uniforms.resolution;

  // Scale up to show multiple checkers
  let scaled = st * 8.0;

  // Apply checker tiling (returns 0 or 1)
  let checker = checkerTile2(scaled);

  // Create black and white checkerboard
  let color = vec3f(checker);

  return vec4f(color, 1.0);
}
