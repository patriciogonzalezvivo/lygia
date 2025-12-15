// @toy
import lygia::space::mirrorTile::mirrorTile2;
import test::Uniforms;

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let st = pos.xy / uniforms.resolution;

  // Scale up to show multiple tiles
  let scaled = st * 4.0;

  // Apply mirror tiling
  let tile = mirrorTile2(scaled);

  // Create gradient pattern that will mirror across tiles
  // Use within-tile coords which are mirrored
  let gradient = tile.xy;

  // Create a pattern that shows the mirroring effect
  let color = vec3f(gradient.x, gradient.y, 0.5);

  return vec4f(color, 1.0);
}
