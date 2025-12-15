// @toy
import lygia::space::windmillTile::windmillTile2;
import test::Uniforms;

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let st = pos.xy / uniforms.resolution;

  // Scale up to show multiple tiles
  let scaled = st * 4.0;

  // Apply windmill tiling (rotates adjacent tiles)
  let tile = windmillTile2(scaled);

  // Create pattern that shows rotation effect
  // Use within-tile coords which are rotated per tile
  let gradient = tile.xy;

  // Create directional gradient that will show rotation
  let color = vec3f(gradient.x, gradient.y * 0.5, gradient.y);

  return vec4f(color, 1.0);
}
