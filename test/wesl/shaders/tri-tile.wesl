import lygia::space::triTile::triTile;
import lygia::generative::random::random;
import test::Uniforms;

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let st = pos.xy / uniforms.resolution;

  // Scale up to show multiple triangles
  let scaled = st * 8.0;

  // Apply triangular tiling
  let tile = triTile(scaled);

  // tile.xy = within-tile coords [0,1]
  // tile.zw = tile index

  // Color each triangle differently based on tile index
  let tileId = tile.z + tile.w * 100.0;
  let hue = random(tileId);

  // Create pattern within each triangle
  let dist = length(tile.xy - 0.5);
  let brightness = 0.4 + 0.4 * (1.0 - dist);

  let color = vec3f(hue, brightness, brightness * 0.8);

  return vec4f(color, 1.0);
}
