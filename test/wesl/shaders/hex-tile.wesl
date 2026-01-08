// @toy
import lygia::space::hexTile::hexTile;
import lygia::generative::random::random;
import test::Uniforms;

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let st = pos.xy / uniforms.resolution;

  // Scale up to show multiple hexagons
  let scaled = st * 8.0;

  // Apply hexagonal tiling
  let tile = hexTile(scaled);

  // tile.xy = within-tile coords [0,1]
  // tile.zw = tile index

  // Create pattern using within-tile coords
  let dist = length(tile.xy - 0.5);

  // Color each hex differently based on tile index
  let tileId = tile.z + tile.w * 100.0;
  let hue = random(tileId);

  // Combine distance pattern with per-hex color
  let brightness = 0.5 + 0.5 * (1.0 - smoothstep(0.3, 0.5, dist));
  let color = vec3f(hue, brightness, brightness);

  return vec4f(color, 1.0);
}
