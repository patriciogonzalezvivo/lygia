// @toy
import lygia::space::brickTile::brickTile2;
import lygia::generative::random::random;
import test::Uniforms;

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let st = pos.xy / uniforms.resolution;

  // Scale up to show multiple bricks
  let scaled = st * 8.0;

  // Apply brick tiling
  let tile = brickTile2(scaled);

  // tile.xy = within-tile coords [0,1]
  // tile.zw = tile index

  // Create brick pattern with mortar lines
  let withinBrick = tile.xy;
  let mortar = 0.05;
  let isMortar = withinBrick.x < mortar || withinBrick.x > (1.0 - mortar) ||
                 withinBrick.y < mortar || withinBrick.y > (1.0 - mortar);

  // Color each brick differently based on tile index
  let tileId = tile.z + tile.w * 100.0;
  let brickColor = 0.4 + 0.4 * random(tileId);

  let color = select(vec3f(brickColor, brickColor * 0.7, brickColor * 0.5), vec3f(0.3), isMortar);

  return vec4f(color, 1.0);
}
