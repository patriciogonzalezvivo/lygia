// @toy
import lygia::draw::stroke::{stroke, strokeEdge};
import lygia::math::consts::PI;

@group(0) @binding(0) var<uniform> uniforms: test::Uniforms;

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let st = pos.xy / uniforms.resolution;

  // Create a grid of circles to test stroke with various widths
  let grid_size = 4.0;
  let cell = fract(st * grid_size);
  let cell_id = floor(st * grid_size);

  // Distance from center of each cell
  let dist = length(cell - 0.5);

  // Vary stroke width based on grid position
  let row = cell_id.y;
  let col = cell_id.x;

  // Compute both stroke variations (must be in uniform control flow)
  // Top half: stroke function with varying widths
  let width1 = 0.05 + (col / grid_size) * 0.15; // Width: 0.05 to 0.2
  let size1 = 0.3; // Fixed radius
  let result1 = stroke(dist, size1, width1);

  // Bottom half: strokeEdge with varying edge smoothness
  let width2 = 0.1;
  let size2 = 0.3;
  let edge = 0.001 + (col / grid_size) * 0.05; // Edge: 0.001 to 0.051
  let result2 = strokeEdge(dist, size2, width2, edge);

  // Select based on row (using mix to keep uniform control flow)
  let blend_factor = smoothstep(1.9, 2.1, row);
  let result = mix(result1, result2, blend_factor);

  return vec4f(result, result, result, 1.0);
}
