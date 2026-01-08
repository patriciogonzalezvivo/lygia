// @toy
import lygia::math::consts::TAU;
import lygia::math::aastep::aastep;
import lygia::math::aafract::aafract;
import lygia::space::cart2polar::cart2polar2;
import lygia::color::space::linear2gamma::linear2gamma;

@group(0) @binding(0) var<uniform> uniforms: test::Uniforms;

// Visual test comparing anti-aliased vs regular functions on a complex spiral pattern
// Based on https://www.shadertoy.com/view/wtjGzt by FabriceNeyret2
//
// WHAT TO LOOK FOR IN THE IMAGE:
// - Top-left: fract() creates continuous gradient with moire/aliasing in spiral center
// - Top-right: aafract() creates smooth gradient throughout, no aliasing
// - Bottom-left: step(fract()) creates hard black/white bands with moire/aliasing
// - Bottom-right: aastep(aafract()) creates smooth black/white transitions, no aliasing
//
// The spiral pattern creates increasing frequency toward the center, making
// aliasing artifacts very visible. The AA versions (right side) should be
// smooth throughout, while the regular versions (left) show visual noise.

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let pixel = 1.0 / uniforms.resolution;
  let st = pos.xy * pixel;
  // Transform to [-1,1] range
  // WGSL: Y=0 at top, Y=1 at bottom
  // GLSL: Y=0 at bottom, Y=1 at top
  // Flip Y to match GLSL's coordinate system
  let uv = vec2f(st.x * 2.0 - 1.0, 1.0 - st.y * 2.0);

  let p = cart2polar2(uv);

  // Create spiral field - frequency increases toward center
  let spiralValue = 20.0 * p.x / TAU + 10.0 / max(p.y, 0.001) + uniforms.time;

  // Top half: compare fract vs aafract
  let fractResult = select(fract(spiralValue), aafract(spiralValue), uv.x >= 0.0);

  // Bottom half: apply step/aastep to the fracted value
  let stepResult = select(step(0.5, fractResult), aastep(0.5, fractResult), uv.x >= 0.0);

  // Select top vs bottom
  let v = select(stepResult, fractResult, uv.y > 0.0);

  // Convert to sRGB
  let color = linear2gamma(v);

  return vec4f(vec3f(color), 1.0);
}
