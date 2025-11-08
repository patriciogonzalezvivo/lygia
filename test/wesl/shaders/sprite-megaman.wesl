import lygia::space::ratio::ratio;
import lygia::animation::spriteLoop::spriteLoop;

@group(0) @binding(0) var<uniform> uniforms: test::Uniforms;
@group(0) @binding(1) var sprite_tex: texture_2d<f32>;
@group(0) @binding(2) var sprite_samp: sampler;

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  var color = vec4f(0.0, 0.0, 0.0, 1.0);
  let pixel = 1.0 / uniforms.resolution;
  var st = pos.xy * pixel;
  st = ratio(st, uniforms.resolution);

  let grid = vec2f(10.0, 7.0);

  let time = uniforms.time * 6.0 % 48.0;
  if (time < 6.0) {
    color = spriteLoop(sprite_tex, sprite_samp, st, grid, 0.0, 2.0, time);
  } else if (time < 12.0) {
    color = spriteLoop(sprite_tex, sprite_samp, st, grid, 3.0, 6.0, time);
  } else if (time < 18.0) {
    color = spriteLoop(sprite_tex, sprite_samp, st, grid, 13.0, 16.0, time);
  } else if (time < 24.0) {
    color = spriteLoop(sprite_tex, sprite_samp, st, grid, 23.0, 26.0, time);
  } else if (time < 30.0) {
    color = spriteLoop(sprite_tex, sprite_samp, st, grid, 33.0, 36.0, time);
  } else if (time < 36.0) {
    color = spriteLoop(sprite_tex, sprite_samp, st, grid, 43.0, 46.0, time);
  } else if (time < 42.0) {
    color = spriteLoop(sprite_tex, sprite_samp, st, grid, 50.0, 53.0, time);
  } else {
    color = spriteLoop(sprite_tex, sprite_samp, st, grid, 60.0, 65.0, time);
  }

  return color;
}
