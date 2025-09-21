import package::math::map::map;

/*
contributors: dahart (https://www.shadertoy.com/user/dahart)
description: |
    Anti-aliasing fract function. It clamp except for a 2-pixel wide gradient along the edge
    Based on this example https://www.shadertoy.com/view/4l2BRD
*/

fn aafract(x: f32) -> f32 {
    let afwidth = 2.0 * fwidth(x);
    let fx = fract(x);
    let idx = 1.0 - afwidth;
    return select(map(fx, idx, 1., fx, 0.), fx, fx < idx);
}

fn aafract2(v: vec2f) -> vec2f { return vec2(aafract(v.x), aafract(v.y)); }
