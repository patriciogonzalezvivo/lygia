import lygia::math::nyquist::nyquist;

/*
contributors: dahart (https://www.shadertoy.com/user/dahart)
description: |
    Anti-aliasing fract function. It clamp except for a 2-pixel wide gradient along the edge
    Based on this example https://www.shadertoy.com/view/4l2BRD
*/

fn aafract(x: f32) -> f32 {
    // Use Euclidean length (L2 norm) to match GLSL, not fwidth (L1 norm)
    let afwidth = 2.0 * length(vec2f(dpdx(x), dpdy(x)));
    let fx = fract(x);
    let idx = 1.0 - afwidth;
    let v = select((1.0-fx)/afwidth, fx/idx, fx < idx);
    return nyquist(v, afwidth);
}

fn aafract2(v: vec2f) -> vec2f { return vec2(aafract(v.x), aafract(v.y)); }
