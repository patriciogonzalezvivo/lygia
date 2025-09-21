import package::math::map::map;

/*
contributors: ["dahart", "Fabrice NEYRET"]
description: |
    Is similar to floor() but has a 2-pixel wide gradient between clamped steps 
    to allow the edges in the result to be anti-aliased.
    Based on examples https://www.shadertoy.com/view/4l2BRD and https://www.shadertoy.com/view/3tSGWy
*/

fn aafloor(x: f32) -> f32 {
    let afwidth = 2.0 * fwidth(x);
    let fx = fract(x);
    let idx = 1. - afwidth;
    return select(map(fx, idx, 1., x-fx, x), x - fx, fx < idx);
}

fn aafloor2(x: vec2f) -> vec2f { return vec2f(aafloor(x.x), aafloor(x.y)); }
