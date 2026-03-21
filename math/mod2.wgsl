/*
contributors: Mercury
description: |
    Two dimensional modulus, returns the remainder of a division of two vectors.
    Found at in Mercury's library https://mercury.sexy/hg_sdf/
use: <vec2> mod2(<vec2> x, <vec2> size)
*/

fn mod22(p: vec2f, s: f32) -> vec2f {
    let c = floor((p + s*0.5)/s);
    p = mod(p + s*0.5,s) - s*0.5;
    return c;
}

fn mod22a(p: vec2f, s: vec2f) -> vec2f {
    let c = floor((p + s*0.5)/s);
    p = mod(p + s*0.5,s) - s*0.5;
    return c;
}
