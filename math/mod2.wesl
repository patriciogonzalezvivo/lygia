/*
contributors: Mercury
description: |
    Two dimensional modulus, returns the remainder of a division of two vectors.
    Found at in Mercury's library https://mercury.sexy/hg_sdf/
use: mod2(p: ptr<function, vec2f>, s: f32) -> vec2f | mod22(p: ptr<function, vec2f>, s: vec2f) -> vec2f
license:
    - Mercury
*/

fn mod2(p: ptr<function, vec2f>, s: f32) -> vec2f {
    let c = floor((*p + s * 0.5) / s);
    *p = ((*p + s * 0.5) % s) - s * 0.5;
    return c;
}

fn mod22(p: ptr<function, vec2f>, s: vec2f) -> vec2f {
    let c = floor((*p + s * 0.5) / s);
    *p = ((*p + s * 0.5) % s) - s * 0.5;
    return c;
}
