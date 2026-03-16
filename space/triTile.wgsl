/*
contributors: Mathias Bredholt
description: |
    make some triangular tiles. XY provide coords inside of the tile. ZW provides tile coords
use: <vec4> triTile(<vec2> st [, <float> scale])
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_tiles.frag
*/

fn triTile2(st: vec2f) -> vec4f {
    st *= mat2x2<f32>(1., -1. / 1.7320508, 0., 2. / 1.7320508);
    let f = vec4f(st, -st);
    let i = floor(f);
    f = fract(f);
    return dot(f.xy, f.xy) < dot(f.zw, f.zw)
                ? vec4f(f.xy, vec2f(2., 1.) * i.xy)
                : vec4f(f.zw, -(vec2f(2., 1.) * i.zw + 1.));
}

fn triTile2a(st: vec2f, scale: f32) -> vec4f { return triTile(st * scale); }
