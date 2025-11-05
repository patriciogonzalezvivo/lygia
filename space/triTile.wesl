/*
contributors: Mathias Bredholt
description: |
    make some triangular tiles. XY provide coords inside of the tile. ZW provides tile coords
use: <vec4> triTile(<vec2> st [, <float> scale])
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_tiles.frag
*/

fn triTile(st_in: vec2f) -> vec4f {
    var st = st_in * mat2x2f(1.0, -1.0 / 1.7320508, 0.0, 2.0 / 1.7320508);
    let f_calc = vec4f(st, -st);
    let i = floor(f_calc);
    let f = fract(f_calc);

    if (dot(f.xy, f.xy) < dot(f.zw, f.zw)) {
        return vec4f(f.xy, vec2f(2.0, 1.0) * i.xy);
    } else {
        return vec4f(f.zw, -(vec2f(2.0, 1.0) * i.zw + 1.0));
    }
}

fn triTile_scale(st: vec2f, scale: f32) -> vec4f {
    return triTile(st * scale);
}
