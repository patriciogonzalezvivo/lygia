/*
contributors: Patricio Gonzalez Vivo
description: make some hexagonal tiles. XY provide coordinates of the tile. While Z provides the distance to the center of the tile
use: <vec4> hexTile(<vec2> st)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_tiles.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn hexTile(st: vec2f) -> vec4f {
    let s = vec2f(1., 1.7320508);
    let o = vec2f(.5, 1.);
    st = st.yx;
    
    let i = floor(vec4f(st,st-o)/s.xyxy)+.5;
    let f = vec4f(st-i.xy*s, st-(i.zw+.5)*s);
    
    return dot(f.xy,f.xy) < dot(f.zw,f.zw) ? 
            vec4f(f.yx+.5, i.xy):
            vec4f(f.wz+.5, i.zw+o);
}
