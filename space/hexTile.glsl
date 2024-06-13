/*
contributors: Patricio Gonzalez Vivo
description: make some hexagonal tiles. XY provide coordenates of the tile. While Z provides the distance to the center of the tile
use: <vec4> hexTile(<vec2> st)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_tiles.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_HEXTILE
#define FNC_HEXTILE

vec4 hexTile(vec2 st) {
    vec2 s = vec2(1., 1.7320508);
    vec2 o = vec2(.5, 1.);
    st = st.yx;
    
    vec4 i = floor(vec4(st,st-o)/s.xyxy)+.5;
    vec4 f = vec4(st-i.xy*s, st-(i.zw+.5)*s);
    
    return dot(f.xy,f.xy) < dot(f.zw,f.zw) ? 
            vec4(f.yx+.5, i.xy):
            vec4(f.wz+.5, i.zw+o);
}
#endif