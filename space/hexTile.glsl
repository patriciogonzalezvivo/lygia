#include "scale.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: make some hexagonal tiles. XY provide coordenates of the tile. While Z provides the distance to the center of the tile
use: <vec3> hexTile(<vec2> st [, <float> scale])
options:
    HEXTILE_SIZE
    HEXTILE_H
    HEXTILE_S
*/

#ifndef FNC_HEXTILE
#define FNC_HEXTILE
vec3 hexTile(in vec2 st, in float scl) {
    // this is hack to scale the hexagon to be more squared
    st = scale(st, vec2(1.24,1.)*scl) + vec2(.5,0.);

    vec3 q = vec3(st, .0);
    q.z = -.5 * q.x - q.y;

    float z = -.5 * q.x - q.y;
    q.y -= .5 * q.x;

    vec3 i = floor(q+.5);
    float s = floor(i.x + i.y + i.z);
    vec3 d = abs(i-q);

    // TODO: all this ifs should be avoided
    if( d.x >= d.y && d.x >= d.z ) i.x -= s;
    else if( d.y >= d.x && d.y >= d.z )	i.y -= s;
    else i.z -= s;

    vec2 coord = vec2(i.x, (i.y - i.z + (1.-mod(i.x, 2.)))/2.);
    float dist = length(st - vec2(coord.x, coord.y - .5*mod(i.x-1., 2.)));
    return vec3(coord, dist);
}

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