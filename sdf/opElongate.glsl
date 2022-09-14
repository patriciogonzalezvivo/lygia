/*
original_author:  Inigo Quiles
description: elongate operation of two SDFs 
use: <vec4> opElongate( in <vec3> p, in <vec3> h )
*/

#ifndef FNC_OPELONGATE
#define FNC_OPELONGATE

vec2 opElongate( in vec2 p, in vec2 h ) {
    return p-clamp(p,-h,h); 
}

vec3 opElongate( in vec3 p, in vec3 h ) {
    return p-clamp(p,-h,h); 
}

vec4 opElongate( in vec4 p, in vec4 h ) {
    vec3 q = abs(p)-h;
    return vec4( max(q,0.0), min(max(q.x,max(q.y,q.z)), 0.0) );
}

#endif

