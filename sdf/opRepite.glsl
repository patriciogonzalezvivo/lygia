/*
original_author:  Inigo Quiles
description: repite operation of one 2D SDFs 
use: <vec4> opElongate( in <vec3> p, in <vec3> h )
*/

#ifndef FNC_OPREPITE
#define FNC_OPREPITE

vec2 opRepite( in vec2 p, in float s ) {
    return mod(p+s*0.5,s)-s*0.5;
}

vec3 opRepite( in vec3 p, in vec3 c ) {
    return mod(p+0.5*c,c)-0.5*c;
}

vec2 opRepite( in vec2 p, in vec2 lima, in vec2 limb, in float s ) {
    return p-s*clamp(floor(p/s),lima,limb);
}

vec3 opRepite( in vec3 p, in vec3 lima, in vec3 limb, in float s ) {
    return p-s*clamp(floor(p/s),lima,limb);
}

#endif

