/*
contributors:  Inigo Quiles
description: repeat operation for 2D/3D SDFs 
use: <vec4> opElongate( in <vec3> p, in <vec3> h )
*/

#ifndef FNC_OPREPEAT
#define FNC_OPREPEAT

vec2 opRepeat( in vec2 p, in float s ) {
    return mod(p+s*0.5,s)-s*0.5;
}

vec3 opRepeat( in vec3 p, in vec3 c ) {
    return mod(p+0.5*c,c)-0.5*c;
}

vec2 opRepeat( in vec2 p, in vec2 lima, in vec2 limb, in float s ) {
    return p-s*clamp(floor(p/s),lima,limb);
}

vec3 opRepeat( in vec3 p, in vec3 lima, in vec3 limb, in float s ) {
    return p-s*clamp(floor(p/s),lima,limb);
}

#endif

