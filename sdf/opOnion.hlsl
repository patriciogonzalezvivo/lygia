/*
original_author:  Inigo Quiles
description: onion operation of one SDFs 
use: <vec4> opElongate( in <vec3> p, in <vec3> h )
*/

#ifndef FNC_OPONION
#define FNC_OPONION

float opOnion( in float d, in float h ) {
    return abs(d)-h;
}

#endif

