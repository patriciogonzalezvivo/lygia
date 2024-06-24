#include "../math/saturate.glsl"

/*
contributors:  Inigo Quiles
description: Substraction operation of two SDFs 
use: <float> opSubstraction( in <float> d1, in <float> d2 [, <float> smooth_factor]) 
*/

#ifndef FNC_OPSUBSTRACTION
#define FNC_OPSUBSTRACTION

float opSubtraction( float d1, float d2 ) { return max(-d1, d2); }
vec4  opSubtraction( vec4 d1, vec4 d2 ) { return (-d1.a > d2.a) ? -d1 : d2; }

float opSubtraction( float d1, float d2, float k ) {
    float h = clamp( 0.5 - 0.5*(d2+d1)/k, 0.0, 1.0 );
    return mix( d2, -d1, h ) + k*h*(1.0-h);
}


vec4 opSubtraction( vec4 d1, vec4 d2, float k ) {
    float h = clamp( 0.5 - 0.5*(d2.a+d1.a)/k, 0.0, 1.0 );
    vec4 result = mix( d2, -d1, h );
    result.a += k*h*(1.0-h);
    return result;
}

#endif