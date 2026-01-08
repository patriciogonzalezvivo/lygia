/*
contributors:  Inigo Quiles
description: Subtraction operation of two SDFs 
use: <float> opSubstraction( in <float> d1, in <float> d2 [, <float> smooth_factor]) 
*/

#ifndef FNC_OPSUBSTRACTION
#define FNC_OPSUBSTRACTION

float opSubtraction( float d1, float d2 ) { return max(-d1,d2); }

float opSubtraction( float d1, float d2, float k ) {
    float h = saturate( 0.5 - 0.5*(d2+d1)/k );
    return lerp( d2, -d1, h ) + k*h*(1.0-h);
}

#endif