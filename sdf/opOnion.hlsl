/*
contributors:  Inigo Quiles
description: onion operation of one SDFs 
use: <float> opOnion( in <float> d, in <float> h )
*/

#ifndef FNC_OPONION
#define FNC_OPONION

float opOnion( in float d, in float h ) {
    return abs(d)-h;
}

#endif

