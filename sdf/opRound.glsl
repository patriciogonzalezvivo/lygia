/*
contributors:  Inigo Quiles
description: round SDFs 
use: <float> opRound( in <float> d, <float> h ) 
*/

#ifndef FNC_OPROUND
#define FNC_OPROUND

float opRound( in float d, in float h ) {
    return d - h;
}

#endif

