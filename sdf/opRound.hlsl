/*
contributors:  Inigo Quiles
description: round SDFs 
use: <float> opRound( in <float> d, <float> h ) 
*/

#ifndef FNC_OPREVOLVE
#define FNC_OPREVOLVE

float opRound( in float d, in float h ) {
    return d - h;
}

#endif

