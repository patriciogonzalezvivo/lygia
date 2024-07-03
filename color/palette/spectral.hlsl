/*
description: Include all spectral palettes
*/

#include "spectral/gems.hlsl"
#include "spectral/geoffrey.hlsl"
#include "spectral/soft.hlsl"
#include "spectral/zucconi.hlsl"
#include "spectral/zucconi6.hlsl"

#include "../../math/const.hlsl"

/*
contributors: Martijn Steinrucken
description: Spectral Response Function https://www.shadertoy.com/view/wlSBzD
use: <float3> spectral(<float> value)
license: MIT License Copyright (c) 2020 Martijn Steinrucken
*/

#ifndef FNC_SPECTRAL
#define FNC_SPECTRAL
float3 spectral(const in float x) {
    return  (float3( 1.220023e0,-1.933277e0, 1.623776e0) +
            (float3(-2.965000e1, 6.806567e1,-3.606269e1) +
            (float3( 5.451365e2,-7.921759e2, 6.966892e2) +
            (float3(-4.121053e3, 4.432167e3,-4.463157e3) +
            (float3( 1.501655e4,-1.264621e4, 1.375260e4) +
            (float3(-2.904744e4, 1.969591e4,-2.330431e4) +
            (float3( 3.068214e4,-1.698411e4, 2.229810e4) +
            (float3(-1.675434e4, 7.594470e3,-1.131826e4) +
             float3( 3.707437e3,-1.366175e3, 2.372779e3)
            *x)*x)*x)*x)*x)*x)*x)*x)*x;
}

float3 spectral( in float x, const in float l ) {
    x = 1.0 - x;
    // (optional) rectangular expand
    x = lerp((x * .5)+.25,x,1. - l);
    return float3(
    // RED + VIOLET-FALLOFF
    -.0833333 * ( l - 1. ) * (
    cos( PI * max( 0., min( 1., 12. * abs( ( .0833333 * l + x - .8333333 ) / ( l + 2. ) ) ) ) ) + 1. )
    + .5 * cos( PI * min( 1., ( l + 3. ) * abs( -.1666666 * l + x - .3333333 ) ) ) + .5,
    // GREEN, BLUE
    .5 + .5 * cos( PI * min(
    float2( 1.0, 1.0), abs( float2( x, x ) - float2( .5, ( 1.0 - ( ( 2. + l ) / 3. ) * .5 ) ) )
    * ( float2(3., 3.) + l ) ) ) );
}


#endif