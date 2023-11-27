#include "spectral/gems.glsl"
#include "spectral/geoffrey.glsl"
#include "spectral/soft.glsl"
#include "spectral/zucconi.glsl"
#include "spectral/zucconi6.glsl"

#include "../../math/const.glsl"

/*
contributors: Martijn Steinrucken
description: Spectral Response Function https://www.shadertoy.com/view/wlSBzD
use: <vec3> spectral(<float> value)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_wavelength.frag
license: MIT License Copyright (c) 2020 Martijn Steinrucken
*/

#ifndef FNC_SPECTRAL
#define FNC_SPECTRAL
vec3 spectral(const in float x) {
    return  (vec3( 1.220023e0,-1.933277e0, 1.623776e0) +
            (vec3(-2.965000e1, 6.806567e1,-3.606269e1) +
            (vec3( 5.451365e2,-7.921759e2, 6.966892e2) +
            (vec3(-4.121053e3, 4.432167e3,-4.463157e3) +
            (vec3( 1.501655e4,-1.264621e4, 1.375260e4) +
            (vec3(-2.904744e4, 1.969591e4,-2.330431e4) +
            (vec3( 3.068214e4,-1.698411e4, 2.229810e4) +
            (vec3(-1.675434e4, 7.594470e3,-1.131826e4) +
             vec3( 3.707437e3,-1.366175e3, 2.372779e3)
            *x)*x)*x)*x)*x)*x)*x)*x)*x;
}

vec3 spectral( in float x, const in float l ) {
    x = 1.0 - x;
    // (optional) rectangular expand
    x = mix((x * .5)+.25,x,1. - l);
    return vec3(
    // RED + VIOLET-FALLOFF
    -.0833333 * ( l - 1. ) * (
    cos( PI * max( 0., min( 1., 12. * abs( ( .0833333 * l + x - .8333333 ) / ( l + 2. ) ) ) ) ) + 1. )
    + .5 * cos( PI * min( 1., ( l + 3. ) * abs( -.1666666 * l + x - .3333333 ) ) ) + .5,
    // GREEN, BLUE
    .5 + .5 * cos( PI * min(
    vec2( 1. ), abs( vec2( x ) - vec2( .5, ( 1.0 - ( ( 2. + l ) / 3. ) * .5 ) ) )
    * vec2( 3. + l ) ) ) );
}


#endif