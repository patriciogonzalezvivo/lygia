/*
contributors: [Ivan Dianov, Shadi El Hajj]
description: polar to cartesian conversion.
use: polar2cart(<float2> polar)
*/

#ifndef FNC_POLAR2CART
#define FNC_POLAR2CART

float2 polar2cart(in float2 polar) {
    return float2(cos(polar.x), sin(polar.x)) * polar.y;
}

// https://mathworld.wolfram.com/SphericalCoordinates.html
float3 polar2cart( in float r, in float phi, in float theta) {
    float x = r * cos(theta) * sin(phi);
    float y = r * sin(theta) * sin(phi);
    float z = r * cos(phi);
    return float3(x, y, z);
}

#endif