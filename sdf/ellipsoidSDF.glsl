/*
contributors:  Inigo Quiles
description: generate the SDF of an approximated ellipsoid
use: <float> ellipsoidSDF( in <vec3> p, in <vec3> r )
*/

#ifndef FNC_ELLIPSOIDSDF
#define FNC_ELLIPSOIDSDF

float ellipsoidSDF( in vec3 p, in vec3 r ) {
    float k0 = length(p/r);
    float k1 = length(p/(r*r));
    return k0*(k0-1.0)/k1;
}

#endif