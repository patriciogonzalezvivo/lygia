/*
contributors:  Inigo Quiles
description: generate the SDF of a torus
use: <float> torusSDF( in <vec3> pos, in <vec2> h [, in <float> ra, in <float> rb] ) 
*/

#ifndef FNC_TORUSSDF
#define FNC_TORUSSDF
float torusSDF( vec3 p, vec2 t ) { return length( vec2(length(p.xz)-t.x,p.y) )-t.y; }

float torusSDF(in vec3 p, in vec2 sc, in float ra, in float rb) {
    p.x = abs(p.x);
    float k = (sc.y*p.x>sc.x*p.y) ? dot(p.xy,sc) : length(p.xy);
    return sqrt( dot(p,p) + ra*ra - 2.0*ra*k ) - rb;
}
#endif