/*
contributors:  Inigo Quiles
description: generate the SDF of a link
use: <float> linkSDF( <vec3> p, <float> le, <float> r1, <float> r2 ) 
*/

#ifndef FNC_LINKSDF
#define FNC_LINKSDF
float linkSDF( vec3 p, float le, float r1, float r2 ) {
    vec3 q = vec3( p.x, max(abs(p.y)-le,0.0), p.z );
    return length(vec2(length(q.xy)-r1,q.z)) - r2;
}
#endif