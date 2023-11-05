/*
contributors:  Inigo Quiles
description: generate the SDF of a box
use: <float> boxSDF( in <vec3> pos [, in <vec3> borders ] ) 
*/

#ifndef FNC_BOXSDF
#define FNC_BOXSDF

float boxSDF( vec3 p ) {
    vec3 d = abs(p);
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

float boxSDF( vec3 p, vec3 b ) {
    vec3 d = abs(p) - b;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

#endif