/*
contributors:  Inigo Quiles
description: generate the SDF of a plane
use: <float> planeSDF( in <vec3> pos, in <vec2> h ) 
*/

#ifndef FNC_PLANESDF
#define FNC_PLANESDF
float planeSDF( vec3 p ) { 
   return p.y; 
}

float planeSDF(vec3 p, vec3 planePoint, vec3 planeNormal) {
    return (dot(planeNormal, p) + dot(planeNormal, planePoint)) / length(planeNormal);
}
#endif