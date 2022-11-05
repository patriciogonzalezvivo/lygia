#include "sphereMap.glsl"
/*
original_author: Patricio Gonzalez Vivo
description: matcap
use: 
    - <vec2> matcap(<vec3> normal, <vec3> eye) 
    - <vec2> matcap(<sampler2D> tex, <vec3> normal, <vec3> eye) 
*/

#ifndef FNC_MATCAP
#define FNC_MATCAP
// vec2 matcap(vec3 normal, vec3 eye) {
//     vec3 reflected = reflect(eye, normal);
//     float m = 2.8284271247461903 * sqrt( reflected.z+1.0 );
    // return reflected.xy / m + 0.5;
// }

#define matcap sphereMap
#endif