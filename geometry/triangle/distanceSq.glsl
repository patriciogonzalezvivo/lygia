#include "triangle.glsl"
#include "../../math/saturate.glsl"

/*
contributors: Inigo Quiles
description: Returns the closest sq distance to the surface of a triangle
use: <vec3> closestDistanceSq(<Triangle> tri, <vec3> _pos) 
*/

#ifndef FNC_TRIANGLE_DISTANCE_SQ
#define FNC_TRIANGLE_DISTANCE_SQ

float distanceSq(Triangle _tri, vec3 _pos) {
    // prepare data    
    vec3 v21 = _tri.b - _tri.a; vec3 p1 = _pos - _tri.a;
    vec3 v32 = _tri.c - _tri.b; vec3 p2 = _pos - _tri.b;
    vec3 v13 = _tri.a - _tri.c; vec3 p3 = _pos - _tri.c;
    vec3 nor = cross( v21, v13 );
    return sqrt(    
                    // inside/outside test    
                    (sign( dot(cross(v21,nor),p1)) + 
                     sign( dot(cross(v32,nor),p2)) + 
                     sign( dot(cross(v13,nor),p3)) < 2.0) 
                    ?
                    // 3 edges 
                    min( min( 
                    lengthSq(v21 *  saturate( dot(v21,p1)/lengthSq(v21) ) - p1), 
                    lengthSq(v32 *  saturate( dot(v32,p2)/lengthSq(v32) ) - p2) ), 
                    lengthSq(v13 *  saturate( dot(v13,p3)/lengthSq(v13) ) - p3) ) 
                    :
                    // 1 face    
                    dot(nor,p1)* dot(nor,p1)/lengthSq(nor)
                );
}

#endif