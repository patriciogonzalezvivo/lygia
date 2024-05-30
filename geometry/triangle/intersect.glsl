#include "triangle.glsl"
#include "../../lighting/ray.glsl"

/*
contributors: Inigo Quiles
description: Based on https://www.iquilezles.org/www/articles/intersectors/intersectors.htm
use: <float> intersect(<Triangle> tri, <vec3> rayOrigin, <vec3> rayDir, out <vec3> intersectionPoint)
*/

#ifndef FNC_TRIANGLE_INTERSECT
#define FNC_TRIANGLE_INTERSECT

float intersect(Triangle _tri, vec3 _rayOrigin, vec3 _rayDir, inout vec3 _point) {
    vec3 v1v0 = _tri.b - _tri.a;
    vec3 v2v0 = _tri.c - _tri.a;
    vec3 rov0 = _rayOrigin - _tri.a;
    _point = cross(v1v0, v2v0);
    vec3 q = cross(rov0, _rayDir);
    float d = 1.0f / dot(_rayDir, _point);
    float u = d * -dot(q, v2v0);
    float v = d *  dot(q, v1v0);
    float t = d * -dot(_point, rov0);
    if (u < 0.0f || u > 1.0f || v < 0.0f || (u+v) > 1.0f || t < 0.0f)
        t = 9999999.9f; // No intersection

    return t;
}

float intersect(Triangle _tri, Ray _ray, inout vec3 _point) { return intersect(_tri, _ray.origin, _ray.direction, _point); }
float intersect(Triangle _tri, Ray _ray) { 
    vec3 p;
    return intersect(_tri, _ray, p); 
}

#endif
