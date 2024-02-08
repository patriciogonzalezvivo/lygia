#include "triangle.hlsl"
#include "barycentric.hlsl"

/*
contributors: 
description: Returns the closest point on the surface of a triangle
use: <float3> closestDistance(<Triangle> tri, <float3> _pos) 
*/

#ifndef FNC_TRIANGLE_CLOSEST_POINT
#define FNC_TRIANGLE_CLOSEST_POINT

// https://github.com/nmoehrle/libacc/blob/master/primitives.h#L71
float3 closestPoint(Triangle _tri, float3 _triNormal, float3 _pos) {
    float3 ab = _tri.b - _tri.a;
    float3 ac = _tri.c - _tri.a;
    float3 normal = _triNormal;

    float3 p = _pos - dot(_triNormal, _pos - _tri.a) * _triNormal;
    float3 ap = p - _tri.a;

    float3 bcoords = barycentric(ab, ac, ap);

    if (bcoords.x < 0.0f) {
        float3 bc = _tri.c - _tri.b;
        float n = length( bc );
        float t = max(0.0f, min( dot(bc, p - _tri.b)/n, n));
        return _tri.b + t / n * bc;
    }

    if (bcoords.y < 0.0f) {
        float3 ca = _tri.a - _tri.c;
        float n = length( ca );
        float t = max(0.0f, min( dot(ca, p - _tri.c)/n, n));
        return _tri.c + t / n * ca;
    }

    if (bcoords.z < 0.0f) {
        float n = length( ab );
        float t = max(0.0f, min( dot(ab, p - _tri.a)/n, n));
        return _tri.a + t / n * ab;
    }

    return (_tri.a * bcoords.x + _tri.b * bcoords.y + _tri.c * bcoords.z);
}

// https://github.com/nmoehrle/libacc/blob/master/primitives.h#L71
float3 closestPoint(Triangle _tri, float3 _pos) {
    float3 ab = _tri.b - _tri.a;
    float3 ac = _tri.c - _tri.a;
    float3 normal = normalize( cross(ac,ab) );

    float3 p = _pos - dot(normal, _pos - _tri.a) * normal;
    float3 ap = p - _tri.a;

    float3 bcoords = barycentric(ab, ac, ap);

    if (bcoords.x < 0.0f) {
        float3 bc = _tri.c - _tri.b;
        float n = length( bc );
        float t = max(0.0f, min( dot(bc, p - _tri.b)/n, n));
        return _tri.b + t / n * bc;
    }

    if (bcoords.y < 0.0f) {
        float3 ca = _tri.a - _tri.c;
        float n = length( ca );
        float t = max(0.0f, min( dot(ca, p - _tri.c)/n, n));
        return _tri.c + t / n * ca;
    }

    if (bcoords.z < 0.0f) {
        float n = length( ab );
        float t = max(0.0f, min( dot(ab, p - _tri.a)/n, n));
        return _tri.a + t / n * ab;
    }

    return (_tri.a * bcoords.x + _tri.b * bcoords.y + _tri.c * bcoords.z);
}

#endif