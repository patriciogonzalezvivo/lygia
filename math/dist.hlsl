/*
contributors: Shadi El Hajj
description: Commonly used distance functions.
options:
- DIST_FNC: change the distance function, currently implemented are distEuclidean, distManhattan, distChebychev and distMinkowski
- DIST_MINKOWSKI_P: the power of the Minkowski distance function (1.0 Manhattan, 2.0 Euclidean, Infinity Chebychev)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef DIST_FNC
#define DIST_FNC distEuclidean
#endif

#ifndef DIST_MINKOWSKI_P
#define DIST_MINKOWSKI_P 2 // 1: Manhattan, 2: Euclidean, Infinity: Chebychev
#endif

#ifndef FNC_DIST
#define FNC_DIST

float distEuclidean(float2 a, float2 b) {
    return distance(a, b);
}

float distEuclidean(float3 a, float3 b) {
    return distance(a, b);
}

// https://en.wikipedia.org/wiki/Taxicab_geometry
float distManhattan(float2 a, float2 b) {
    return abs(a.x - b.x) + abs(a.y - b.y);
}

float distManhattan(float3 a, float3 b) {
    return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z);
}

// https://en.wikipedia.org/wiki/Chebyshev_distance
float distChebychev(float2 a, float2 b) {
    return max(abs(a.x - b.x), abs(a.y - b.y));
}

float distChebychev(float3 a, float3 b) {
    return max(abs(a.x - b.x), max(abs(a.y - b.y), abs(a.z - b.z)));
}

// https://en.wikipedia.org/wiki/Minkowski_distance
float distMinkowski(float2 a, float2 b) {
    return  pow(pow(abs(a.x - b.x), DIST_MINKOWSKI_P)
              + pow(abs(a.y - b.y), DIST_MINKOWSKI_P),
            1.0 / DIST_MINKOWSKI_P);
}

float distMinkowski(float3 a, float3 b) {
    return  pow(pow(abs(a.x - b.x), DIST_MINKOWSKI_P)
              + pow(abs(a.y - b.y), DIST_MINKOWSKI_P)
              + pow(abs(a.z - b.z), DIST_MINKOWSKI_P),
            1.0 / DIST_MINKOWSKI_P);
}

float dist(float2 a, float2 b) { return DIST_FNC(a, b); }
float dist(float3 a, float3 b) { return DIST_FNC(a, b); }

#endif