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
#define DIST_MINKOWSKI_P 2.0 // 1: Manhattan, 2: Euclidean, Infinity: Chebychev
#endif

#ifndef FNC_DIST
#define FNC_DIST

float distEuclidean(vec2 a, vec2 b) { return distance(a, b); }
float distEuclidean(vec3 a, vec3 b) { return distance(a, b); }
float distEuclidean(vec4 a, vec4 b) { return distance(a, b); }

// https://en.wikipedia.org/wiki/Taxicab_geometry
float distManhattan(vec2 a, vec2 b) { return abs(a.x - b.x) + abs(a.y - b.y); }
float distManhattan(vec3 a, vec3 b) { return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z); }
float distManhattan(vec4 a, vec4 b) { return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z) + abs(a.w - b.w); }

// https://en.wikipedia.org/wiki/Chebyshev_distance
float distChebychev(vec2 a, vec2 b) { return max(abs(a.x - b.x), abs(a.y - b.y)); }
float distChebychev(vec3 a, vec3 b) { return max(abs(a.x - b.x), max(abs(a.y - b.y), abs(a.z - b.z))); }
float distChebychev(vec4 a, vec4 b) { return max(abs(a.x - b.x), max(abs(a.y - b.y), max(abs(a.z - b.z), abs(a.w - b.w) ))); }

// https://en.wikipedia.org/wiki/Minkowski_distance
float distMinkowski(vec2 a, vec2 b) { return  pow(pow(abs(a.x - b.x), DIST_MINKOWSKI_P) + pow(abs(a.y - b.y), DIST_MINKOWSKI_P), 1.0 / DIST_MINKOWSKI_P); }
float distMinkowski(vec3 a, vec3 b) { return  pow(pow(abs(a.x - b.x), DIST_MINKOWSKI_P) + pow(abs(a.y - b.y), DIST_MINKOWSKI_P) + pow(abs(a.z - b.z), DIST_MINKOWSKI_P), 1.0 / DIST_MINKOWSKI_P); }
float distMinkowski(vec4 a, vec4 b) { return  pow(pow(abs(a.x - b.x), DIST_MINKOWSKI_P) + pow(abs(a.y - b.y), DIST_MINKOWSKI_P) + pow(abs(a.z - b.z), DIST_MINKOWSKI_P) + pow(abs(a.w - b.w), DIST_MINKOWSKI_P), 1.0 / DIST_MINKOWSKI_P); }

float dist(vec2 a, vec2 b) { return DIST_FNC(a, b); }
float dist(vec3 a, vec3 b) { return DIST_FNC(a, b); }
float dist(vec4 a, vec4 b) { return DIST_FNC(a, b); }

#endif