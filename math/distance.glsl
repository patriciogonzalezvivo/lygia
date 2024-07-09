/*
contributors: Shadi El Hajj
description: Commonly used distance functions.
*/

#ifndef DIST_MINKOWSKI_P
#define DIST_MINKOWSKI_P 2 // 1: Manhattan, 2: Euclidean, Infinity: Chebychev
#endif

float euclidean(vec2 a, vec2 b) {
    return distance(a, b);
}

float euclidean(vec3 a, vec3 b) {
    return distance(a, b);
}

// https://en.wikipedia.org/wiki/Taxicab_geometry
float manhattan(vec2 a, vec2 b) {
    return abs(a.x - b.x) + abs(a.y - b.y);
}

float manhattan(vec3 a, vec3 b) {
    return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z);
}

// https://en.wikipedia.org/wiki/Chebyshev_distance
float chebychev(vec2 a, vec2 b) {
    return max(abs(a.x - b.x), abs(a.y - b.y));
}

float chebychev(vec3 a, vec3 b) {
    return max(abs(a.x - b.x), max(abs(a.y - b.y), abs(a.z - b.z)));
}

// https://en.wikipedia.org/wiki/Minkowski_distance
float minkowski(vec2 a, vec2 b) {
    return  pow(pow(abs(a.x - b.x), DIST_MINKOWSKI_P)
              + pow(abs(a.y - b.y), DIST_MINKOWSKI_P),
            1.0 / DIST_MINKOWSKI_P);
}

float minkowski(vec3 a, vec3 b) {
    return  pow(pow(abs(a.x - b.x), DIST_MINKOWSKI_P)
              + pow(abs(a.y - b.y), DIST_MINKOWSKI_P)
              + pow(abs(a.z - b.z), DIST_MINKOWSKI_P),
            1.0 / DIST_MINKOWSKI_P);
}