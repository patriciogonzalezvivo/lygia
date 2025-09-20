#include "../math/decimate.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Decimate normal vector by reducing the number of unique directions based on a given precision value.
use: <vec3> decimateNormal( <vec3> normal, <float> press )
license:
    - Copyright (c) 2025 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2025 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


#ifndef FNC_DECIMATE_NORMAL
#define FNC_DECIMATE_NORMAL

vec3 decimateNormal(vec3 normal, float press) {
    // convert normals into phi, theta angles
    float phi = atan(normal.y, normal.x);
    float theta = acos(normal.z);
    phi = decimate(phi, press);
    theta = decimate(theta, press);
    // convert angles into normal
    normal = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
    return normal;
}

#endif