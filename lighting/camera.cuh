/*
contributors: Patricio Gonzalez Vivo
description: Generic Camera Structure
*/

#ifndef STR_CAMERA
#define STR_CAMERA
struct Camera {
    float3 pos;
    float3 dir;

    float3 up;
    float3 side;

    float invhalffov;
    float maxdist = 10.0f;
};
#endif