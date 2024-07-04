/*
contributors: Patricio Gonzalez Vivo
description: Generic Camera Structure
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef STR_CAMERA
#define STR_CAMERA
struct Camera {
    vec3 pos;
    vec3 dir;

    vec3 up;
    vec3 side;

    float invhalffov;
    float maxdist;
};
#endif