/*
contributors: Patricio Gonzalez Vivo
description: Directional Light Structure
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef STR_LIGHT_DIRECTIONAL
#define STR_LIGHT_DIRECTIONAL
struct LightDirectional {
    vec3    direction;
    vec3    color;
    float   intensity;
};
#endif
