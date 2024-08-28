/*
contributors: Patricio Gonzalez Vivo
description: Point light structure
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef STR_LIGHT_POINT
#define STR_LIGHT_POINT
struct LightPoint {
    float3    position;
    float3    color;
    float   intensity;
    float   falloff;
};
#endif
