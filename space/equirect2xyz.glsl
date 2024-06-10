#include "../math/const.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: equirect 2D projection to 3D vector
use: <vec3> equirect2xyz(<vec2> uv)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_EQUIRECT2XYZ
#define FNC_EQUIRECT2XYZ
vec3 equirect2xyz(vec2 uv) {
    float Phi = PI - uv.y * PI;
    float Theta = uv.x * TWO_PI;
    vec3 dir = vec3(cos(Theta), 0.0, sin(Theta));
    dir.y   = cos(Phi);//clamp(cos(Phi), MinCos, 1.0);
    dir.xz *= sqrt(1.0 - dir.y * dir.y);
    return dir;
}
#endif