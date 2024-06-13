/*
contributors: Patricio Gonzalez Vivo
description: get screen coordinates from view position
use: <vec2> view2screenPosition(<vec3> viewPosition )
options:
    - CAMERA_PROJECTION_MATRIX: mat4 matrix with camera projection
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef CAMERA_PROJECTION_MATRIX
#define CAMERA_PROJECTION_MATRIX u_projectionMatrix
#endif

#ifndef FNC_VIEW2SCREENPOSITION
#define FNC_VIEW2SCREENPOSITION
vec2 view2screenPosition(vec3 viewPosition){
    vec4 clip = CAMERA_PROJECTION_MATRIX * vec4(viewPosition, 1.0);
    return (clip.xy / clip.w + 1.0) * 0.5;
}
#endif