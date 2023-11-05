/*
contributors: Patricio Gonzalez Vivo
description: get screen coordinates from view position 
use: <vec2> view2screenPosition(<vec3> viewPosition )
options:
    - CAMERA_PROJECTION_MATRIX: mat4 matrix with camera projection
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