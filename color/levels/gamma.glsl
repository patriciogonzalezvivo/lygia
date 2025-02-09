
/*
contributors: Johan Ismael
description: |
    Color gamma correction similar to Levels adjustment in Photoshop
    Adapted from Romain Dura (http://mouaif.wordpress.com/?p=94)
use: levelsGamma(<vec3|vec4> color, <float|vec3> gamma)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_GAMMA
#define FNC_GAMMA
vec3 levelsGamma(in vec3 v, in vec3 g) { return pow(v, 1.0 / g); }
vec3 levelsGamma(in vec3 v, in float g) { return levelsGamma(v, vec3(g)); }
vec4 levelsGamma(in vec4 v, in vec3 g) { return vec4(levelsGamma(v.rgb, g), v.a); }
vec4 levelsGamma(in vec4 v, in float g) { return vec4(levelsGamma(v.rgb, g), v.a); }
#endif