/*
contributors: Patricio Gonzalez Vivo
description: Generate a matrix to change a the contrast of any color
use: contrastMatrix(<float> amount)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_brightnessContrastMatrix.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_CONTRASTMATRIX
#define FNC_CONTRASTMATRIX
float4x4 contrastMatrix(in float amount) {
    float t = ( 1. - amount ) * .5;
    return float4x4(  amount, .0, .0, t,
                      .0, amount, .0, t,
                      .0, .0, amount, t,
                      .0, .0,     .0, 1. );
}
#endif
