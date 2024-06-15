/*
contributors: Patricio Gonzalez Vivo
description: Generate a matrix to change a the brightness of any color
use: brightnessMatrix(<float> amount)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_brightnessContrastMatrix.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_BRIGHTNESSMATRIX
#define FNC_BRIGHTNESSMATRIX
mat4 brightnessMatrix(in float a) {
    return mat4(  1., 0., 0., 0.,
                  0., 1., 0., 0.,
                  0., 0., 1., 0.,
                  a, a, a, 1. );
}
#endif
