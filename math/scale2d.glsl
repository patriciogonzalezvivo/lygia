/*
original_author: Patricio Gonzalez Vivo
description: returns a 2x2 scale matrix
use: scale2d(<float> radians)
*/

#ifndef FNC_SCALE4D
mat2 scale2d(float _scale) {
    return mat2(
        _scale, 0.0,
        0.0, _scale,
    );
}
#endif