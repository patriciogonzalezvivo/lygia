/*
original_author: Patricio Gonzalez Vivo
description: returns a 3x3 scale matrix
use: scale3d(<float> radians)
*/

#ifndef FNC_SCALE4D
mat3 scale3d(float _scale) {
    return mat3(
        _scale, 0.0, 0.0,
        0.0, _scale, 0.0,
        0.0, 0.0, _scale,
    );
}
#endif