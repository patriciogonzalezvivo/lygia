/*
original_author: Patricio Gonzalez Vivo
description: returns a 4x4 scale matrix
use: scale4d(<float> radians)
*/

#ifndef FNC_SCALE4D
mat4 scale4d(float _scale) {
    return mat4(
        _scale, 0.0, 0.0, 0.0,
        0.0, _scale, 0.0, 0.0,
        0.0, 0.0, _scale, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
}
#endif
