#include "identity.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: |
    Linear interpolation between two quaternions.
    This function is based on the implementation of slerp() found in the GLM library.
use: <QUAT> quatLerp(<QUAT> a, <QUAT> b, <float> t) 
*/

#ifndef FNC_QUATLERP
#define FNC_QUATLERP

QUAT quatLerp(QUAT a, QUAT b, float t) {
    // if either input is zero, return the other.
    if (length(a) == 0.0) {
        if (length(b) == 0.0) {
            return QUAT_IDENTITY;
        }
        return b;
    } else if (length(b) == 0.0) {
        return a;
    }

    float cosHalfAngle = a.w * b.w + dot(a.xyz, b.xyz);

    if (cosHalfAngle >= 1.0 || cosHalfAngle <= -1.0) {
        return a;
    } else if (cosHalfAngle < 0.0) {
        b.xyz = -b.xyz;
        b.w = -b.w;
        cosHalfAngle = -cosHalfAngle;
    }

    float blendA;
    float blendB;
    if (cosHalfAngle < 0.99) {
        // do proper slerp for big angles
        float halfAngle = acos(cosHalfAngle);
        float sinHalfAngle = sin(halfAngle);
        float oneOverSinHalfAngle = 1.0 / sinHalfAngle;
        blendA = sin(halfAngle * (1.0 - t)) * oneOverSinHalfAngle;
        blendB = sin(halfAngle * t) * oneOverSinHalfAngle;
    } else {
        // do lerp if angle is really small.
        blendA = 1.0 - t;
        blendB = t;
    }

    QUAT result = QUAT(blendA * a.xyz + blendB * b.xyz, blendA * a.w + blendB * b.w);
    if (length(result) > 0.0) {
        return normalize(result);
    }
    return QUAT_IDENTITY;
}

#endif