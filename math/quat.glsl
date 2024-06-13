#include "quat/mul.glsl"
#include "quat/identity.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    creates a quaternion (QUAT) from a given radian of rotation about a given axis or from a given forward vector and up vector
use:
    - <QUAT> quat(<vec3> axis, <float> r)
    - <QUAT> quat(<vec3> forward [, <vec3> up])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_QUAT
#define FNC_QUAT

// A given r of rotation about a given axis
QUAT quat(vec3 axis, float r) {
    float sn = sin(r * 0.5);
    float cs = cos(r * 0.5);
    return QUAT(axis * sn, cs);
}

QUAT quat(vec3 f, vec3 up) {
    vec3 right = normalize(cross(f, -up));
    up = normalize(cross(f, right));

    float m00 = right.x;
    float m01 = right.y;
    float m02 = right.z;
    float m10 = up.x;
    float m11 = up.y;
    float m12 = up.z;
    float m20 = f.x;
    float m21 = f.y;
    float m22 = f.z;

    float num8 = (m00 + m11) + m22;
    QUAT q = QUAT_IDENTITY;
    if (num8 > 0.0) {
        float num = sqrt(num8 + 1.0);
        q.w = num * 0.5;
        num = 0.5 / num;
        q.x = (m12 - m21) * num;
        q.y = (m20 - m02) * num;
        q.z = (m01 - m10) * num;
        return q;
    }

    if ((m00 >= m11) && (m00 >= m22)) {
        float num7 = sqrt(((1.0 + m00) - m11) - m22);
        float num4 = 0.5 / num7;
        q.x = 0.5 * num7;
        q.y = (m01 + m10) * num4;
        q.z = (m02 + m20) * num4;
        q.w = (m12 - m21) * num4;
        return q;
    }

    if (m11 > m22) {
        float num6 = sqrt(((1.0 + m11) - m00) - m22);
        float num3 = 0.5 / num6;
        q.x = (m10 + m01) * num3;
        q.y = 0.5 * num6;
        q.z = (m21 + m12) * num3;
        q.w = (m20 - m02) * num3;
        return q;
    }

    float num5 = sqrt(((1.0 + m22) - m00) - m11);
    float num2 = 0.5 / num5;
    q.x = (m20 + m02) * num2;
    q.y = (m21 + m12) * num2;
    q.z = 0.5 * num5;
    q.w = (m01 - m10) * num2;
    return q;
}

QUAT quat(vec3 f) { return quat(f, vec3(0.0, 1.0, 0.0)); }

#endif