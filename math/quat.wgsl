#include "quat/mul.wgsl"
#include "quat/identity.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    creates a quaternion (QUAT) from a given radian of rotation about a given axis or from a given forward vector and up vector
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// A given r of rotation about a given axis
fn quat(axis: vec3f, r: f32) -> vec4f {
    let sn = sin(r * 0.5);
    let cs = cos(r * 0.5);
    return vec4f(axis * sn, cs);
}

fn quatFowardUp(f: vec3f, _up: vec3f) -> vec4f {
    let right = normalize(cross(f, -_up));
    let up = normalize(cross(f, right));

    let m00 = right.x;
    let m01 = right.y;
    let m02 = right.z;
    let m10 = up.x;
    let m11 = up.y;
    let m12 = up.z;
    let m20 = f.x;
    let m21 = f.y;
    let m22 = f.z;

    let num8 = (m00 + m11) + m22;
    var q = vec4f(0.0, 0.0, 0.0, 1.0);
    if (num8 > 0.0) {
        var num = sqrt(num8 + 1.0);
        q.w = num * 0.5;
        num = 0.5 / num;
        q.x = (m12 - m21) * num;
        q.y = (m20 - m02) * num;
        q.z = (m01 - m10) * num;
        return q;
    }

    if ((m00 >= m11) && (m00 >= m22)) {
        let num7 = sqrt(((1.0 + m00) - m11) - m22);
        let num4 = 0.5 / num7;
        q.x = 0.5 * num7;
        q.y = (m01 + m10) * num4;
        q.z = (m02 + m20) * num4;
        q.w = (m12 - m21) * num4;
        return q;
    }

    if (m11 > m22) {
        let num6 = sqrt(((1.0 + m11) - m00) - m22);
        let num3 = 0.5 / num6;
        q.x = (m10 + m01) * num3;
        q.y = 0.5 * num6;
        q.z = (m21 + m12) * num3;
        q.w = (m20 - m02) * num3;
        return q;
    }

    let num5 = sqrt(((1.0 + m22) - m00) - m11);
    let num2 = 0.5 / num5;
    q.x = (m20 + m02) * num2;
    q.y = (m21 + m12) * num2;
    q.z = 0.5 * num5;
    q.w = (m01 - m10) * num2;
    return q;
}
