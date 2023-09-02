#include "type.glsl"

// Quaternion multiplication
// http://mathworld.wolfram.com/Quaternion.html

#ifndef FNC_QUATMUL
#define FNC_QUATMUL
QUAT quatMul(QUAT q1, QUAT q2) {
    return QUAT(
        q2.xyz * q1.w + q1.xyz * q2.w + cross(q1.xyz, q2.xyz),
        q1.w * q2.w - dot(q1.xyz, q2.xyz)
    );
}
#endif