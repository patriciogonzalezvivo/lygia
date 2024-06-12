/*
contributors: Patricio Gonzalez Vivo
description: |
    Linear interpolation between two quaternions.
    This function is based on the implementation of slerp() found in the GLM library.
use: <QUAT> quatLerp(<QUAT> a, <QUAT> b, <float> t)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn quatLerp(qa: vec4f, _qb: vec4f, t: f32) -> vec4f {
    // qa = normalize(qa);
    // qb = normalize(qb);

    // Calculate angle between them.
    var cosHalfTheta = qa.w * _qb.w + dot(qa.xyz, _qb.xyz);
    
    // avoid taking the longer way: choose one representation
    let qb = select(_qb, -_qb, cosHalfTheta < 0.0);
    cosHalfTheta = select(cosHalfTheta, -cosHalfTheta, cosHalfTheta < 0.0);

    // if qa = qb or qa = -qb then theta = 0 and we can return qa
    if (abs(cosHalfTheta) >= 1.0) // greater-sign necessary for numerical stability
        return qa;

    // Calculate temporary values.
    let halfTheta = acos(cosHalfTheta);
    let sinHalfTheta = sqrt(1.0 - cosHalfTheta * cosHalfTheta); // NOTE: we checked above that |cosHalfTheta| < 1
    // if theta = pi then result is not fully defined
    // we could rotate around any axis normal to qa or qb
    if (abs(sinHalfTheta) < 0.001/*some epsilon*/)
        // return quatAdd( quatMul(qa, 0.5), quatMul(qb, 0.5));
        return normalize( qa * 0.5 + qb * 0.5 );

    let ratioA = sin((1.0 - t) * halfTheta) / sinHalfTheta;
    let ratioB = sin(t * halfTheta) / sinHalfTheta;

    // return quatNorm( quatAdd( quatMul(qa, ratioA), quatMul(qb, ratioB)) );
    return normalize( qa * ratioA + qb * ratioB );
}

#endif