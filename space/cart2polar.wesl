/*
contributors: [Ivan Dianov, Kathy McGuiness]
description: cartesian to polar transformation.
use: <vec2|vec3> cart2polar(<vec2|vec3> st)
*/

fn cart2polar2(st: vec2f) -> vec2f {
    return vec2f(atan2(st.y, st.x), length(st));
}

// https://mathworld.wolfram.com/SphericalCoordinates.html
fn cart2polar3(st: vec3f) -> vec3f {
    let r = length(st);
    let phi = acos(st.z / r);
    let theta = atan2(st.y, st.x);
    return vec3f(r, phi, theta);
}
