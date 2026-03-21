/*
contributors: [Ivan Dianov, Shadi El Hajj]
description: polar to cartesian conversion.
use: polar2cart(<vec2> polar)
*/

fn polar2cart2(polar: vec2f) -> vec2f {
    return vec2f(cos(polar.x), sin(polar.x)) * polar.y;
}

// https://mathworld.wolfram.com/SphericalCoordinates.html
fn polar2cart(r: f32, phi: f32, theta: f32) -> vec3f {
    let x = r * cos(theta) * sin(phi);
    let y = r * sin(theta) * sin(phi);
    let z = r * cos(phi);
    return vec3f(x, y, z);
}
