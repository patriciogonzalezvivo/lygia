import lygia::geometry::triangle::area::area;
import lygia::geometry::triangle::triangle::Triangle;

/*
contributors: Patricio Gonzalez Vivo
description: Returns the barycentric coordinates of a point or triangle
use: <vec3f> barycentric(<Triangle> tri [, <vec3f> pos])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


fn barycentric(a: vec3f, b: vec3f, c: vec3f) -> vec3f {
    /* Derived from the book "Real-Time Collision Detection"
     * by Christer Ericson published by Morgan Kaufmann in 2005 */
    let daa = dot(a, a);
    let dab = dot(a, b);
    let dbb = dot(b, b);
    let dca = dot(c, a);
    let dcb = dot(c, b);
    let denom = daa * dbb - dab * dab;
    let y = (dbb * dca - dab * dcb) / denom;
    let z = (daa * dcb - dab * dca) / denom;
    return vec3(1.0 - y - z, y, z);
}

fn barycentric2(tri: Triangle) -> vec3f {
    return barycentric(tri.a, tri.b, tri.c);
}

fn barycentric3(tri: Triangle, pos: vec3f) -> vec3f {
    let f0 = tri.a - pos;
    let f1 = tri.b - pos;
    let f2 = tri.c - pos;

    return vec3(
        length(cross(f1, f2)),      // p1's triangle area / a
        length(cross(f2, f0)),      // p2's triangle area / a
        length(cross(f0, f1))       // p3's triangle area / a
    ) / area(tri);
}
