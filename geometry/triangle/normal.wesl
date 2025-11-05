
import lygia::geometry::triangle::triangle::Triangle;

/*
contributors: Patricio Gonzalez Vivo
description: Returns the normal of a triangle
use: <vec3f> normal(<Triangle> tri)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


fn normal(tri: Triangle) -> vec3f {
    return normalize(cross(tri.b - tri.a, tri.c - tri.a));
}
