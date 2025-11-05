
import lygia::geometry::triangle::triangle::Triangle;

/*
contributors: Patricio Gonzalez Vivo
description: Returns the centroid of a triangle
use: <vec3f> centroid(<Triangle> tri)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


fn centroid(tri: Triangle) -> vec3f {
    return (tri.a + tri.b + tri.c) * 0.3333333333333;
}
