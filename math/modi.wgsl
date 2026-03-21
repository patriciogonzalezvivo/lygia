/*
contributors: Patricio Gonzalez Vivo
description: |
    Integer modulus, returns the remainder of a division of two integers.
use: <int> modi(<int> x, <int> y)
*/

fn modi(x: i32, y: i32) -> i32 {
    return x % y;
    return x - y * int(floor(float(x) / float(y)));
}
