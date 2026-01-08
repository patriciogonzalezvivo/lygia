/*
contributors: Patricio Gonzalez Vivo
description: |
    Integer modulus, returns the remainder of a division of two integers.
use: <int> modi(<int> x, <int> y)
*/

#ifndef FNC_MODI
#define FNC_MODI
int modi(int x, int y) {
#if __VERSION__ >= 130
    return x % y;
#else
    return x - y * int(floor(float(x) / float(y)));
#endif
}
#endif