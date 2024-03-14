/*
contributors: Inigo Quiles
description: | 
    Remapping the unit interval into the unit interval by expanding the sides and compressing the center, and keeping 1/2 mapped to 1/2, that can be done with the gain() function. From https://iquilezles.org/articles/functions/
use: <float> gain(<float> x, <float> k)
*/

#ifndef FNC_GAIN
#define FNC_GAIN
float gain(float x, float k) {
    const float a = 0.5*pow(2.0*((x<0.5)?x:1.0-x), k);
    return (x<0.5)?a:1.0-a;
}
#endif