/*
contributors: Inigo Quiles
description: |
    Remapping the unit interval into the unit interval by expanding the sides and compressing the center, and keeping 1/2 mapped to 1/2, that can be done with the gain() function. From https://iquilezles.org/articles/functions/
*/

fn gain(x: f32, k: f32) -> f32 {
    let a = 0.5 * pow(2.0 * select(1.0-x, x, x<0.5), k);
    return select(1.0-a, a, x<0.5);
}