/*
contributors: Ian Heisters
description: Transforms the input signal into a triangle wave. For instance, if x goes between 0 and 2, the returned value will go from 0 to 1, and then 1 to 0 in a triangle shape.
*/

fn mirror(x: f32) -> f32 {
    let f = fract(x);
    let m = floor(x % 2.0);
    let fm = f * m;
    return f + m - fm * 2.0;
}

fn mirror2(xy: vec2f) -> vec2f {
    let f = fract(xy);
    let m = floor(xy % 2.0);
    let fm = f * m;
    return f + m - fm * 2.0;
}
