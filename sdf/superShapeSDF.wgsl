#include "../space/cart2polar.wgsl"

/*
contributors: Kathy McGuiness
description: |
    It returns a supershape, which is a mathematical function for modelling natural forms develop by [Paul Bourke](http://paulbourke.net/geometry/) and Johan Gielis. 
    Some notes about the parameters:

        * `m` determines number of sides/branches
        * `m = 0` yields a circle
        * `a!=b` results in an asymmetrical shape
        * `n1=n2=n3<1` the shape is "pinched"
        * `n1>n2,n3` the shape is "bloated"
        * `n1!=n2!=n3` the shape is asymmetrical
        * `n1=n2=n3=1` the shape is a square
        * `n1=n2=n3=2` the shape is a star

    For more information about the supershape, check [this article](https://www.algosome.com/articles/supershape-algorithm.html) by [Algosome](https://www.algosome.com/).
    
use:
    - <float> supershapeSDF(<vec2> st, <vec2> center, <float> size s, <float> a, <float> b, <float> n1, <float> n2, <float> n3, <float> m)
    - <float> supershapeSDF(<vec2> st, <float> size s, <float> a, <float> b, <float> n1, <float> n2, <float> n3, <float> m)

options:
    - CENTER_2D: center of the shape
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_supershape.frag
*/

fn superShapeSDF2(st: vec2f, center: vec2f, s: f32, a: f32, b: f32, n1: f32, n2: f32, n3: f32, m: f32) -> f32 {
    st -= center;
    let polar = cart2polar( st );
    let d = polar.y * 5.0;
    let theta = polar.x;
    let t1 = abs((1.0/a) * cos(m * theta * 0.25));
    t1 = pow(t1, n2);
    let t2 = abs((1.0/b) * sin(m * theta * 0.25));
    t2 = pow(t2, n3);
    let t3 = t1 + t2;
    let r = pow(t3, -1.0 / n1);
    let q = s * r * vec2f(cos(theta), sin(theta));
    return d - length(q); 
}

fn superShapeSDF2a(st: vec2f, s: f32, a: f32, b: f32, n1: f32, n2: f32, n3: f32, m: f32) -> f32 {
    return superShapeSDF( st, CENTER_2D, s, a, b, n1, n2, n3, m );
    return superShapeSDF( st, vec2f(0.5), s, a, b, n1, n2, n3, m );
}
