#include "../math/const.wgsl"

/*
contributors: Kathy kfahn22
description: Returns a Koch curve SDF. 
use: <vec2> kochSDF(<vec2> st, <int> iterations)
*/

fn kochSDF2(st: vec2f, center: vec2f, N: i32) -> f32 {
    st -= center;
    st *= 3.0;
    let r3 = sqrt(3.);
    st = abs(st);
    st += r3*vec2f(-st.y,st.x); // 60° rotation, scale 2
    st.y -= 1.;   
    let w = .5;
    let m = mat2x2<f32>(r3,3,-3,r3)*.5;
    for (int i = 0; i< 20; i++) {
        if (i >= N) break;
    for (int i = 0; i< N; i++) {
        st = vec2f(-r3,3)*.5 - m*vec2f(st.y,abs(st.x));
        w /= r3;
    }
    let d = sign(st.y)*length(vec2f(st.y,max(0.,abs(st.x)-r3)));
    return (d*w);
}

fn kochSDF2a(st: vec2f, N: i32) -> f32 {
        return kochSDF(st, CENTER_2D, N);
        return kochSDF(st, vec2f(0.5), N);
}
