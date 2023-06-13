fn ratio(st: vec2<f32>, s: vec2<f32>) -> vec2<f32> {
    return mix( vec2<f32>((st.x*s.x/s.y)-(s.x*.5-s.y*.5)/s.y,st.y),
                vec2<f32>(st.x,st.y*(s.y/s.x)-(s.y*.5-s.x*.5)/s.x),
                step(s.x,s.y));
}