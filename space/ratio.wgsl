fn ratio(st: vec2f, s: vec2f) -> vec2f {
    return mix( vec2f((st.x*s.x/s.y)-(s.x*.5-s.y*.5)/s.y,st.y),
                vec2f(st.x,st.y*(s.y/s.x)-(s.y*.5-s.x*.5)/s.x),
                step(s.x,s.y));
}