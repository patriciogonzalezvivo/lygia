fn lch2lab(lch: vec3f) -> vec3f {
    return vec3f(
        lch.x,
        lch.y * cos(lch.z * 0.01745329251),
        lch.y * sin(lch.z * 0.01745329251)
    );
}