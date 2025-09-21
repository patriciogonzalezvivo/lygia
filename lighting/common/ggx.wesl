fn GGX(N: vec3f, H: vec3f, NoH: f32, roughness: f32) -> f32 {
    let NxH = cross(N, H);
    let oneMinusNoHSquared = dot(NxH, NxH);

    // let oneMinusNoHSquared = 1.0 - NoH * NoH;

    let a = NoH * roughness;
    let k = roughness / (oneMinusNoHSquared + a * a);
    let d = (k * k) * 0.31830988618379067153776752674503; // 1/PI
    return min(d, 65504.0);
}
