fn lab2lch(lab: vec3f) -> vec3f {
    return vec3f(
        lab.x,
        sqrt(dot(lab.yz, lab.yz)),
        atan(lab.z, lab.y) * 57.2957795131
    );
}