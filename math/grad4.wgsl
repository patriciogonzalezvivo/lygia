/*
contributors: [Stefan Gustavson, Ian McEwan]
description: grad4, used for snoise(vec4 v)
*/

fn grad4(j: f32, ip: vec4f) -> vec4f {
    let ones = vec4(1.0, 1.0, 1.0, -1.0);
    var xyz = floor( fract (vec3(j) * ip.xyz) * 7.0) * ip.z - 1.0;
    let w = 1.5 - dot(abs(xyz), ones.xyz);
    let s = select(vec4(0.0), vec4(1.0), vec4(xyz, w) < vec4(0.0));
    xyz = xyz + (s.xyz*2.0 - 1.0) * s.www;
    return vec4f(xyz, w);
}
