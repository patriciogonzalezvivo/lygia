/*
contributors: Inigo Quiles
description: Cast a ray
*/

// #include "map.wgls"

fn raymarchCast(ro: vec3f, rd: vec3f ) -> f32 {
    var dist = -1.0;
    var t = -1.0;
    for (var i = 0; i < 120 && t < 100.0; i++) {
        let h = map( ro + rd * t );
        if (abs(h) < (0.001 * t)) { 
            dist = t; 
            break;
        }
        t += h;
    }
    return dist;
}