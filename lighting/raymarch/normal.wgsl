/*
contributors: Inigo Quiles
description: Calculate normals http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
examples:
    - /shaders/lighting_raymarching.frag

*/

// #include "map.wgls"

fn raymarchNormal( pos: vec3f ) -> vec3f {
    let eps = 0.002;
    let v1 = vec3f( 1.0,-1.0,-1.0);
    let v2 = vec3f(-1.0,-1.0, 1.0);
    let v3 = vec3f(-1.0, 1.0,-1.0);
    let v4 = vec3f( 1.0, 1.0, 1.0);
    return normalize( v1 * map( pos + v1 * eps ) + 
                      v2 * map( pos + v2 * eps ) + 
                      v3 * map( pos + v3 * eps ) + 
                      v4 * map( pos + v4 * eps ) );
}