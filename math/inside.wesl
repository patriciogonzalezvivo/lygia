/*
contributors: Patricio Gonzalez Vivo
description: does the position lie within the triangle
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn inside(_x: f32, _min: f32, _max: f32) -> bool { return !(_x < _min || _x > _max); }

fn inside2(_v: vec2f, _min: vec2f, _max: vec2f) -> bool {
    return !(_v.x < _min.x || _v.x > _max.x || 
             _v.y < _min.y || _v.y > _max.y);
}

fn inside3(_v: vec3f, _min: vec3f, _max: vec3f) -> bool {
    return !(_v.x < _min.x || _v.x > _max.x || 
             _v.y < _min.y || _v.y > _max.y ||
             _v.z < _min.z || _v.z > _max.z);
}

fn insideAABB(_v: vec2f, _aabb: vec4f) -> bool { return inside(_v, _aabb.xy, _aabb.zw); }
