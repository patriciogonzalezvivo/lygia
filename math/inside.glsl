/*
contributors: Patricio Gonzalez Vivo
description: does the position lie within the triangle
use:
    - bool inside(<float|vec2|vec3> value, <float|vec2|vec3> min, <float|vec2|vec3> max)
    - bool inside(<vec2|vec3> value, <vec4|AABB> aabb)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_INSIDE
#define FNC_INSIDE
bool inside(float _x, float _min, float _max) {
    return !(_x < _min || _x > _max);
}

bool inside(vec2 _v, vec2 _min, vec2 _max) {
    return !(_v.x < _min.x || _v.x > _max.x || 
             _v.y < _min.y || _v.y > _max.y);
}

bool inside(vec3 _v, vec3 _min, vec3 _max) {
    return !(_v.x < _min.x || _v.x > _max.x || 
             _v.y < _min.y || _v.y > _max.y ||
             _v.z < _min.z || _v.z > _max.z);
}

bool inside(vec2 _v, vec4 _aabb) {
    return inside(_v, _aabb.xy, _aabb.zw);
}

#ifdef STR_AABB
bool inside(vec3 _v, AABB _aabb) {
    return inside(_v, _aabb.min, _aabb.max);
}
#endif

#endif