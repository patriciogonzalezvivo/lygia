#ifndef FNC_TEXTURESHADOW
#define FNC_TEXTURESHADOW

float textureShadow(const sampler2D _shadowMap, in vec4 _coord) {
    vec3 shadowCoord = _coord.xyz / _coord.w;
    return texture2D(_shadowMap, shadowCoord.xy).r;
}

float textureShadow(const sampler2D _shadowMap, in vec3 _coord) {
    return textureShadow(_shadowMap, vec4(_coord, 1.0));
}

float textureShadow(const sampler2D depths, vec2 uv, float compare){
    return step(compare, texture2D(depths, uv).r );
}

#endif