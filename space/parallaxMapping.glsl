#include "../sampler.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: get parallax mapping coordinates
use: parallaxMapping(<SAMPLER_TYPE> tex, <vec3> V, <vec2> T, <float> parallaxHeight)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - PARALLAXMAPPING_FNC()
    - PARALLAXMAPPING_SAMPLER_FNC(UV)
    - PARALLAXMAPPING_SCALE
    - PARALLAXMAPPING_NUMSEARCHES
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef PARALLAXMAPPING_SAMPLER_FNC
#define PARALLAXMAPPING_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).r
#endif

#if defined(PARALLAXMAPPING_SEETP)
#define PARALLAXMAPPING_FNC parallaxMapping_steep
#elif defined(PARALLAXMAPPING_RELIEF)
#define PARALLAXMAPPING_FNC parallaxMapping_relief
#elif defined(PARALLAXMAPPING_OCCLUSION)
#define PARALLAXMAPPING_FNC parallaxMapping_occlusion
#else
#define PARALLAXMAPPING_FNC parallaxMapping_simple
#endif

#ifndef PARALLAXMAPPING_SCALE
#define PARALLAXMAPPING_SCALE 0.01
#endif

#ifndef PARALLAXMAPPING_NUMSEARCHES
#define PARALLAXMAPPING_NUMSEARCHES 10.0
#endif

#ifndef FNC_PARALLAXMAPPING
#define FNC_PARALLAXMAPPING

//////////////////////////////////////////////////////
//  Implements Parallax Mapping technique
//  Returns modified texture coordinates, and last used depth
//
//  http://sunandblackcat.com/tipFullView.php?topicid=28

vec2 parallaxMapping_simple(in SAMPLER_TYPE tex, in vec3 V, in vec2 T, out float parallaxHeight) {

    // get depth for this fragment
    float initialHeight = PARALLAXMAPPING_SAMPLER_FNC(tex, T);

    // calculate amount of offset for Parallax Mapping
    vec2 texCoordOffset = PARALLAXMAPPING_SCALE * V.xy / V.z * initialHeight;

    // calculate amount of offset for Parallax Mapping With Offset Limiting
    texCoordOffset = PARALLAXMAPPING_SCALE * V.xy * initialHeight;

    // retunr modified texture coordinates
    return T - texCoordOffset;
}


vec2 parallaxMapping_steep(in SAMPLER_TYPE tex, in vec3 V, in vec2 T, out float parallaxHeight) {

    // determine number of layers from angle between V and N
    const float minLayers = PARALLAXMAPPING_NUMSEARCHES * 0.5;
    const float maxLayers = 15.0;
    float numLayers = mix(maxLayers, minLayers, abs(dot(vec3(0.0, 0.0, 1.0), V)));

    // height of each layer
    float layerHeight = 1.0 / numLayers;
    // depth of current layer
    float currentLayerHeight = 0.0;
    // shift of texture coordinates for each iteration
    vec2 dtex = PARALLAXMAPPING_SCALE * V.xy / V.z / numLayers;

    // current texture coordinates
    vec2 currentTextureCoords = T;

    // get first depth from heightmap
    float heightFromTexture = PARALLAXMAPPING_SAMPLER_FNC(tex, currentTextureCoords);

    // while point is above surface
    while(heightFromTexture > currentLayerHeight) {
        // to the next layer
        currentLayerHeight += layerHeight;
        // shift texture coordinates along vector V
        currentTextureCoords -= dtex;
        // get new depth from heightmap
        heightFromTexture = PARALLAXMAPPING_SAMPLER_FNC(tex, currentTextureCoords);
    }

    // return results
    parallaxHeight = currentLayerHeight;
    return currentTextureCoords;
}

vec2 parallaxMapping_relief(in SAMPLER_TYPE tex, in vec3 V, in vec2 T, out float parallaxHeight) {
    // determine required number of layers
    const float minLayers = PARALLAXMAPPING_NUMSEARCHES;
    const float maxLayers = 15.0;
    float numLayers = mix(maxLayers, minLayers, abs(dot(vec3(0.0, 0.0, 1.0), V)));

    // height of each layer
    float layerHeight = 1.0 / numLayers;
    // depth of current layer
    float currentLayerHeight = 0.0;
    // shift of texture coordinates for each iteration
    vec2 dtex = PARALLAXMAPPING_SCALE * V.xy / V.z / numLayers;

    // current texture coordinates
    vec2 currentTextureCoords = T;

    // depth from heightmap
    float heightFromTexture = PARALLAXMAPPING_SAMPLER_FNC(tex, currentTextureCoords);

    // while point is above surface
    while (heightFromTexture > currentLayerHeight) {
        // go to the next layer
        currentLayerHeight += layerHeight; 
        // shift texture coordinates along V
        currentTextureCoords -= dtex;
        // new depth from heightmap
        heightFromTexture = PARALLAXMAPPING_SAMPLER_FNC(tex, currentTextureCoords);
    }

    ///////////////////////////////////////////////////////////
    // Start of Relief Parallax Mapping

    // decrease shift and height of layer by half
    vec2 deltaTexCoord = dtex * 0.5;
    float deltaHeight = layerHeight * 0.5;

    // return to the mid point of previous layer
    currentTextureCoords += deltaTexCoord;
    currentLayerHeight -= deltaHeight;

    // binary search to increase precision of Steep Paralax Mapping
    const int numSearches = 5;
    for (int i = 0; i < numSearches; i++) {
        // decrease shift and height of layer by half
        deltaTexCoord *= 0.5;
        deltaHeight *= 0.5;

        // new depth from heightmap
        heightFromTexture = PARALLAXMAPPING_SAMPLER_FNC(tex, currentTextureCoords);

        // shift along or agains vector V
        if(heightFromTexture > currentLayerHeight) // below the surface
        {
            currentTextureCoords -= deltaTexCoord;
            currentLayerHeight += deltaHeight;
        }
        else // above the surface
        {
            currentTextureCoords += deltaTexCoord;
            currentLayerHeight -= deltaHeight;
        }
    }

    // return results
    parallaxHeight = currentLayerHeight;    
    return currentTextureCoords;
}


#if defined(PARALLAXMAPPING_OCCLUSION)
vec2 parallaxMapping_occlusion(in SAMPLER_TYPE tex, in vec3 V, in vec2 T, out float parallaxHeight) {
    // determine optimal number of layers
    const float minLayers = PARALLAXMAPPING_NUMSEARCHES;
    const float maxLayers = 15.0;

    float numLayers = mix(maxLayers, minLayers, abs(dot(vec3(0.0, 0.0, 1.0), V)));

    // height of each layer
    float layerHeight = 1.0 / numLayers;
    // current depth of the layer
    float curLayerHeight = 0.0;
    // shift of texture coordinates for each layer
    vec2 dtex = PARALLAXMAPPING_SCALE * V.xy / V.z / numLayers;

    // current texture coordinates
    vec2 currentTextureCoords = T;

    // depth from heightmap
    float heightFromTexture = PARALLAXMAPPING_SAMPLER_FNC(tex, currentTextureCoords);

    // while point is above the surface
    while(heightFromTexture > curLayerHeight) {
        // to the next layer
        curLayerHeight += layerHeight; 
        // shift of texture coordinates
        currentTextureCoords -= dtex;
        // new depth from heightmap
        heightFromTexture = PARALLAXMAPPING_SAMPLER_FNC(tex, currentTextureCoords);
    }

    ///////////////////////////////////////////////////////////
    // vec3 L = normalize(v_toLightInTangentSpace);
    // vec2 texStep	= PARALLAXMAPPING_SCALE * L.xy / L.z / numLayers;

    // previous texture coordinates
    vec2 prevTCoords = currentTextureCoords;// + texStep;

    // heights for linear interpolation
    float nextH	= heightFromTexture - curLayerHeight;
    float prevH	= PARALLAXMAPPING_SAMPLER_FNC(tex, prevTCoords) - curLayerHeight + layerHeight;

    // proportions for linear interpolation
    float weight = nextH / (nextH - prevH);

    // interpolation of texture coordinates
    vec2 finalTexCoords = prevTCoords * weight + currentTextureCoords * (1.0 - weight);

    // interpolation of depth values
    parallaxHeight = curLayerHeight + prevH * weight + nextH * (1.0 - weight);

    // return result
    return finalTexCoords;
}
#endif

vec2 parallaxMapping(in SAMPLER_TYPE tex, in vec3 V, in vec2 T, out float parallaxHeight) {
    return PARALLAXMAPPING_FNC(tex, V, T, parallaxHeight);
}

vec2 parallaxMapping(in SAMPLER_TYPE tex, in vec3 V, in vec2 T) {
    float height;
    return PARALLAXMAPPING_FNC(tex, V, T, height);
}

#endif