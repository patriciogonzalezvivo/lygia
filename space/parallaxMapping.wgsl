#include "../sampler.wgsl"

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

// #define PARALLAXMAPPING_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).r

// #define PARALLAXMAPPING_FNC parallaxMapping_steep
// #define PARALLAXMAPPING_FNC parallaxMapping_relief
// #define PARALLAXMAPPING_FNC parallaxMapping_occlusion
// #define PARALLAXMAPPING_FNC parallaxMapping_simple

const PARALLAXMAPPING_SCALE: f32 = 0.01;

const PARALLAXMAPPING_NUMSEARCHES: f32 = 10.0;

//////////////////////////////////////////////////////
//  Implements Parallax Mapping technique
//  Returns modified texture coordinates, and last used depth
//
//  http://sunandblackcat.com/tipFullView.php?topicid=28

fn parallaxMapping_simple(tex: SAMPLER_TYPE, V: vec3f, T: vec2f, parallaxHeight: f32) -> vec2f {

    // get depth for this fragment
    let initialHeight = PARALLAXMAPPING_SAMPLER_FNC(tex, T);

    // calculate amount of offset for Parallax Mapping
    let texCoordOffset = PARALLAXMAPPING_SCALE * V.xy / V.z * initialHeight;

    // calculate amount of offset for Parallax Mapping With Offset Limiting
    texCoordOffset = PARALLAXMAPPING_SCALE * V.xy * initialHeight;

    // return modified texture coordinates
    return T - texCoordOffset;
}

fn parallaxMapping_steep(tex: SAMPLER_TYPE, V: vec3f, T: vec2f, parallaxHeight: f32) -> vec2f {

    // determine number of layers from angle between V and N
    let minLayers = PARALLAXMAPPING_NUMSEARCHES * 0.5;
    let maxLayers = 15.0;
    let numLayers = mix(maxLayers, minLayers, abs(dot(vec3f(0.0, 0.0, 1.0), V)));

    // height of each layer
    let layerHeight = 1.0 / numLayers;
    // depth of current layer
    let currentLayerHeight = 0.0;
    // shift of texture coordinates for each iteration
    let dtex = PARALLAXMAPPING_SCALE * V.xy / V.z / numLayers;

    // current texture coordinates
    let currentTextureCoords = T;

    // get first depth from heightmap
    let heightFromTexture = PARALLAXMAPPING_SAMPLER_FNC(tex, currentTextureCoords);

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

fn parallaxMapping_relief(tex: SAMPLER_TYPE, V: vec3f, T: vec2f, parallaxHeight: f32) -> vec2f {
    // determine required number of layers
    let minLayers = PARALLAXMAPPING_NUMSEARCHES;
    let maxLayers = 15.0;
    let numLayers = mix(maxLayers, minLayers, abs(dot(vec3f(0.0, 0.0, 1.0), V)));

    // height of each layer
    let layerHeight = 1.0 / numLayers;
    // depth of current layer
    let currentLayerHeight = 0.0;
    // shift of texture coordinates for each iteration
    let dtex = PARALLAXMAPPING_SCALE * V.xy / V.z / numLayers;

    // current texture coordinates
    let currentTextureCoords = T;

    // depth from heightmap
    let heightFromTexture = PARALLAXMAPPING_SAMPLER_FNC(tex, currentTextureCoords);

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
    let deltaTexCoord = dtex * 0.5;
    let deltaHeight = layerHeight * 0.5;

    // return to the mid point of previous layer
    currentTextureCoords += deltaTexCoord;
    currentLayerHeight -= deltaHeight;

    // binary search to increase precision of Steep Paralax Mapping
    let numSearches = 5;
    for (int i = 0; i < numSearches; i++) {
        // decrease shift and height of layer by half
        deltaTexCoord *= 0.5;
        deltaHeight *= 0.5;

        // new depth from heightmap
        heightFromTexture = PARALLAXMAPPING_SAMPLER_FNC(tex, currentTextureCoords);

        // shift along or against vector V
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

fn parallaxMapping_occlusion(tex: SAMPLER_TYPE, V: vec3f, T: vec2f, parallaxHeight: f32) -> vec2f {
    // determine optimal number of layers
    let minLayers = PARALLAXMAPPING_NUMSEARCHES;
    let maxLayers = 15.0;

    let numLayers = mix(maxLayers, minLayers, abs(dot(vec3f(0.0, 0.0, 1.0), V)));

    // height of each layer
    let layerHeight = 1.0 / numLayers;
    // current depth of the layer
    let curLayerHeight = 0.0;
    // shift of texture coordinates for each layer
    let dtex = PARALLAXMAPPING_SCALE * V.xy / V.z / numLayers;

    // current texture coordinates
    let currentTextureCoords = T;

    // depth from heightmap
    let heightFromTexture = PARALLAXMAPPING_SAMPLER_FNC(tex, currentTextureCoords);

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
    let prevTCoords = currentTextureCoords;// + texStep;

    // heights for linear interpolation
    let nextH = heightFromTexture - curLayerHeight;
    let prevH = PARALLAXMAPPING_SAMPLER_FNC(tex, prevTCoords) - curLayerHeight + layerHeight;

    // proportions for linear interpolation
    let weight = nextH / (nextH - prevH);

    // interpolation of texture coordinates
    let finalTexCoords = prevTCoords * weight + currentTextureCoords * (1.0 - weight);

    // interpolation of depth values
    parallaxHeight = curLayerHeight + prevH * weight + nextH * (1.0 - weight);

    // return result
    return finalTexCoords;
}

fn parallaxMapping(tex: SAMPLER_TYPE, V: vec3f, T: vec2f, parallaxHeight: f32) -> vec2f {
    return PARALLAXMAPPING_FNC(tex, V, T, parallaxHeight);
}

fn parallaxMappinga(tex: SAMPLER_TYPE, V: vec3f, T: vec2f) -> vec2f {
    var height: f32;
    return PARALLAXMAPPING_FNC(tex, V, T, height);
}
