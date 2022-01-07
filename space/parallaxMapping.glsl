/*
author: Patricio Gonzalez Vivp
description: get parallax mapping coordinates
use: parallaxMapping(<sampler2D> heightTex, <vec3> V, <vec2> T, <float> parallaxHeight) 
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_PARALLAXMAPPING
#define FNC_PARALLAXMAPPING

#ifndef PARALLAXMAPPING_SAMPLER_FNC
#define PARALLAXMAPPING_SAMPLER_FNC(POS_UV) texture2D(heightTex, POS_UV).r
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

#ifndef PARALLAX_SCALE
#define PARALLAX_SCALE 0.01
#endif

#ifndef PARALLAX_NUMSEARCHES
#define PARALLAX_NUMSEARCHES 10.0
#endif


//////////////////////////////////////////////////////
//  Implements Parallax Mapping technique
//  Returns modified texture coordinates, and last used depth
//
//  http://sunandblackcat.com/tipFullView.php?topicid=28

vec2 parallaxMapping_simple(in sampler2D heightTex, in vec3 V, in vec2 T, out float parallaxHeight) {

    // get depth for this fragment
    float initialHeight = PARALLAXMAPPING_SAMPLER_FNC(T);

    // calculate amount of offset for Parallax Mapping
    vec2 texCoordOffset = PARALLAX_SCALE * V.xy / V.z * initialHeight;

    // calculate amount of offset for Parallax Mapping With Offset Limiting
    texCoordOffset = PARALLAX_SCALE * V.xy * initialHeight;

    // retunr modified texture coordinates
    return T - texCoordOffset;
}


vec2 parallaxMapping_steep(in sampler2D heightTex, in vec3 V, in vec2 T, out float parallaxHeight) {

    // determine number of layers from angle between V and N
    const float minLayers = PARALLAX_NUMSEARCHES * 0.5;
    const float maxLayers = 15.0;
    float numLayers = mix(maxLayers, minLayers, abs(dot(vec3(0.0, 0.0, 1.0), V)));

    // height of each layer
    float layerHeight = 1.0 / numLayers;
    // depth of current layer
    float currentLayerHeight = 0.0;
    // shift of texture coordinates for each iteration
    vec2 dtex = PARALLAX_SCALE * V.xy / V.z / numLayers;

    // current texture coordinates
    vec2 currentTextureCoords = T;

    // get first depth from heightmap
    float heightFromTexture = PARALLAXMAPPING_SAMPLER_FNC(currentTextureCoords);

    // while point is above surface
    while(heightFromTexture > currentLayerHeight) {
        // to the next layer
        currentLayerHeight += layerHeight;
        // shift texture coordinates along vector V
        currentTextureCoords -= dtex;
        // get new depth from heightmap
        heightFromTexture = PARALLAXMAPPING_SAMPLER_FNC(currentTextureCoords);
    }

    // return results
    parallaxHeight = currentLayerHeight;
    return currentTextureCoords;
}

vec2 parallaxMapping_relief(in sampler2D heightTex, in vec3 V, in vec2 T, out float parallaxHeight) {
    // determine required number of layers
    const float minLayers = PARALLAX_NUMSEARCHES;
    const float maxLayers = 15.0;
    float numLayers = mix(maxLayers, minLayers, abs(dot(vec3(0.0, 0.0, 1.0), V)));

    // height of each layer
    float layerHeight = 1.0 / numLayers;
    // depth of current layer
    float currentLayerHeight = 0.0;
    // shift of texture coordinates for each iteration
    vec2 dtex = PARALLAX_SCALE * V.xy / V.z / numLayers;

    // current texture coordinates
    vec2 currentTextureCoords = T;

    // depth from heightmap
    float heightFromTexture = PARALLAXMAPPING_SAMPLER_FNC(currentTextureCoords);

    // while point is above surface
    while (heightFromTexture > currentLayerHeight) {
        // go to the next layer
        currentLayerHeight += layerHeight; 
        // shift texture coordinates along V
        currentTextureCoords -= dtex;
        // new depth from heightmap
        heightFromTexture = PARALLAXMAPPING_SAMPLER_FNC(currentTextureCoords);
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
        heightFromTexture = PARALLAXMAPPING_SAMPLER_FNC(currentTextureCoords);

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
vec2 parallaxMapping_occlusion(in sampler2D heightTex, in vec3 V, in vec2 T, out float parallaxHeight) {
    // determine optimal number of layers
    const float minLayers = PARALLAX_NUMSEARCHES;
    const float maxLayers = 15.0;

    float numLayers = mix(maxLayers, minLayers, abs(dot(vec3(0.0, 0.0, 1.0), V)));

    // height of each layer
    float layerHeight = 1.0 / numLayers;
    // current depth of the layer
    float curLayerHeight = 0.0;
    // shift of texture coordinates for each layer
    vec2 dtex = PARALLAX_SCALE * V.xy / V.z / numLayers;

    // current texture coordinates
    vec2 currentTextureCoords = T;

    // depth from heightmap
    float heightFromTexture = PARALLAXMAPPING_SAMPLER_FNC(currentTextureCoords);

    // while point is above the surface
    while(heightFromTexture > curLayerHeight) {
        // to the next layer
        curLayerHeight += layerHeight; 
        // shift of texture coordinates
        currentTextureCoords -= dtex;
        // new depth from heightmap
        heightFromTexture = PARALLAXMAPPING_SAMPLER_FNC(currentTextureCoords);
    }

    ///////////////////////////////////////////////////////////
    // vec3 L = normalize(v_toLightInTangentSpace);
    // vec2 texStep	= PARALLAX_SCALE * L.xy / L.z / numLayers;

    // previous texture coordinates
    vec2 prevTCoords = currentTextureCoords;// + texStep;

    // heights for linear interpolation
    float nextH	= heightFromTexture - curLayerHeight;
    float prevH	= PARALLAXMAPPING_SAMPLER_FNC(prevTCoords) - curLayerHeight + layerHeight;

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

vec2 parallaxMapping(in sampler2D heightTex, in vec3 V, in vec2 T, out float parallaxHeight) {
    return PARALLAXMAPPING_FNC(heightTex, V, T, parallaxHeight);
}

vec2 parallaxMapping(in sampler2D heightTex, in vec3 V, in vec2 T) {
    float height;
    return PARALLAXMAPPING_FNC(heightTex, V, T, height);
}

#endif