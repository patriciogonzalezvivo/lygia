/*
contributors:  Shadi El Hajj
description: Structure to hold shading variables
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

struct ShadingData {
   var V: vec3f;
   var N: vec3f;
   var H: vec3f;
   var L: vec3f;
   var R: vec3f;

   var NoV: f32;
   var NoL: f32;
   var NoH: f32;

   var roughness: f32;
   var linearRoughness: f32;
   var diffuseColor: vec3f;
   var specularColor: vec3f;

   var energyCompensation: vec3f;

   var directDiffuse: vec3f;
   var directSpecular: vec3f;
   var indirectDiffuse: vec3f;
   var indirectSpecular: vec3f;
};
