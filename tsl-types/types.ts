import type { overloadingFn } from 'three/tsl'
import type * as THREE from 'three/webgpu'

type Prettify<T> = {
  [K in keyof T]: T[K]
} & {}
export type OverloadFnParams = Prettify<Parameters<typeof overloadingFn>[0]>

export type tsl_float = THREE.Node<'float'>

export type tsl_vec2 = THREE.Node<'vec2'>
export type tsl_vec3 = THREE.Node<'vec3'>
export type tsl_vec4 = THREE.Node<'vec4'>

export type tsl_mat2 = THREE.Node<'mat2'>
export type tsl_mat3 = THREE.Node<'mat3'>
export type tsl_mat4 = THREE.Node<'mat4'>
