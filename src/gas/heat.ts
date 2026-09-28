import type { Vector3 } from 'three';
import {
  Fn,
  If,
  atomicAdd,
  atomicLoad,
  atomicStore,
  exp,
  instanceIndex,
  instancedArray,
  uint,
  uniform,
  vec3,
  vec4,
} from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import type { FluidSystem } from '../fluids/index.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/** A spherical region that heats the air inside it to temperature 1. */
export interface HeatSource {
  /** Center of the region. Mutate it to move the source. */
  readonly position: Vector3;
  /** Radius in metres. */
  readonly radius: number;
}

/**
 * Temperatures are summed on the GPU as 32-bit fixed-point integers, 4096
 * steps per degree. Temperatures stay in [0, 1], so above about a million
 * air particles the scale shrinks to keep the sum from overflowing.
 */
export function fixedPointScale(count: number): number {
  return Math.max(1, Math.min(4096, Math.floor(0xffffffff / count)));
}

/**
 * Per-particle air temperature with Boussinesq buoyancy: air inside a heat
 * source is set to temperature 1, cools exponentially, and accelerates
 * upward by `buoyancy · (T − T̄)`, where `T̄` is the mean air temperature.
 * Lifting against the mean keeps the net force on the air zero, so the
 * column as a whole doesn't drift.
 */
export class AirHeat {
  readonly temperature: StorageBufferNode<'float'>;
  readonly buoyancy: UniformNode<'float', number>;
  readonly cooling: UniformNode<'float', number>;
  /** Storage buffers this heat model owns, for disposal. */
  readonly buffers: StorageBufferNode<'float' | 'uint'>[] = [];

  constructor(
    private readonly fluid: FluidSystem,
    private readonly sources: readonly HeatSource[],
    buoyancy: number,
    cooling: number,
  ) {
    this.temperature = instancedArray(fluid.range.count, 'float');
    this.buffers.push(this.temperature);
    // Temperature belongs to its air particle, so it moves with it when the air is sorted.
    fluid.addParticleData({ buffer: this.temperature, type: 'float', local: true });
    this.buoyancy = uniform(buoyancy, 'float');
    this.cooling = uniform(cooling, 'float');
  }

  build(dt: UniformNode<'float', number>): ComputeNode[] {
    const { particles, range } = this.fluid;
    const { temperature, buoyancy, cooling } = this;
    const total = (instancedArray(1, 'uint') as Any).setAtomic(true);
    this.buffers.push(total);
    const fixed = fixedPointScale(range.count);
    const sources = this.sources.map((source) => ({
      position: uniform(source.position),
      radiusSq: source.radius ** 2,
    }));

    const reset = Fn(() => {
      atomicStore(total.element(0), uint(0));
    })()
      .compute(1)
      .setName('heat.reset');
    const heat = Fn(() => {
      const k: Any = instanceIndex;
      const p: Any = particles.positions.element(k.add(uint(range.start))).xyz;
      const t: Any = temperature.element(k).toVar();
      for (const source of sources) {
        const offset: Any = p.sub(source.position);
        If(offset.dot(offset).lessThan(source.radiusSq), () => {
          t.assign(1);
        });
      }
      t.mulAssign(exp(cooling.mul(dt).negate()));
      temperature.element(k).assign(t);
      atomicAdd(total.element(0), t.mul(fixed).toUint());
    })()
      .compute(range.count)
      .setName('heat.heat');
    const lift = Fn(() => {
      const k: Any = instanceIndex;
      const velocity: Any = particles.velocities.element(k.add(uint(range.start)));
      const mean: Any = (atomicLoad(total.element(0)) as Any).toFloat().div(fixed * range.count);
      const dv: Any = buoyancy.mul(temperature.element(k).sub(mean)).mul(dt);
      velocity.assign(vec4(velocity.xyz.add(vec3(0, dv, 0)), velocity.w));
    })()
      .compute(range.count)
      .setName('heat.lift');
    return [reset, heat, lift];
  }
}
