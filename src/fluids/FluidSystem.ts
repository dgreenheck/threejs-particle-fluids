import { instancedArray, uniform } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import {
  Accumulator,
  NeighborList,
  assertRange,
  createSphKernelUniforms,
  type Material,
  type MaterialKernels,
  type ParticleRange,
  type ParticleSystem,
  type SolverContext,
} from '../core/index.js';
import { buildAdhesionKernel } from './sim/adhesion.js';
import { buildBoundaryVolumeKernel } from './sim/boundaryVolume.js';
import { buildCellSortKernels, type ParticleData } from './sim/cellSort.js';
import { buildColorFieldNormalKernel, buildSurfaceTensionKernel } from './sim/cohesion.js';
import { buildLambdaKernel } from './sim/lambda.js';
import { buildApplyDeltaKernel, buildPositionDeltaKernel } from './sim/positionDelta.js';
import type { FluidKernelContext } from './sim/shared.js';
import {
  buildVelocityApplyKernel,
  buildVelocityWalkKernel,
  buildVorticityGradientKernel,
  type ViscosityBuffers,
  type VorticityBuffers,
} from './sim/velocity.js';

export interface FluidSystemOptions {
  /** Particles that make up the fluid. Default: every particle in the system. */
  readonly range?: ParticleRange;
  /** Rest density in kg/m³. Default 1000 (water). */
  readonly restDensity?: number;
  /** Distance between particles at rest, in metres. Default 2 × particle radius. Sets each particle's mass. */
  readonly particleSpacing?: number;
  /** Radius particles interact within (SPH smoothing length). Default 2 × `particleSpacing`. */
  readonly smoothingRadius?: number;
  /** How much the fluid may compress (XPBD compliance). 0 is incompressible. Default 1e-4. */
  readonly compliance?: number;
  /**
   * XSPH viscosity: blends each particle's velocity with its neighbors'.
   * About 0.01 for water, up to ~0.3 for syrupy flow. For very thick
   * liquids use {@link ViscositySolver}.
   */
  readonly viscosity?: number;
  /** Vorticity confinement strength: puts back swirling motion damped by the solver. */
  readonly vorticity?: number;
  /** Surface tension (Akinci et al. 2013): pulls the fluid into drops and smooth sheets. */
  readonly surfaceTension?: number;
  /** Attraction toward boundary particles, which makes the fluid wet and cling to solids. */
  readonly adhesion?: number;
  /**
   * Re-sort the fluid's particles into neighbor-grid cell order at the start
   * of every step, so particles that are close in space are close in memory.
   * Kernels read each neighbor's data by particle index; once a liquid has
   * mixed, unsorted particles scatter those reads across the buffers.
   * Default off.
   *
   * A particle's index within `range` changes from step to step, and each
   * particle's state (position, velocity, density, …) moves with it. Leave
   * this off when anything reads or writes particular fluid particles by
   * index across steps, such as an emitter that releases particles by slot.
   */
  readonly sortByCell?: boolean;
}

/**
 * A liquid or gas simulated with Position Based Fluids (Macklin & Müller
 * 2013). Add it to a {@link SimLoop}'s `materials`.
 *
 * Optional effects (`viscosity`, `vorticity`, `surfaceTension`, `adhesion`)
 * are only compiled into the solver when given in the options. Once given,
 * even as 0, they can be changed at any time through the matching property.
 *
 * ```ts
 * const water = new FluidSystem(particles, { viscosity: 0.02, surfaceTension: 0.1 });
 * water.viscosity = 0.05;
 * ```
 *
 * Solids interact with the fluid through {@link addBoundary}.
 */
export class FluidSystem implements Material {
  readonly particles: ParticleSystem;
  readonly range: ParticleRange;
  readonly restDensity: number;
  readonly particleSpacing: number;
  readonly smoothingRadius: number;
  /** Mass of one particle: `restDensity · particleSpacing³`. */
  readonly mass: number;
  /** Density per particle, updated every solver iteration. */
  readonly density: StorageBufferNode<'float'>;
  readonly neighborRadius: number;

  private readonly options: FluidSystemOptions;
  private readonly uniforms: Partial<
    Record<'viscosity' | 'vorticity' | 'surfaceTension' | 'adhesion', UniformNode<'float', number>>
  > = {};
  private readonly boundaries: { range: ParticleRange; dynamic: boolean }[] = [];
  private readonly particleData: ParticleData[] = [];
  /** 1 once the neighbor grid has been built, so the cell order is valid. */
  private readonly sortReady = uniform(0, 'uint' as 'float') as unknown as UniformNode<
    'uint',
    number
  >;
  private stepsStarted = 0;
  private context: FluidKernelContext | undefined;

  constructor(particles: ParticleSystem, options: FluidSystemOptions = {}) {
    const range = options.range ?? { start: 0, count: particles.capacity };
    assertRange(particles, range, 'FluidSystem');
    this.particles = particles;
    this.range = range;
    this.options = options;
    this.restDensity = positive(options.restDensity ?? 1000, 'restDensity');
    this.particleSpacing = positive(
      options.particleSpacing ?? 2 * particles.particleRadius,
      'particleSpacing',
    );
    this.smoothingRadius = positive(
      options.smoothingRadius ?? 2 * this.particleSpacing,
      'smoothingRadius',
    );
    this.neighborRadius = this.smoothingRadius;
    const compliance = options.compliance ?? 1e-4;
    if (!(compliance >= 0))
      throw new Error(`FluidSystem: compliance must be ≥ 0, got ${compliance}`);
    this.mass = this.restDensity * this.particleSpacing ** 3;

    for (const key of ['viscosity', 'vorticity', 'surfaceTension', 'adhesion'] as const) {
      const value = options[key];
      if (value === undefined) continue;
      this.uniforms[key] = uniform(finite(value, key), 'float');
    }

    // The fluid owns its particles' mass.
    particles.setInvMass(range, 1 / this.mass);

    // Surface tension divides by density before the first iteration has computed it.
    this.density = instancedArray(particles.capacity, 'float');
    (this.density.value.array as Float32Array).fill(
      this.restDensity,
      range.start,
      range.start + range.count,
    );
  }

  /** XSPH viscosity. Requires `viscosity` in the options. */
  get viscosity(): number {
    return this.uniform('viscosity').value;
  }
  set viscosity(value: number) {
    this.uniform('viscosity').value = finite(value, 'viscosity');
  }

  /** Vorticity confinement strength. Requires `vorticity` in the options. */
  get vorticity(): number {
    return this.uniform('vorticity').value;
  }
  set vorticity(value: number) {
    this.uniform('vorticity').value = finite(value, 'vorticity');
  }

  /** Surface tension strength. Requires `surfaceTension` in the options. */
  get surfaceTension(): number {
    return this.uniform('surfaceTension').value;
  }
  set surfaceTension(value: number) {
    this.uniform('surfaceTension').value = finite(value, 'surfaceTension');
  }

  /** Adhesion strength. Requires `adhesion` in the options. */
  get adhesion(): number {
    return this.uniform('adhesion').value;
  }
  set adhesion(value: number) {
    this.uniform('adhesion').value = finite(value, 'adhesion');
  }

  /**
   * Treat the particles in `range` as a solid boundary (Akinci et al. 2012):
   * the fluid can't pass through them, pushes on them (buoyancy), and wets
   * them when `adhesion` is set. Use the surface particles of a soft body or
   * a cloth. Call before creating the {@link SimLoop}. Boundaries may not
   * overlap the fluid or each other.
   *
   * @param options.dynamic Recompute the boundary each substep because it
   *   moves or deforms. Default `true`; pass `false` for static geometry.
   */
  addBoundary(range: ParticleRange, options: { readonly dynamic?: boolean } = {}): void {
    if (this.context) {
      throw new Error('FluidSystem.addBoundary: add boundaries before creating the SimLoop');
    }
    assertRange(this.particles, range, 'FluidSystem.addBoundary');
    if (overlaps(range, this.range)) {
      throw new Error('FluidSystem.addBoundary: a boundary cannot overlap the fluid');
    }
    if (this.boundaries.some((boundary) => overlaps(range, boundary.range))) {
      throw new Error('FluidSystem.addBoundary: boundaries cannot overlap each other');
    }
    this.boundaries.push({ range, dynamic: options.dynamic ?? true });
  }

  /**
   * @internal Register a per-particle buffer that belongs to this fluid's
   * particles, so {@link FluidSystemOptions.sortByCell} moves it with them.
   * `local` buffers are indexed from 0 at `range.start`. Call before
   * creating the {@link SimLoop}.
   */
  addParticleData(data: ParticleData): void {
    if (this.context) {
      throw new Error('FluidSystem.addParticleData: register data before creating the SimLoop');
    }
    this.particleData.push(data);
  }

  /** @internal Called by {@link SimLoop.step}. */
  update(): void {
    // The first step builds the grid; from the second on its cell order is valid.
    if (this.stepsStarted++ === 1) this.sortReady.value = 1;
  }

  /**
   * True if, at the last neighbor rebuild, some fluid particle had more than
   * 64 neighbors within `smoothingRadius` and the extras were dropped.
   * Stalls on the GPU, so use it for debugging and tests.
   */
  async readbackOverflow(): Promise<boolean> {
    return this.kernelContext.neighbors.readbackOverflow();
  }

  /** @internal Kernel inputs, available once the fluid is in a {@link SimLoop}. */
  get kernelContext(): FluidKernelContext {
    if (!this.context) throw new Error('FluidSystem: add the fluid to a SimLoop first');
    return this.context;
  }

  build({ particles, dt, hashGrid }: SolverContext): MaterialKernels {
    const range = this.range;
    const sph = createSphKernelUniforms(this.smoothingRadius);
    const neighbors = new NeighborList(particles, range);
    const context: FluidKernelContext = {
      particles,
      range,
      neighbors,
      sph,
      dt,
      restDensity: uniform(this.restDensity, 'float'),
      particleVolume: uniform(this.particleSpacing ** 3, 'float'),
      mass: uniform(this.mass, 'float'),
    };
    this.context = context;
    const vec4Buffer = () => instancedArray(particles.capacity, 'vec4');

    const init: ComputeNode[] = [];
    const preSolve: ComputeNode[] = [];
    const volumeKernel = (dynamic: boolean, positions: StorageBufferNode<'vec4'>) => {
      const ranges = this.boundaries.filter((b) => b.dynamic === dynamic).map((b) => b.range);
      return ranges.length > 0
        ? [buildBoundaryVolumeKernel({ particles, grid: hashGrid, sph, ranges, positions })]
        : [];
    };
    // Dynamic boundaries are recomputed every substep, before anything reads them.
    preSolve.push(...volumeKernel(true, particles.predictedPositions));
    init.push(...volumeKernel(false, particles.positions));
    preSolve.push(...neighbors.buildKernels(hashGrid, sph.hSq));

    // Surface tension and adhesion change velocities once per substep, and
    // move predicted positions to match, before the pressure solve.
    const { surfaceTension, adhesion } = this.uniforms;
    if (surfaceTension || adhesion) {
      const impulses = new Accumulator(particles, 50, 'fluidImpulses');
      if (surfaceTension) {
        const normal = vec4Buffer();
        preSolve.push(
          buildColorFieldNormalKernel(context, normal),
          buildSurfaceTensionKernel(context, {
            gamma: surfaceTension,
            normal,
            density: this.density,
            accumulator: impulses,
          }),
        );
      }
      if (adhesion)
        preSolve.push(buildAdhesionKernel(context, { beta: adhesion, accumulator: impulses }));
      preSolve.push(
        impulses.buildApplyKernel(
          [particles.velocities, { buffer: particles.predictedPositions, scale: dt }],
          range,
        ),
      );
    }

    // Pressure solve. With boundaries, the fluid's push on them is scattered
    // into `reaction` and applied right after the fluid's own correction,
    // over the span that covers every boundary range.
    const deltaX = vec4Buffer();
    const lambda = instancedArray(particles.capacity, 'float');
    const boundaryRanges = this.boundaries.map((boundary) => boundary.range);
    const reaction =
      boundaryRanges.length > 0
        ? { accumulator: new Accumulator(particles, 10, 'fluidReaction'), ranges: boundaryRanges }
        : undefined;
    const solve = [
      buildLambdaKernel(context, {
        compliance: uniform(this.options.compliance ?? 1e-4, 'float'),
        density: this.density,
        lambda,
      }),
      buildPositionDeltaKernel(context, { lambda, deltaX, ...(reaction ? { reaction } : {}) }),
      buildApplyDeltaKernel(context, deltaX),
      ...(reaction
        ? [
            reaction.accumulator.buildApplyKernel(
              [particles.predictedPositions],
              coveringRange(boundaryRanges),
            ),
          ]
        : []),
    ];

    const postSolve: ComputeNode[] = [];
    const vorticity: VorticityBuffers | undefined = this.uniforms.vorticity && {
      strength: this.uniforms.vorticity,
      omega: vec4Buffer(),
      omegaLength: instancedArray(particles.capacity, 'float'),
      eta: vec4Buffer(),
    };
    const viscosity: ViscosityBuffers | undefined = this.uniforms.viscosity && {
      c: this.uniforms.viscosity,
      deltaV: vec4Buffer(),
    };
    if (vorticity || viscosity) {
      const passes = { ...(vorticity ? { vorticity } : {}), ...(viscosity ? { viscosity } : {}) };
      postSolve.push(buildVelocityWalkKernel(context, passes));
      if (vorticity) postSolve.push(buildVorticityGradientKernel(context, vorticity));
      postSolve.push(buildVelocityApplyKernel(context, passes));
    }

    const beforeStep: ComputeNode[] = [];
    if (this.options.sortByCell) {
      beforeStep.push(
        ...buildCellSortKernels({
          particles,
          grid: hashGrid,
          range,
          data: [
            ...particles.perParticleBuffers,
            { buffer: this.density, type: 'float' },
            ...this.particleData,
          ],
          enabled: this.sortReady,
        }).kernels,
      );
    }

    return { init, beforeStep, preSolve, solve, postSolve, noSelfContacts: range };
  }

  private uniform(
    key: 'viscosity' | 'vorticity' | 'surfaceTension' | 'adhesion',
  ): UniformNode<'float', number> {
    const node = this.uniforms[key];
    if (!node) {
      throw new Error(
        `FluidSystem: pass \`${key}\` in the options to enable it before changing it`,
      );
    }
    return node;
  }
}

function overlaps(a: ParticleRange, b: ParticleRange): boolean {
  return a.start < b.start + b.count && b.start < a.start + a.count;
}

/** The smallest range that contains every range in `ranges`. */
function coveringRange(ranges: readonly ParticleRange[]): ParticleRange {
  const start = Math.min(...ranges.map((range) => range.start));
  const end = Math.max(...ranges.map((range) => range.start + range.count));
  return { start, count: end - start };
}

function finite(value: number, name: string): number {
  if (!Number.isFinite(value)) throw new Error(`FluidSystem: ${name} must be finite, got ${value}`);
  return value;
}

function positive(value: number, name: string): number {
  if (!(value > 0) || !Number.isFinite(value)) {
    throw new Error(`FluidSystem: ${name} must be positive, got ${value}`);
  }
  return value;
}
