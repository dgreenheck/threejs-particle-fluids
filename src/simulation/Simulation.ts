import {
  Box3,
  Color,
  DoubleSide,
  Euler,
  MeshStandardMaterial,
  PlaneGeometry,
  Quaternion,
  Vector3,
  type BufferGeometry,
  type Camera,
  type InstancedMesh,
  type Mesh,
  type Object3D,
  type Scene,
} from 'three';
import { MeshPhysicalNodeMaterial, type WebGPURenderer } from 'three/webgpu';

import { ClothSystem, createClothGraph, createClothSurface } from '../cloth/index.js';
import {
  FrameStepper,
  ParticleSystem,
  PrimitiveSet,
  SDFCollider,
  SimLoop,
  type Collider,
  type Material,
  type ParticleInit,
  type ParticleRange,
  type SDFData,
} from '../core/index.js';
import {
  FluidSurfaceRenderer,
  FluidSystem,
  ViscositySolver,
  type FluidAppearance,
} from '../fluids/index.js';
import { GasSystem, GasVolumeRenderer } from '../gas/index.js';
import { createParticleMesh } from '../render/particles.js';
import { bakeMeshToSdf } from '../sdf/index.js';
import { SoftbodyMesh, SoftbodySystem, voxelize, type SoftbodyDef } from '../softbody/index.js';
import {
  Cloth,
  Fluid,
  Smoke,
  Softbody,
  clothBendCompliance,
  softbodyCompliance,
} from './handles.js';
import {
  boxObstacle,
  capsuleObstacle,
  clearOf,
  fillBox,
  radiusForBudget,
  sdfObstacle,
  sphereObstacle,
  worldGeometry,
  type Obstacle,
} from './layout.js';

interface SimulationBaseOptions {
  /** A renderer from {@link createParticleRenderer}. */
  readonly renderer: WebGPURenderer;
  /** Scene the simulation draws into. Liquids reflect its environment map and main directional light. */
  readonly scene: Scene;
  readonly camera: Camera;
  /**
   * The box everything happens in. Walls keep particles inside it; the top
   * is open unless `closed` is set. Liquids and smoke are drawn inside it.
   * Smoke needs one, because the air fills it.
   */
  readonly container?: Box3;
  /** Put a lid on the container. Default `false`; smoke always gets one. */
  readonly closed?: boolean;
  /** Gravity in m/s². Default `(0, -9.81, 0)`, or `(0, -1, 0)` for smoke. Change it later with {@link Simulation.gravity}. */
  readonly gravity?: Vector3;
  /** Solver substeps per step. Default: chosen from the particle size and what's in the scene. */
  readonly substeps?: number;
}

/**
 * Options for {@link Simulation}. Give exactly one of `particles` or
 * `particleRadius`; every particle in a simulation shares one size.
 */
export type SimulationOptions = SimulationBaseOptions &
  (
    | {
        /**
         * The most particles to use in total. The simulation picks the
         * smallest particle size at which everything added fits within this
         * many, so it uses close to this many. Each fluid, soft body, and
         * cloth gets a share in proportion to its volume or area.
         */
        readonly particles: number;
        readonly particleRadius?: undefined;
      }
    | {
        /**
         * Radius of every particle in metres. Particles are spaced
         * `2 × particleRadius` apart, so the count follows from the size of
         * what's added.
         */
        readonly particleRadius: number;
        readonly particles?: undefined;
      }
  );

export interface FluidOptions {
  /** Fill this box with liquid. Give this or `mesh`. */
  readonly box?: Box3;
  /** Fill this closed mesh with liquid, in its current world placement. The mesh is hidden. */
  readonly mesh?: Mesh;
  /** Default 0.01, about water. */
  readonly viscosity?: number;
  /** Default 0.1. */
  readonly surfaceTension?: number;
  /** Default 0.02. */
  readonly vorticity?: number;
  /** Default 0.1. */
  readonly adhesion?: number;
  /**
   * Extra thickness for honey-like liquids, around 20 for honey. Giving it,
   * even as 0, adds a slower solver that stays stable at these values, and
   * lets you change {@link Fluid.thickness} later.
   */
  readonly thickness?: number;
  /** Tint of the liquid. Shorthand for `appearance.color`. */
  readonly color?: number;
  /** Everything about the look; see {@link FluidAppearance}. */
  readonly appearance?: Partial<FluidAppearance>;
}

export interface SmokeOptions {
  /** Where smoke is released and the air is heated. Default: centered on the container floor. */
  readonly source?: Vector3;
  /** Radius of the source in metres. Default 0.15. */
  readonly radius?: number;
  /** Tracers released per second. Default 4000. */
  readonly rate?: number;
  /**
   * The highest `rate` you'll set later, which sizes the tracer pool.
   * Default: twice `rate`. Higher rates are capped to it.
   */
  readonly maxRate?: number;
  /** Seconds each tracer lives. Default 6. */
  readonly lifetime?: number;
  /** Upward acceleration of heated air in m/s². Default 3. */
  readonly heat?: number;
  /** How fast the air cools, per second. Default 0.6. */
  readonly cooling?: number;
  /** How opaque the smoke looks. Default 0.7. */
  readonly opacity?: number;
  /** Color of lit smoke. Default `0xd8dfe6`. */
  readonly color?: number;
  /** Color of smoke in its own shadow. Default `0x3b4758`. */
  readonly shadowColor?: number;
}

export interface SoftbodyOptions {
  /**
   * A closed mesh in its current world placement. It's hidden and replaced by
   * a deforming copy that keeps its standard material's colors and maps.
   */
  readonly mesh: Mesh;
  /** 0 is firm rubber, 1 is loose jelly. Default 0.3. */
  readonly softness?: number;
  /** Density in kg/m³. Water is 1000, so lower floats and higher sinks. Default 500. */
  readonly density?: number;
}

export interface ClothOptions {
  /** Size in metres. */
  readonly width: number;
  readonly height: number;
  /** Center of the cloth. Default `(0, 1, 0)`. */
  readonly position?: Vector3;
  /**
   * Orientation. The cloth starts upright in the XY plane, facing +z.
   * `new Euler(-Math.PI / 2, 0, 0)` lays it flat.
   */
  readonly rotation?: Euler;
  /** What holds it up: its top edge, its two top corners, all four corners, or nothing. Default `'top'`. */
  readonly pin?: 'top' | 'top-corners' | 'corners' | 'none';
  /** 0 is stiff canvas, 1 drapes like silk. Default 0.75. */
  readonly softness?: number;
  /** Mass per area in kg/m². Default 0.1, a light fabric. Heavier cloth holds liquid without leaking. */
  readonly weight?: number;
  /** Wind velocity in m/s. Default none. */
  readonly wind?: Vector3;
  /** Fabric color. Default `0x870b21`. */
  readonly color?: number;
  /** Replaces the default sheen material. Its position and normal nodes are overwritten. */
  readonly material?: MeshPhysicalNodeMaterial;
}

export interface ColliderOptions {
  /** Friction, 0 for slick to about 1 for sticky. Default 0.5. */
  readonly friction?: number;
}

export interface MovingColliderOptions extends ColliderOptions {
  /**
   * Follow this object as it moves. The obstacle's center becomes the
   * object's position; boxes also take its rotation. The object's scale is
   * ignored. Moving obstacles drag liquid and cloth along by friction.
   */
  readonly follow?: Object3D;
}

type ColliderDef =
  | { kind: 'floor'; height: number; friction: number }
  | { kind: 'sphere'; center: Vector3; radius: number; follow?: Object3D; friction: number }
  | {
      kind: 'box';
      center: Vector3;
      half: Vector3;
      rotation: Quaternion;
      follow?: Object3D;
      friction: number;
    }
  | {
      kind: 'capsule';
      a: Vector3;
      b: Vector3;
      radius: number;
      follow?: Object3D;
      friction: number;
    }
  | { kind: 'mesh'; mesh: Mesh; resolution: number; friction: number };

interface FluidEntry {
  readonly handle: Fluid;
  readonly options: FluidOptions;
  geometry?: ReturnType<typeof worldGeometry>;
}
interface SmokeEntry {
  readonly handle: Smoke;
  readonly options: SmokeOptions;
}
interface SoftbodyEntry {
  readonly handle: Softbody;
  readonly geometry: ReturnType<typeof worldGeometry>;
}
interface ClothEntry {
  readonly handle: Cloth;
  readonly options: ClothOptions;
}

/** Spacing of cloth particles in particle radii. Above 2, so neighbors never touch and cloth can fold onto itself. */
const CLOTH_SPACING = 2.2;
/** Below this many particles a soft body can't hold its shape's detail. */
const MIN_SOFTBODY_PARTICLES = 100;
/** Below this many particles along a side, cloth folds look faceted. */
const MIN_CLOTH_SIDE = 10;

/** Particle positions and the per-material pieces laid out at one radius. */
interface Layout {
  readonly radius: number;
  readonly init: ParticleInit[];
  readonly bodies: { readonly shape: ReturnType<typeof voxelize>; readonly start: number }[];
  readonly cloths: {
    readonly graph: ReturnType<typeof createClothGraph>;
    readonly offset: number;
    readonly columns: number;
    readonly rows: number;
  }[];
  /** Soft body and cloth particles come first, in `[0, solidEnd)`. */
  readonly solidEnd: number;
  readonly fluidRanges: ParticleRange[];
  readonly airRange: ParticleRange | undefined;
}

/**
 * The easy way in: say what you want, where, and the simulation takes care
 * of particles, sizing, coupling, and drawing.
 *
 * ```ts
 * const renderer = await createParticleRenderer();
 * const sim = new Simulation({
 *   renderer,
 *   scene,
 *   camera,
 *   container: new Box3(min, max),
 *   particles: 5000,
 * });
 * sim.addFluid({ box: new Box3(new Vector3(-0.5, 0, -0.3), new Vector3(0, 0.5, 0.3)) });
 *
 * async function frame() {
 *   await sim.step();
 *   renderer.render(scene, camera);
 *   requestAnimationFrame(frame);
 * }
 * ```
 *
 * Add everything before the first {@link step}; that's when the particles are
 * made. Every material's settings can still change afterwards. For full
 * control, build the same scene from {@link ParticleSystem}, the material
 * classes, and {@link SimLoop} directly.
 */
export class Simulation {
  private readonly options: SimulationOptions;
  private readonly fluids: FluidEntry[] = [];
  private readonly smokes: SmokeEntry[] = [];
  private readonly softbodies: SoftbodyEntry[] = [];
  private readonly cloths: ClothEntry[] = [];
  private readonly colliderDefs: ColliderDef[] = [];
  private readonly objects: Object3D[] = [];
  private readonly debug: Object3D[] = [];
  private readonly shown: Object3D[] = [];
  private readonly disposers: (() => void)[] = [];
  private readonly followers: { collider: SDFCollider; mesh: Mesh }[] = [];
  private readonly stepper = new FrameStepper({ fixedDt: 1 / 60 });
  private readonly gravityValue: Vector3;
  private particleView = false;
  private started: Promise<void> | undefined;
  private stepping: Promise<void> | undefined;
  private lastStepMs = -Infinity;
  private system: ParticleSystem | undefined;
  private simLoop: SimLoop | undefined;
  private radius: number;

  constructor(options: SimulationOptions) {
    const { particles, particleRadius } = options;
    if ((particles === undefined) === (particleRadius === undefined)) {
      throw new Error('Simulation: give exactly one of `particles` or `particleRadius`');
    }
    if (particles !== undefined && !(Number.isInteger(particles) && particles > 0)) {
      throw new Error(`Simulation: particles must be a positive integer, got ${particles}`);
    }
    if (particleRadius !== undefined && !(particleRadius > 0 && Number.isFinite(particleRadius))) {
      throw new Error(
        `Simulation: particleRadius must be a positive number of metres, got ${particleRadius}`,
      );
    }
    this.options = options;
    this.radius = particleRadius ?? 0;
    this.gravityValue = options.gravity?.clone() ?? new Vector3(0, -9.81, 0);
  }

  /** Fill a box or a mesh with liquid. */
  addFluid(options: FluidOptions): Fluid {
    this.assertNotStarted('addFluid');
    this.assertNoSmoke('addFluid');
    if (!options.box === !options.mesh) throw new Error('addFluid: give either `box` or `mesh`');
    const handle = new Fluid(
      {
        viscosity: options.viscosity ?? 0.01,
        surfaceTension: options.surfaceTension ?? 0.1,
        vorticity: options.vorticity ?? 0.02,
        adhesion: options.adhesion ?? 0.1,
        thickness: options.thickness ?? 0,
      },
      {
        color: 0x3a9fcf,
        attenuationDistance: 0.6,
        ...options.appearance,
        ...(options.color === undefined ? {} : { color: options.color }),
      },
      options.thickness !== undefined,
    );
    this.fluids.push({
      handle,
      options,
      ...(options.mesh ? { geometry: worldGeometry(options.mesh) } : {}),
    });
    return handle;
  }

  /** Release smoke from a heated source. The air fills the container, so a container is required. */
  addSmoke(options: SmokeOptions = {}): Smoke {
    this.assertNotStarted('addSmoke');
    const container = this.options.container;
    if (!container) throw new Error('addSmoke: smoke needs a `container` for the air to fill');
    if (this.smokes.length) throw new Error('addSmoke: a simulation can have one smoke source');
    if (this.fluids.length || this.softbodies.length || this.cloths.length) {
      throw new Error(
        'addSmoke: gas and liquid can’t be simulated together, so smoke can’t share a simulation with liquids, soft bodies, or cloth. Use a separate Simulation for the smoke',
      );
    }
    const source =
      options.source?.clone() ??
      new Vector3(
        container.getCenter(new Vector3()).x,
        container.min.y,
        container.getCenter(new Vector3()).z,
      );
    const rate = options.rate ?? 4000;
    const handle = new Smoke({
      rate,
      maxRate: Math.max(rate, options.maxRate ?? rate * 2),
      heat: options.heat ?? 3,
      cooling: options.cooling ?? 0.6,
      opacity: options.opacity ?? 0.7,
      source,
      sourceRadius: options.radius ?? 0.15,
    });
    this.smokes.push({ handle, options });
    if (!this.options.gravity) this.gravityValue.set(0, -1, 0);
    return handle;
  }

  /** Turn a closed mesh into a soft body. */
  addSoftbody(options: SoftbodyOptions): Softbody {
    this.assertNotStarted('addSoftbody');
    this.assertNoSmoke('addSoftbody');
    const handle = new Softbody(options.mesh, {
      softness: options.softness ?? 0.3,
      density: options.density ?? 500,
    });
    this.softbodies.push({ handle, geometry: worldGeometry(options.mesh) });
    return handle;
  }

  /** Add a rectangle of cloth. */
  addCloth(options: ClothOptions): Cloth {
    this.assertNotStarted('addCloth');
    this.assertNoSmoke('addCloth');
    const handle = new Cloth({
      softness: options.softness ?? 0.75,
      weight: options.weight ?? 0.1,
      wind: options.wind?.clone() ?? new Vector3(),
    });
    this.cloths.push({ handle, options });
    return handle;
  }

  /** A floor at `height` that nothing falls through. */
  addFloor(options: ColliderOptions & { readonly height?: number } = {}): void {
    this.addCollider({ kind: 'floor', height: options.height ?? 0, friction: friction(options) });
  }

  /** A solid sphere, fixed at `center` or following an object. */
  addSphere(
    options: MovingColliderOptions & { readonly radius: number; readonly center?: Vector3 },
  ): void {
    this.addCollider({
      kind: 'sphere',
      center: options.center?.clone() ?? new Vector3(),
      radius: options.radius,
      ...(options.follow ? { follow: options.follow } : {}),
      friction: friction(options),
    });
  }

  /** A solid box of the given full `size`, fixed at `center` or following an object. */
  addBox(
    options: MovingColliderOptions & {
      readonly size: Vector3;
      readonly center?: Vector3;
      readonly rotation?: Euler;
    },
  ): void {
    this.addCollider({
      kind: 'box',
      center: options.center?.clone() ?? new Vector3(),
      half: options.size.clone().multiplyScalar(0.5),
      rotation: new Quaternion().setFromEuler(options.rotation ?? new Euler()),
      ...(options.follow ? { follow: options.follow } : {}),
      friction: friction(options),
    });
  }

  /**
   * A solid rod with rounded ends from `start` to `end`. With `follow`, its
   * midpoint moves with the object and it turns as the object turns.
   */
  addCapsule(
    options: MovingColliderOptions & {
      readonly start: Vector3;
      readonly end: Vector3;
      readonly radius: number;
    },
  ): void {
    this.addCollider({
      kind: 'capsule',
      a: options.start.clone(),
      b: options.end.clone(),
      radius: options.radius,
      ...(options.follow ? { follow: options.follow } : {}),
      friction: friction(options),
    });
  }

  /**
   * Make any closed mesh solid. It follows the mesh as it moves. The shape is
   * baked on the CPU when the simulation starts, which takes longer for
   * detailed meshes; a few thousand triangles is plenty.
   */
  addMesh(options: ColliderOptions & { readonly mesh: Mesh; readonly resolution?: number }): void {
    this.addCollider({
      kind: 'mesh',
      mesh: options.mesh,
      resolution: options.resolution ?? 64,
      friction: friction(options),
    });
  }

  /** Gravity in m/s². Mutate it to change gravity at any time. */
  get gravity(): Vector3 {
    return this.gravityValue;
  }

  /**
   * Radius of every particle, in metres. With the `particles` option, it's
   * chosen when the simulation starts and is 0 until then.
   */
  get particleRadius(): number {
    return this.radius;
  }
  /** Number of particles in use. Known once the simulation starts. */
  get particleCount(): number {
    return this.system?.capacity ?? 0;
  }

  /** Show the raw particles instead of the rendered surfaces, to see what's being simulated. */
  get showParticles(): boolean {
    return this.particleView;
  }
  set showParticles(enabled: boolean) {
    this.particleView = enabled;
    for (const object of this.debug) object.visible = enabled;
    for (const object of this.shown) object.visible = !enabled;
  }

  /** The {@link ParticleSystem} underneath, for advanced use. Available once the simulation starts. */
  get particleSystem(): ParticleSystem {
    if (!this.system) throw new Error('Simulation.particleSystem is created on the first step');
    return this.system;
  }
  /** The {@link SimLoop} underneath, for advanced use. Available once the simulation starts. */
  get loop(): SimLoop {
    if (!this.simLoop) throw new Error('Simulation.loop is created on the first step');
    return this.simLoop;
  }

  /**
   * Advance the simulation and update what it draws. Call it once per frame,
   * before rendering. Without `dt`, it keeps pace with the clock at 60 steps
   * per second. The first call builds everything that was added.
   */
  step(dt?: number): Promise<void> {
    // A step already running (from an animation loop that doesn't wait) covers this frame.
    this.stepping ??= this.runStep(dt).finally(() => (this.stepping = undefined));
    return this.stepping;
  }

  /** Build the particles, physics, and renderers now instead of on the first step. */
  start(): Promise<void> {
    this.started ??= Promise.resolve()
      .then(() => this.build())
      .catch((error: unknown) => {
        // Nothing reaches the GPU before the checks that can fail, so the
        // scene can be fixed and started again.
        this.started = undefined;
        this.restoreSources();
        throw error;
      });
    return this.started;
  }

  /** Remove everything from the scene and free its GPU resources. */
  dispose(): void {
    for (const object of [...this.objects, ...this.debug]) object.removeFromParent();
    for (const dispose of this.disposers) dispose();
    this.disposers.length = 0;
    this.simLoop?.dispose();
    this.system?.dispose();
    this.restoreSources();
  }

  private async runStep(dt: number | undefined): Promise<void> {
    await this.start();
    if (dt === undefined) {
      // After a pause, start the clock again instead of catching up.
      const now = performance.now();
      if (now - this.lastStepMs > 250) this.stepper.reset();
      this.lastStepMs = now;
      await this.stepper.pump(now, (fixed) => this.advance(fixed));
    } else {
      await this.advance(dt);
    }
    await this.render();
  }

  /** Show the meshes that soft bodies and liquids were made from again. */
  private restoreSources(): void {
    for (const { handle } of this.softbodies) handle.source.visible = true;
    for (const { options } of this.fluids) if (options.mesh) options.mesh.visible = true;
  }

  private addCollider(def: ColliderDef): void {
    this.assertNotStarted('adding obstacles');
    this.colliderDefs.push(def);
  }

  private assertNotStarted(what: string): void {
    if (this.started) {
      throw new Error(`Simulation: ${what} must happen before the first step() or start()`);
    }
  }

  private assertNoSmoke(what: string): void {
    if (this.smokes.length) {
      throw new Error(
        `${what}: gas and liquid can’t be simulated together, so this can’t share a simulation with smoke. Use a separate Simulation for the smoke`,
      );
    }
  }

  private async advance(dt: number): Promise<void> {
    for (const { collider, mesh } of this.followers) {
      mesh.updateWorldMatrix(true, false);
      collider.setTransform(mesh.matrixWorld);
    }
    this.loop.gravity.copy(this.gravityValue);
    for (const { handle } of this.smokes) {
      handle.carry += Math.min(handle.settings.rate, handle.settings.maxRate) * dt;
      const count = Math.floor(handle.carry);
      handle.carry -= count;
      const { source, sourceRadius } = handle.settings;
      for (let i = 0; i < count; i++) {
        const angle = Math.random() * Math.PI * 2;
        const r = sourceRadius * 0.9 * Math.sqrt(Math.random());
        handle.system!.emit([
          source.x + Math.cos(angle) * r,
          source.y + this.radius + Math.random() * sourceRadius * 0.3,
          source.z + Math.sin(angle) * r,
        ]);
      }
    }
    for (const { handle } of this.cloths) handle.system!.wind.copy(handle.settings.wind);
    await this.loop.step(dt);
  }

  private async render(): Promise<void> {
    if (!this.particleView) {
      for (const { handle } of this.fluids) await handle.renderer!.update();
    }
    for (const { handle } of this.smokes) await handle.renderer!.update();
  }

  private build(): void {
    const { renderer, scene, camera, container } = this.options;
    const smoke = this.smokes.length > 0;
    if (!this.fluids.length && !smoke && !this.softbodies.length && !this.cloths.length) {
      throw new Error('Simulation: add a fluid, smoke, soft body, or cloth before stepping');
    }
    if (this.fluids.length && !container) {
      console.warn(
        'Simulation: without a `container`, liquid is only drawn near where it starts. Give a container to draw it wherever it flows.',
      );
    }

    const sdfs = new Map<Mesh, SDFData>();
    let layout: Layout;
    if (this.options.particleRadius !== undefined) {
      const r = this.options.particleRadius;
      layout = this.layout(r, this.obstacles(sdfs, 4 * r));
    } else {
      layout = this.fitBudget(this.options.particles, sdfs);
    }
    const r = (this.radius = layout.radius);
    const { init, solidEnd, fluidRanges, airRange } = layout;

    for (const range of fluidRanges)
      if (range.count === 0)
        throw new Error('addFluid: the fluid has no room; check its box and the container');
    const bodyDefs: SoftbodyDef[] = this.softbodies.map(({ handle }, i) => {
      const { shape, start } = layout.bodies[i]!;
      if (shape.count === 0)
        throw new Error('addSoftbody: the mesh is too small for the particle size');
      handle.index = i;
      handle.count = shape.count;
      return {
        range: { start, count: shape.count },
        surfaceCount: shape.surfaceCount,
        compliance: softbodyCompliance(handle.settings.softness, shape.count),
        edges: shape.edges,
      };
    });
    const clothGraphs = layout.cloths;
    this.cloths.forEach(({ handle }, i) => {
      const { graph, columns } = clothGraphs[i]!;
      handle.segments = columns;
      handle.count = graph.positions.length;
    });
    this.fluids.forEach(({ handle, options }, i) => {
      handle.count = fluidRanges[i]!.count;
      if (options.mesh) options.mesh.visible = false;
    });
    if (airRange) this.smokes[0]!.handle.count = airRange.count;
    this.warnIfCoarse(layout);

    const particles = (this.system = new ParticleSystem(renderer, init.length, r));
    particles.uploadParticles(init);

    // Physics.
    const materials: Material[] = [];
    const softbodySystem = bodyDefs.length
      ? new SoftbodySystem(particles, { bodies: bodyDefs, shapeMatching: 'local' })
      : undefined;
    const clothSystems = this.cloths.map(({ handle }, i) => {
      const { graph, offset, columns } = clothGraphs[i]!;
      const cloth = new ClothSystem(particles, {
        graph,
        offset,
        bendCompliance: clothBendCompliance(
          handle.settings.softness,
          columns,
          handle.settings.weight,
        ),
        stretchTolerance: 0.06,
        wind: handle.settings.wind.clone(),
        drag: 0.18,
        lift: 0.02,
        damping: 0.1,
      });
      handle.system = cloth;
      return cloth;
    });
    const solidRanges: ParticleRange[] = [
      ...(softbodySystem
        ? softbodySystem.bodies.map((_, i) => softbodySystem.surfaceRange(i))
        : []),
      ...clothSystems.map((cloth) => cloth.range),
    ];
    this.fluids.forEach(({ handle }, i) => {
      const { viscosity, surfaceTension, vorticity, adhesion, thickness } = handle.settings;
      const fluid = new FluidSystem(particles, {
        range: fluidRanges[i]!,
        viscosity,
        surfaceTension,
        vorticity,
        adhesion,
        sortByCell: true,
      });
      for (const range of solidRanges) fluid.addBoundary(range);
      handle.system = fluid;
      materials.push(fluid);
      if (handle.thickEnabled) {
        handle.thick = new ViscositySolver(fluid, {
          viscosity: thickness,
          iterations: Math.round(16 * Math.cbrt(fluidRanges[i]!.count / 10000)),
        });
        materials.push(handle.thick);
      }
    });
    if (softbodySystem) {
      materials.push(softbodySystem);
      this.softbodies.forEach(({ handle }) => (handle.system = softbodySystem));
    }
    materials.push(...clothSystems);
    if (smoke) {
      const { handle, options } = this.smokes[0]!;
      const air = new FluidSystem(particles, {
        range: airRange!,
        viscosity: 0.02,
        vorticity: 0.06,
        sortByCell: true,
      });
      const height = container!.max.y - container!.min.y;
      const gas = new GasSystem(air, {
        capacity: Math.ceil(handle.settings.maxRate * (options.lifetime ?? 6) * 1.1) + 1000,
        lifetime: options.lifetime ?? 6,
        bounds: new Box3(
          container!.min.clone().subScalar(1),
          container!.max.clone().setY(container!.max.y - height * 0.1),
        ),
        heatSources: [{ position: handle.settings.source, radius: handle.settings.sourceRadius }],
        buoyancy: handle.settings.heat,
        cooling: handle.settings.cooling,
      });
      handle.system = gas;
      this.disposers.push(() => gas.dispose());
      materials.push(gas, air);
    }

    const colliders: Collider[] = [];
    const walls = new PrimitiveSet(particles);
    if (container) {
      const { min, max } = container;
      walls.addPlane(new Vector3(0, 1, 0), min);
      walls.addPlane(new Vector3(1, 0, 0), min);
      walls.addPlane(new Vector3(-1, 0, 0), max);
      walls.addPlane(new Vector3(0, 0, 1), min);
      walls.addPlane(new Vector3(0, 0, -1), max);
      if (this.options.closed || smoke) walls.addPlane(new Vector3(0, -1, 0), max);
    }
    const clothThickness = this.cloths.length ? 1.3 * CLOTH_SPACING * r - r : 0;
    for (const def of this.colliderDefs) {
      const muS = def.friction,
        muK = def.friction * 0.8;
      if (def.kind === 'floor')
        walls.addPlane(new Vector3(0, 1, 0), new Vector3(0, def.height, 0), { muS, muK });
      if (def.kind === 'sphere') {
        const slot = walls.addSphere(def.center, def.radius, { muS, muK });
        if (def.follow) walls.attach(slot, def.follow);
      }
      if (def.kind === 'box') {
        const slot = walls.addBox(def.center, def.half, { rotation: def.rotation, muS, muK });
        if (def.follow) walls.attach(slot, def.follow);
      }
      if (def.kind === 'capsule') {
        const slot = walls.addCapsule(def.a, def.b, def.radius, { muS, muK });
        if (def.follow) walls.attach(slot, def.follow);
      }
      if (def.kind === 'mesh') {
        const collider = new SDFCollider(particles, sdfs.get(def.mesh)!, {
          muS,
          muK,
          thickness: clothThickness,
        });
        collider.setTransform(def.mesh.matrixWorld);
        this.followers.push({ collider, mesh: def.mesh });
        colliders.push(collider);
        this.disposers.push(() => collider.dispose());
      }
    }
    if (walls.count > 0) colliders.unshift(walls);
    this.disposers.push(() => walls.dispose());

    const needsContact = bodyDefs.length > 0 || this.cloths.length > 0;
    const base = Math.max(
      this.fluids.length ? 4 : 0,
      smoke ? 2 : 0,
      bodyDefs.length ? 6 : 0,
      this.cloths.length ? 8 : 0,
    );
    const substeps =
      this.options.substeps ?? Math.min(24, Math.ceil(base * Math.max(1, 0.018 / r)));
    this.simLoop = new SimLoop(particles, {
      substeps,
      iterations: 2,
      gravity: this.gravityValue,
      materials,
      colliders,
      ...(needsContact ? { contact: { muS: 0.3, muK: 0.2 } } : {}),
    });

    // Drawing.
    const bounds = container
      ? container.clone().expandByVector(new Vector3(0.03, 0.02, 0.03))
      : contentBounds(init, r);
    const surfaceColliders = colliders.filter(
      (c): c is PrimitiveSet | SDFCollider => c instanceof PrimitiveSet || c instanceof SDFCollider,
    );
    const solids = solidEnd > 0 ? { start: 0, count: solidEnd } : undefined;
    this.fluids.forEach(({ handle }, i) => {
      const surface = new FluidSurfaceRenderer(handle.system!, {
        renderer,
        scene,
        camera,
        bounds,
        colliders: surfaceColliders,
        ...(solids ? { solids } : {}),
        motionStretch: 0.02,
        appearance: handle.look,
      });
      handle.renderer = surface;
      this.show(surface.mesh);
      this.disposers.push(() => surface.dispose());
      this.debugView(
        createParticleMesh(particles, {
          range: fluidRanges[i]!,
          color: handle.look.color ?? 0x5fb9ff,
        }),
      );
    });
    this.softbodies.forEach(({ handle, geometry }) => {
      // Standard and node standard materials share the fields the skin copies.
      const source = handle.source.material;
      const material =
        !Array.isArray(source) && 'roughness' in source && 'color' in source
          ? (source as MeshStandardMaterial)
          : undefined;
      const mesh = new SoftbodyMesh(softbodySystem!, handle.index, geometry, material);
      this.disposers.push(() => {
        mesh.geometry.dispose();
        mesh.material.dispose();
      });
      mesh.castShadow = mesh.receiveShadow = true;
      handle.skinned = mesh;
      handle.source.visible = false;
      this.show(mesh);
      this.debugView(
        createParticleMesh(particles, {
          range: softbodySystem!.particleRange(handle.index),
          color: 0xd8bdd2,
        }),
      );
    });
    this.cloths.forEach(({ handle, options }, i) => {
      const { columns, rows } = clothGraphs[i]!;
      const material =
        options.material ??
        new MeshPhysicalNodeMaterial({
          color: new Color(options.color ?? 0x870b21),
          side: DoubleSide,
          roughness: 0.9,
          sheen: 1,
          sheenColor: new Color(options.color ?? 0x870b21).lerp(new Color(0xffffff), 0.35),
          sheenRoughness: 0.7,
        });
      const surface = createClothSurface(clothSystems[i]!, {
        columns: columns + 1,
        rows: rows + 1,
        material,
      });
      surface.castShadow = surface.receiveShadow = true;
      handle.surfaceMesh = surface;
      this.disposers.push(() => {
        surface.geometry.dispose();
        if (!options.material) material.dispose();
      });
      this.show(surface);
      this.debugView(
        createParticleMesh(particles, {
          range: clothSystems[i]!.range,
          radius: r,
          color: options.color ?? 0xd54b5c,
        }),
      );
    });
    if (smoke) {
      const { handle, options } = this.smokes[0]!;
      const size = container!.getSize(new Vector3());
      const top = container!.max.y - size.y * 0.1;
      const scale = 128 / Math.max(size.x, top - container!.min.y, size.z);
      const resolution = [size.x, top - container!.min.y, size.z].map((s) =>
        Math.max(8, Math.min(128, Math.round(s * scale))),
      ) as [number, number, number];
      const volume = new GasVolumeRenderer(handle.system!, {
        renderer,
        min: container!.min.clone(),
        max: container!.max.clone().setY(top),
        resolution,
        steps: 80,
        density: handle.settings.opacity,
        ...(options.color === undefined ? {} : { color: options.color }),
        ...(options.shadowColor === undefined ? {} : { shadowColor: options.shadowColor }),
      });
      handle.renderer = volume;
      this.show(volume.object);
      this.disposers.push(() => volume.dispose());
      this.debugView(
        createParticleMesh(particles, { range: airRange!, radius: r * 0.25, color: 0x6d8a9c }),
      );
    }
    this.showParticles = this.particleView;
  }

  /**
   * The layout with the smallest particle radius that fits within `budget`
   * particles. The first guess comes from the content's rough volume and
   * area; each pass then rescales the radius by the ratio of the count it got
   * to the budget, keeping the best layout that fits.
   */
  private fitBudget(budget: number, sdfs: Map<Mesh, SDFData>): Layout {
    const { container } = this.options;
    const boundsVolume = (box: Box3) => {
      if (box.isEmpty()) return 0;
      const size = box.getSize(new Vector3());
      return size.x * size.y * size.z;
    };
    const meshBounds = (geometry: BufferGeometry) => {
      geometry.computeBoundingBox();
      return boundsVolume(geometry.boundingBox!);
    };
    let volume = 0;
    for (const { options, geometry } of this.fluids) {
      volume += options.box
        ? boundsVolume(container ? options.box.clone().intersect(container) : options.box)
        : meshBounds(geometry!);
    }
    for (const { geometry } of this.softbodies) volume += meshBounds(geometry);
    if (this.smokes.length) volume += boundsVolume(container!);
    const area = this.cloths.reduce((sum, { options }) => sum + options.width * options.height, 0);
    const guess = radiusForBudget(volume, area, CLOTH_SPACING, budget);

    // Mesh obstacles are baked once, padded for radii well above the guess.
    const obstacles = this.obstacles(sdfs, 8 * guess);
    let r = guess;
    let best: Layout | undefined;
    for (let pass = 0; pass < 12; pass++) {
      const layout = this.layout(r, obstacles);
      const count = layout.init.length;
      if (count <= budget && (!best || count > best.init.length)) best = layout;
      if (count <= budget && count >= 0.97 * budget) break;
      r *= Math.cbrt(Math.max(count, 1) / (0.985 * budget));
    }
    // Every pass overshot: grow the particles until the scene fits.
    for (let pass = 0; !best && pass < 100; pass++) {
      r *= 1.05;
      const layout = this.layout(r, obstacles);
      if (layout.init.length <= budget) best = layout;
    }
    if (!best) throw new Error(`Simulation: the scene can't fit in ${budget} particles`);
    if (4 * best.radius > 8 * guess) {
      sdfs.clear();
      this.obstacles(sdfs, 4 * best.radius);
    }
    return best;
  }

  /**
   * Obstacles that particles mustn't start inside. Bakes each `addMesh`
   * shape into `sdfs`, padded by `padding` metres.
   */
  private obstacles(sdfs: Map<Mesh, SDFData>, padding: number): Obstacle[] {
    const obstacles: Obstacle[] = [];
    for (const def of this.colliderDefs) {
      if (def.kind === 'sphere' && !def.follow)
        obstacles.push(sphereObstacle(def.center, def.radius));
      if (def.kind === 'sphere' && def.follow)
        obstacles.push(sphereObstacle(def.follow.getWorldPosition(new Vector3()), def.radius));
      if (def.kind === 'box') {
        const center = def.follow ? def.follow.getWorldPosition(new Vector3()) : def.center;
        const rotation = def.follow
          ? def.follow.getWorldQuaternion(new Quaternion())
          : def.rotation;
        obstacles.push(boxObstacle(center, def.half, rotation));
      }
      if (def.kind === 'capsule') {
        const shift = def.follow
          ? def.follow
              .getWorldPosition(new Vector3())
              .sub(def.a.clone().add(def.b).multiplyScalar(0.5))
          : new Vector3();
        obstacles.push(
          capsuleObstacle(def.a.clone().add(shift), def.b.clone().add(shift), def.radius),
        );
      }
      if (def.kind === 'floor') obstacles.push((p) => p.y - def.height);
      if (def.kind === 'mesh') {
        const sdf = bakeMeshToSdf(def.mesh.geometry, { resolution: def.resolution, padding });
        sdfs.set(def.mesh, sdf);
        def.mesh.updateWorldMatrix(true, false);
        obstacles.push(sdfObstacle(sdf, def.mesh.matrixWorld));
      }
    }
    return obstacles;
  }

  /** Lay out every particle at radius `r`: soft bodies, then cloth, then liquid, then air. */
  private layout(r: number, obstacles: readonly Obstacle[]): Layout {
    const { container } = this.options;
    const spacing = 2 * r;
    const init: ParticleInit[] = [];
    const solidPositions: Float32Array[] = [];
    const bodies = this.softbodies.map(({ handle, geometry }) => {
      const shape = voxelize(geometry, { particleRadius: r, largestPiece: true });
      const start = init.length;
      const invMass = 1 / (handle.settings.density * spacing ** 3);
      const p = shape.positions;
      for (let i = 0; i < shape.count; i++) {
        init.push({ position: [p[i * 3]!, p[i * 3 + 1]!, p[i * 3 + 2]!], invMass });
      }
      solidPositions.push(p);
      return { shape, start };
    });
    const cloths = this.cloths.map(({ handle, options }) => {
      const columns = Math.max(2, Math.round(options.width / (CLOTH_SPACING * r)));
      const rows = Math.max(2, Math.round(options.height / (CLOTH_SPACING * r)));
      const geometry = new PlaneGeometry(options.width, options.height, columns, rows);
      geometry.applyQuaternion(new Quaternion().setFromEuler(options.rotation ?? new Euler()));
      const at = options.position ?? new Vector3(0, 1, 0);
      geometry.translate(at.x, at.y, at.z);
      const graph = createClothGraph(geometry, {
        surfaceDensity: handle.settings.weight,
        pinnedIndices: pinned(options.pin ?? 'top', columns, rows),
      });
      geometry.dispose();
      const offset = init.length;
      for (const position of graph.positions) init.push({ position });
      solidPositions.push(new Float32Array(graph.positions.flat()));
      return { graph, offset, columns, rows };
    });
    const solidEnd = init.length;
    const shrink = (box: Box3) => box.clone().expandByScalar(-r);
    const fluidRanges = this.fluids.map(({ options, geometry }) => {
      let points: number[];
      if (options.box) {
        points = fillBox(
          container ? options.box.clone().intersect(shrink(container)) : options.box,
          spacing,
        );
      } else {
        points = Array.from(voxelize(geometry!, { particleRadius: r }).positions);
      }
      points = clearOf(points, spacing, obstacles, solidPositions);
      const start = init.length;
      for (let i = 0; i < points.length; i += 3)
        init.push({ position: [points[i]!, points[i + 1]!, points[i + 2]!] });
      return { start, count: init.length - start };
    });
    let airRange: ParticleRange | undefined;
    if (this.smokes.length) {
      const start = init.length;
      const points = fillBox(shrink(container!), spacing);
      for (let i = 0; i < points.length; i += 3)
        init.push({ position: [points[i]!, points[i + 1]!, points[i + 2]!] });
      airRange = { start, count: init.length - start };
    }
    return { radius: r, init, bodies, cloths, solidEnd, fluidRanges, airRange };
  }

  /** Warn about soft bodies and cloth too coarse to keep their shape. */
  private warnIfCoarse(layout: Layout): void {
    const fix = this.options.particles === undefined ? 'lower particleRadius' : 'raise particles';
    const radius = `particleRadius ${layout.radius.toPrecision(3)}`;
    this.softbodies.forEach(({ handle }, i) => {
      const count = layout.bodies[i]!.shape.count;
      if (count > 0 && count < MIN_SOFTBODY_PARTICLES) {
        const name = handle.source.name ? `"${handle.source.name}"` : `#${i}`;
        console.warn(
          `Simulation: soft body ${name} has ${count} particles at ${radius}, too few to keep its shape. To give it more, ${fix} or scale the scene up.`,
        );
      }
    });
    this.cloths.forEach((_, i) => {
      const { columns, rows } = layout.cloths[i]!;
      if (Math.min(columns, rows) + 1 < MIN_CLOTH_SIDE) {
        console.warn(
          `Simulation: cloth #${i} is ${columns + 1} × ${rows + 1} particles at ${radius}, too few to fold smoothly. To give it more, ${fix}.`,
        );
      }
    });
  }

  private show(object: Object3D): void {
    this.options.scene.add(object);
    this.objects.push(object);
    this.shown.push(object);
  }

  private debugView(mesh: InstancedMesh): void {
    mesh.visible = false;
    this.options.scene.add(mesh);
    this.debug.push(mesh);
    this.disposers.push(() => {
      mesh.geometry.dispose();
      (mesh.material as { dispose(): void }).dispose();
    });
  }
}

function friction(options: ColliderOptions): number {
  return options.friction ?? 0.5;
}

function pinned(pin: NonNullable<ClothOptions['pin']>, columns: number, rows: number): number[] {
  const width = columns + 1;
  const corner = (column: number, row: number) => row * width + column;
  if (pin === 'top') return Array.from({ length: width }, (_, i) => i);
  if (pin === 'top-corners') return [corner(0, 0), corner(columns, 0)];
  if (pin === 'corners')
    return [corner(0, 0), corner(columns, 0), corner(0, rows), corner(columns, rows)];
  return [];
}

/** A drawing box around every particle, with room for liquid to spread. */
function contentBounds(init: readonly ParticleInit[], radius: number): Box3 {
  const box = new Box3();
  const point = new Vector3();
  for (const { position } of init) box.expandByPoint(point.fromArray(position));
  const size = box.getSize(new Vector3());
  return box.expandByVector(size.multiplyScalar(0.5).addScalar(4 * radius));
}
