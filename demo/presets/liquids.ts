import { Box3, Group, Object3D, Vector3, type InstancedMesh } from 'three';
import { Fn, If, float, instanceIndex, uniform, vec4 } from 'three/tsl';
import {
  FluidSurfaceRenderer,
  FluidSystem,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  type FluidSurfaceRendererOptions,
  type ParticleInit,
} from '../../src/index.js';
import { basin, block, glassTank, pedestal, pivotStand } from '../runtime/stage.js';
import type { BuildContext, Experiment, Values } from '../types.js';
import { fitRadius, lattice, particleView, scaledSubsteps, tank } from './shared.js';

// TSL's generated operator chains need the broad node type at graph boundaries.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/** A fluid's rendered surface plus the particle view the demo can switch to. */
export interface LiquidVisual {
  readonly surface: FluidSurfaceRenderer;
  readonly dots: InstancedMesh;
  update(): Promise<void>;
  setParticleView(enabled: boolean): void;
}

export function liquidVisual(
  ctx: BuildContext,
  fluid: FluidSystem,
  options: Omit<FluidSurfaceRendererOptions, 'renderer' | 'scene' | 'camera'> & {
    readonly dotColor?: number;
  },
): LiquidVisual {
  const surface = new FluidSurfaceRenderer(fluid, {
    renderer: ctx.renderer,
    scene: ctx.scene,
    camera: ctx.camera,
    ...options,
  });
  const dots = particleView(fluid.particles, {
    range: fluid.range,
    color: options.dotColor ?? options.appearance?.color ?? 0x5fb9ff,
  });
  let particlesShown = false;
  return {
    surface,
    dots,
    // The surface is skipped while the particles are shown.
    update: () => (particlesShown ? Promise.resolve() : surface.update()),
    setParticleView(enabled) {
      particlesShown = enabled;
      dots.visible = enabled;
      surface.mesh.visible = !enabled;
    },
  };
}

/** Spin the marble drop starts with, in rad/s. */
const MARBLE_SPIN = 0.5;
const WAVE_CEILING = 0.7;
const WAVE_WALL_HALF = 0.0325;
// Tall enough that the tank's corners clear the stand through a full turn.
const WAVE_PIVOT = 1.05;
const WAVE_LIFT = WAVE_PIVOT - WAVE_CEILING / 2;
// Walls alternate between rising from the floor and hanging from the lid,
// each spanning half the height, so water snakes over and under them.
const WAVE_WALLS = [-0.48, -0.16, 0.16, 0.48].map((x, i) => {
  const fromFloor = i % 2 === 0;
  return {
    x,
    bottom: fromFloor ? 0 : WAVE_CEILING / 2,
    top: fromFloor ? WAVE_CEILING / 2 : WAVE_CEILING,
  };
});

/**
 * The sealed wave tank on its pivot. Every wall is a box collider attached
 * to the tipping group, so the particles see the tank itself move.
 */
function waveTank(particles: ParticleSystem, halfX: number, halfZ: number) {
  const pivot = new Group();
  pivot.position.y = WAVE_PIVOT;
  const body = new Group();
  body.position.y = -WAVE_CEILING / 2;
  pivot.add(body);
  body.add(glassTank(halfX * 2, halfZ * 2, WAVE_CEILING, 0.012));
  pivot.updateMatrixWorld(true);
  const colliders = new PrimitiveSet(particles);
  const friction = { muS: 0.08, muK: 0.04 };
  const addWall = (center: Vector3, half: Vector3) => {
    const anchor = new Object3D();
    anchor.position.copy(center);
    body.add(anchor);
    colliders.attach(colliders.addBox(center, half, friction), anchor);
  };
  // Thick slabs just outside the interior, so fast particles can't tunnel out.
  const t = 0.1,
    h = WAVE_CEILING / 2;
  addWall(new Vector3(0, -t, 0), new Vector3(halfX + t, t, halfZ + t));
  addWall(new Vector3(0, WAVE_CEILING + t, 0), new Vector3(halfX + t, t, halfZ + t));
  for (const side of [-1, 1]) {
    addWall(new Vector3(side * (halfX + t), h, 0), new Vector3(t, h + t, halfZ + t));
    addWall(new Vector3(0, h, side * (halfZ + t)), new Vector3(halfX + t, h + t, t));
  }
  for (const wall of WAVE_WALLS) {
    const height = wall.top - wall.bottom;
    const center = new Vector3(wall.x, wall.bottom + height / 2, 0);
    addWall(center, new Vector3(WAVE_WALL_HALF, height / 2, halfZ));
    body.add(block([WAVE_WALL_HALF * 2, height, halfZ * 2], center.toArray(), 0x8da8ac, 0.009));
  }
  return { pivot, stand: pivotStand(halfZ * 2 + 0.1, WAVE_PIVOT), colliders };
}

export async function buildFluid(
  ctx: BuildContext,
  values: Values,
  kind: 'wave' | 'impact' | 'marble',
): Promise<Experiment> {
  const halfX = 0.8,
    halfZ = 0.55;
  const fill = (radius: number): ParticleInit[] => {
    const spacing = radius * 2;
    if (kind === 'wave') {
      // A layer held against the lid, which drops through the walls at the start.
      return lattice(
        [-halfX, WAVE_LIFT + WAVE_CEILING - 0.25, -halfZ],
        [halfX, WAVE_LIFT + WAVE_CEILING - radius, halfZ],
        spacing,
        (x, y) =>
          !WAVE_WALLS.some(
            (wall) =>
              Math.abs(x - wall.x) < WAVE_WALL_HALF + radius &&
              y - WAVE_LIFT > wall.bottom - radius &&
              y - WAVE_LIFT < wall.top + radius,
          ),
      );
    }
    if (kind === 'impact') {
      const center = values['height']!;
      return [
        ...lattice([-0.72, radius, -0.46], [0.72, 0.17, 0.46], spacing),
        ...lattice(
          [-0.24, center - 0.24, -0.24],
          [0.24, center + 0.24, 0.24],
          spacing,
          (x, y, z) => x * x + (y - center) ** 2 + z * z < 0.24 ** 2,
        ),
      ];
    }
    return lattice(
      [-0.31, 0.2, -0.31],
      [0.31, 0.82, 0.31],
      spacing,
      (x, y, z) => x * x + (y - 0.51) ** 2 + z * z < 0.3 ** 2,
    ).map(({ position: [x, y, z] }) => ({
      position: [x, y, z],
      velocity: [-z * MARBLE_SPIN, 0, x * MARBLE_SPIN],
    }));
  };
  // Size particles so the preset fills its volume with the requested count.
  const radius = fitRadius(fill, ctx.particles, 0.018);
  const initial = fill(radius);
  const particles = new ParticleSystem(ctx.renderer, initial.length, radius);
  particles.uploadParticles(initial);
  const fluid = new FluidSystem(particles, {
    viscosity: values['viscosity']!,
    surfaceTension: values['tension']!,
    // The marble's own spin is its only motion; confinement would keep adding to it.
    ...(kind === 'marble' ? {} : { vorticity: 0.025 }),
    sortByCell: true,
  });
  const wave = kind === 'wave' ? waveTank(particles, halfX, halfZ) : undefined;
  const colliders = wave?.colliders ?? tank(particles, halfX, halfZ);
  const substeps = scaledSubsteps(3, ctx.particles),
    iterations = 2;
  const loop = new SimLoop(particles, {
    substeps,
    iterations,
    // The marble has no downward gravity; it is pulled toward its center below.
    gravity: new Vector3(0, kind === 'marble' ? 0 : -values['gravity']!, 0),
    colliders: [colliders],
    materials: [fluid],
  });

  // The wave tank's interior sweeps a disc of radius |(halfX, ceiling / 2)|.
  const sweep = Math.hypot(halfX, WAVE_CEILING / 2) + 0.03;
  const bounds =
    kind === 'wave'
      ? new Box3(
          new Vector3(-sweep, WAVE_PIVOT - sweep, -halfZ - 0.03),
          new Vector3(sweep, WAVE_PIVOT + sweep, halfZ + 0.03),
        )
      : kind === 'marble'
        ? new Box3(new Vector3(-0.66, -0.02, -0.66), new Vector3(0.66, 1.15, 0.66))
        : new Box3(
            new Vector3(-halfX - 0.03, -0.02, -halfZ - 0.03),
            new Vector3(halfX + 0.03, values['height']! + 0.35, halfZ + 0.03),
          );
  const looks = {
    wave: { color: 0x123f8f, attenuationDistance: 0.18, scattering: 0.16, roughness: 0.1 },
    impact: { color: 0x5aa6cf, attenuationDistance: 0.6, scattering: 0.04, roughness: 0.08 },
    marble: {
      color: 0x7fc4d8,
      attenuationDistance: 0.8,
      scattering: 0.03,
      envIntensity: 1.2,
      roughness: 0.07,
    },
  };
  const visual = liquidVisual(ctx, fluid, {
    bounds,
    colliders: [colliders],
    appearance: looks[kind],
    // The wave grid covers the tank's whole swing; give it more voxels to keep detail.
    ...(kind === 'wave'
      ? { voxelBudget: 1.6 * FluidSurfaceRenderer.defaultVoxelBudget(initial.length) }
      : {}),
  });

  // Liquid marble: gravity toward the drop's center instead of downward.
  const attractor = uniform(new Vector3(0, 0.51, 0));
  const pull = uniform(values['gravity']!);
  const stepDt = uniform(1 / 60);
  const attract = Fn(() => {
    const velocity: Any = particles.velocities.element(instanceIndex);
    const offset: Any = attractor.sub(particles.positions.element(instanceIndex).xyz);
    const change: Any = offset.div(offset.length().max(0.25)).mul(pull).mul(stepDt);
    velocity.assign(vec4(velocity.xyz.add(change), velocity.w));
  })().compute(initial.length);

  // Clicks: the marble bursts outward from its center; pools get a tighter splash.
  const hit = uniform(new Vector3());
  const splashRadius = kind === 'marble' ? 0.4 : 0.22;
  const strength = kind === 'marble' ? 3.2 : 3.8;
  const splash = Fn(() => {
    const position: Any = particles.positions.element(instanceIndex).xyz;
    const distance: Any = position.sub(hit).length();
    If(distance.lessThan(splashRadius), () => {
      const velocity: Any = particles.velocities.element(instanceIndex);
      // Smooth, wide falloff pulls out a broad lobe rather than a thin thread.
      const falloff: Any = float(1).sub(distance.div(splashRadius).pow(2)).pow(2);
      const direction: Any =
        kind === 'marble'
          ? position.sub(hit).div(distance.max(1e-4))
          : hit.sub(attractor).normalize();
      velocity.assign(vec4(velocity.xyz.add(direction.mul(falloff.mul(strength))), velocity.w));
    });
  })().compute(initial.length);

  let angle = 0;
  return {
    particles,
    loop,
    objects: wave
      ? [wave.stand, wave.pivot, visual.surface.mesh, visual.dots]
      : [
          kind === 'marble' ? pedestal(0.62) : basin(halfX * 2, halfZ * 2),
          visual.surface.mesh,
          visual.dots,
        ],
    particleCount: initial.length,
    substeps,
    iterations,
    async update(dt) {
      if (wave) {
        // Accumulate the angle so speed changes don't make the tank jump.
        angle = (angle + dt * values['speed']!) % (Math.PI * 2);
        wave.pivot.rotation.z = angle;
      }
      if (kind === 'marble') {
        stepDt.value = dt;
        await ctx.renderer.computeAsync(attract);
      }
    },
    prepareRender: () => visual.update(),
    setReflections: (enabled) => (visual.surface.reflections = enabled),
    setParticleView: (enabled) => visual.setParticleView(enabled),
    setParameter(key, value) {
      values[key] = value;
      if (key === 'gravity') {
        if (kind === 'marble') pull.value = value;
        else loop.gravity.y = -value;
      }
      if (key === 'viscosity') fluid.viscosity = value;
      if (key === 'tension') fluid.surfaceTension = value;
    },
    async interact(uv) {
      const point = await visual.surface.pick(uv);
      if (!point) return false;
      if (kind === 'marble') hit.value.copy(attractor.value);
      else {
        hit.value.copy(point);
        attractor.value.copy(point).add(new Vector3(0, -0.2, 0));
      }
      await ctx.renderer.computeAsync(splash);
      return true;
    },
    dispose() {
      visual.surface.dispose();
      particles.dispose();
      loop.dispose();
    },
  };
}
