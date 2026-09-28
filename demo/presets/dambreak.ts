import { Box3, Quaternion, Vector3 } from 'three';
import { FluidSystem, ParticleSystem, SDFCollider, SimLoop } from '../../src/index.js';
import { basin } from '../runtime/stage.js';
import type { BuildContext, Experiment, Values } from '../types.js';
import { BUNNY_YAW, loadBunny } from './honey.js';
import { liquidVisual } from './liquids.js';
import { fitRadius, lattice, scaledSubsteps, tank } from './shared.js';

const HALF_X = 1,
  HALF_Z = 0.45,
  /** The water starts as a column filling this much of the tank's length. */
  COLUMN = 0.6,
  /** The column starts this far above the floor, so it lands with a splash. */
  LIFT = 0.2,
  BUNNY_SCALE = 1.6;

/** A column of water drops, collapses, and floods into a large Stanford bunny. */
export async function buildDamBreak(ctx: BuildContext, values: Values): Promise<Experiment> {
  const height = values['height']!;
  const fill = (r: number) =>
    lattice([-HALF_X, LIFT + r, -HALF_Z], [-HALF_X + COLUMN, LIFT + height, HALF_Z], r * 2);
  const radius = fitRadius(fill, ctx.particles, 0.015);
  const initial = fill(radius);
  const particles = new ParticleSystem(ctx.renderer, initial.length, radius);
  particles.uploadParticles(initial);
  const fluid = new FluidSystem(particles, {
    viscosity: values['viscosity']!,
    surfaceTension: values['tension']!,
    vorticity: 0.02,
    sortByCell: true,
  });

  const walls = tank(particles, HALF_X, HALF_Z);
  const bunny = await loadBunny();
  // Face the bunny toward the oncoming water.
  const yaw = BUNNY_YAW - Math.PI / 2;
  bunny.mesh.rotation.y = yaw;
  bunny.mesh.scale.setScalar(BUNNY_SCALE);
  const bunnyCollider = new SDFCollider(particles, bunny.sdf, {
    rotation: new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), yaw),
    scale: BUNNY_SCALE,
    muS: 0.3,
    muK: 0.2,
  });

  const substeps = scaledSubsteps(4, ctx.particles),
    iterations = 2;
  const loop = new SimLoop(particles, {
    substeps,
    iterations,
    gravity: new Vector3(0, -values['gravity']!, 0),
    materials: [fluid],
    colliders: [walls, bunnyCollider],
  });

  const visual = liquidVisual(ctx, fluid, {
    bounds: new Box3(
      new Vector3(-HALF_X - 0.03, -0.02, -HALF_Z - 0.03),
      new Vector3(HALF_X + 0.03, LIFT + 1.1, HALF_Z + 0.03),
    ),
    colliders: [walls, bunnyCollider],
    motionStretch: 0.02,
    appearance: { color: 0x2f9e4f, attenuationDistance: 0.16, scattering: 0.18, roughness: 0.06 },
  });

  return {
    particles,
    loop,
    objects: [
      basin(HALF_X * 2, HALF_Z * 2, 1.05, undefined, 0.015),
      bunny.mesh,
      visual.surface.mesh,
      visual.dots,
    ],
    particleCount: initial.length,
    substeps,
    iterations,
    prepareRender: () => visual.update(),
    setReflections: (enabled) => (visual.surface.reflections = enabled),
    setParticleView: (enabled) => visual.setParticleView(enabled),
    setParameter(key, value) {
      values[key] = value;
      if (key === 'gravity') loop.gravity.y = -value;
      if (key === 'viscosity') fluid.viscosity = value;
      if (key === 'tension') fluid.surfaceTension = value;
    },
    dispose() {
      visual.surface.dispose();
      bunnyCollider.dispose();
      particles.dispose();
      loop.dispose();
    },
  };
}
