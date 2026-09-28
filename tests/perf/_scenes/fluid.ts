// `fluid` scenes: a column of water falls onto a floor plane. Position Based
// Fluids pressure solve with XSPH viscosity and vorticity confinement; no
// particle contacts and no boundary particles.

import { Vector3 } from 'three';
import { FluidSystem, ParticleSystem, PrimitiveSet, SimLoop } from '../../../src/index.js';

import type { PerfRenderer } from '../_helpers/PerfRenderer.js';
import { FRAME_DT, fluidColumn, type BuiltScene } from './_helpers.js';

const SPACING = 0.025;
const SMOOTHING_RADIUS = 0.04;
const SUBSTEPS = 4;
const ITERATIONS = 2;

function buildFluidScene(
  perf: PerfRenderer,
  id: string,
  count: number,
  origin: readonly [number, number, number],
  shuffled = false,
  sortByCell = false,
): BuiltScene {
  const particles = new ParticleSystem(perf.renderer, count, SPACING / 2);
  const column = fluidColumn({ count, spacing: SPACING, origin });
  particles.uploadParticles(shuffled ? shuffle(column) : column);

  const floor = new PrimitiveSet(particles);
  floor.addPlane(new Vector3(0, 1, 0), new Vector3());

  const fluid = new FluidSystem(particles, {
    smoothingRadius: SMOOTHING_RADIUS,
    viscosity: 0.01,
    vorticity: 0.1,
    sortByCell,
  });
  const loop = new SimLoop(particles, {
    substeps: SUBSTEPS,
    iterations: ITERATIONS,
    materials: [fluid],
    colliders: [floor],
  });

  return {
    spec: {
      id,
      particleCount: count,
      substeps: SUBSTEPS,
      iterations: ITERATIONS,
      stepFrame: () => loop.step(FRAME_DT),
    },
    dispose: () => {
      loop.dispose();
      floor.dispose();
      particles.dispose();
    },
  };
}

export function buildFluid10kScene(perf: PerfRenderer): BuiltScene {
  return buildFluidScene(perf, 'fluid-10k', 10_000, [-0.25, 0.5, -0.25]);
}

export function buildFluid100kScene(perf: PerfRenderer): BuiltScene {
  return buildFluidScene(perf, 'fluid-100k', 100_000, [-0.6, 0.5, -0.6]);
}

/**
 * The 100k scene with its particles uploaded in random order. After a long
 * run, mixing leaves a fluid's particle order with no spatial pattern; this
 * starts it that way, so neighbor walks read memory as they would then.
 */
export function buildFluid100kShuffledScene(perf: PerfRenderer): BuiltScene {
  return buildFluidScene(perf, 'fluid-100k-shuffled', 100_000, [-0.6, 0.5, -0.6], true);
}

/** The shuffled scene with `sortByCell`, which puts the particles back in grid order every step. */
export function buildFluid100kShuffledSortedScene(perf: PerfRenderer): BuiltScene {
  return buildFluidScene(
    perf,
    'fluid-100k-shuffled-sorted',
    100_000,
    [-0.6, 0.5, -0.6],
    true,
    true,
  );
}

/** Fisher–Yates shuffle with a fixed seed, so every run starts the same. */
function shuffle<T>(items: T[]): T[] {
  let seed = 1;
  const random = (): number => {
    seed = (seed * 16807) % 2147483647;
    return seed / 2147483647;
  };
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [items[i], items[j]] = [items[j]!, items[i]!];
  }
  return items;
}
