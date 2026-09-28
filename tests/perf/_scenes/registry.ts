// Scenes the profile and fingerprint suites can run: every demo preset,
// built exactly as the demo builds it, and the benchmark scenes.

import { PerspectiveCamera, Scene, Vector3 } from 'three';

import { presets, defaults } from '../../../demo/presets/index.js';
import { particleCount, type ParticleLevel } from '../../../demo/types.js';
import type { ParticleSystem } from '../../../src/index.js';
import type { PerfRenderer } from '../_helpers/PerfRenderer.js';
import type { BuiltScene } from './_helpers.js';
import {
  buildFluid10kScene,
  buildFluid100kScene,
  buildFluid100kShuffledScene,
  buildFluid100kShuffledSortedScene,
} from './fluid.js';
import { buildFluidBodies10kScene, buildFluidBodies100kScene } from './fluid-bodies.js';
import {
  buildFluidSurfaceTension10kScene,
  buildFluidSurfaceTension100kScene,
} from './fluid-surface-tension.js';
import { buildSoftbody10kScene, buildSoftbody100kScene } from './softbody.js';

const FRAME_DT = 1 / 60;

/** One frame of a scene, split into its simulation and render-preparation work. */
export interface ProfileScene {
  /**
   * Frames to run before measuring when none are asked for: half the
   * preset's demo duration, so scenes that build up (a pour, a flood) are
   * measured mid-run. Absent for benchmark scenes.
   */
  readonly warmupFrames?: number;
  /** The scene's particles, for readback. Absent for benchmark scenes. */
  readonly particles?: ParticleSystem;
  readonly id: string;
  readonly particleCount: number;
  readonly substeps: number;
  readonly iterations: number;
  readonly simulate: () => Promise<void>;
  readonly prepareRender?: () => Promise<void>;
  readonly dispose: () => void;
}

export type SceneFactory = (perf: PerfRenderer, level: ParticleLevel) => Promise<ProfileScene>;

function fromBenchmark(build: (perf: PerfRenderer) => BuiltScene): SceneFactory {
  return async (perf) => {
    const built = build(perf);
    const { spec } = built;
    return {
      id: spec.id,
      particleCount: spec.particleCount,
      substeps: spec.substeps,
      iterations: spec.iterations,
      simulate: spec.stepFrame,
      dispose: built.dispose,
    };
  };
}

function fromPreset(id: string): SceneFactory {
  return async (perf, level) => {
    const preset = presets.find((p) => p.id === id)!;
    const camera = new PerspectiveCamera(38, 16 / 9, 0.02, 100);
    camera.position.set(...preset.camera);
    camera.lookAt(new Vector3(...preset.target));
    const experiment = await preset.build(
      {
        renderer: perf.renderer,
        scene: new Scene(),
        camera,
        particles: particleCount(preset, level),
      },
      defaults(preset),
    );
    let time = 0;
    return {
      id: preset.id,
      warmupFrames: Math.round((preset.duration / 2) * 60),
      particles: experiment.particles,
      particleCount: experiment.particleCount,
      substeps: experiment.substeps,
      iterations: experiment.iterations,
      simulate: async () => {
        await experiment.update?.(FRAME_DT, time);
        await experiment.loop.step(FRAME_DT);
        time += FRAME_DT;
      },
      ...(experiment.prepareRender && {
        prepareRender: async () => experiment.prepareRender!(),
      }),
      dispose: () => experiment.dispose(),
    };
  };
}

export const SCENES: Record<string, SceneFactory> = {
  ...Object.fromEntries(presets.map((p) => [p.id, fromPreset(p.id)])),
  'fluid-10k': fromBenchmark(buildFluid10kScene),
  'fluid-100k': fromBenchmark(buildFluid100kScene),
  'fluid-100k-shuffled': fromBenchmark(buildFluid100kShuffledScene),
  'fluid-100k-shuffled-sorted': fromBenchmark(buildFluid100kShuffledSortedScene),
  'fluid-bodies-10k': fromBenchmark(buildFluidBodies10kScene),
  'fluid-bodies-100k': fromBenchmark(buildFluidBodies100kScene),
  'fluid-surface-tension-10k': fromBenchmark(buildFluidSurfaceTension10kScene),
  'fluid-surface-tension-100k': fromBenchmark(buildFluidSurfaceTension100kScene),
  'softbody-10k': fromBenchmark(buildSoftbody10kScene),
  'softbody-100k': fromBenchmark(buildSoftbody100kScene),
};

/** Ids of every demo preset, the default scene list. */
export const PRESET_IDS: readonly string[] = presets.map((p) => p.id);
