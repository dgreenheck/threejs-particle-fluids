import { Box3, BoxGeometry, CylinderGeometry, Group, Mesh, Vector3 } from 'three';
import { Fn, color, instanceIndex, mix, uniform, vec2, vec3, vec4 } from 'three/tsl';
import {
  FluidSystem,
  GasSpriteRenderer,
  GasSystem,
  GasVolumeRenderer,
  ParticleSystem,
  SimLoop,
  type Material,
} from '../../src/index.js';
import { material, pedestal } from '../runtime/stage.js';
import type { BuildContext, Experiment, Values } from '../types.js';
import { fitRadius, lattice, particleView, scaledSubsteps, tank } from './shared.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

const HALF = 0.5,
  TOP = 1.9,
  VENT = 0.17,
  SMOKE_TOP = 1.7;

export function buildVortex(ctx: BuildContext, values: Values): Experiment {
  // Still air fills the column. The vent heats it, and the hot air rises and
  // carries the smoke.
  const fill = (r: number) => lattice([-HALF, r, -HALF], [HALF, TOP, HALF], r * 2);
  const radius = fitRadius(fill, ctx.particles, 0.035);
  const initial = fill(radius);
  const count = initial.length;
  const detailed = ctx.particles >= 25000;
  const particles = new ParticleSystem(ctx.renderer, count, radius);
  particles.uploadParticles(initial);
  const air = new FluidSystem(particles, {
    viscosity: 0.02,
    vorticity: 0.06,
    sortByCell: true,
  });
  const smoke = new GasSystem(air, {
    capacity: detailed ? 42000 : 30000,
    lifetime: 6,
    // Smoke vents out near the top instead of pooling there.
    bounds: new Box3(new Vector3(-1, -1, -1), new Vector3(1, SMOKE_TOP, 1)),
    heatSources: [{ position: new Vector3(), radius: VENT }],
    buoyancy: values['heat']!,
    cooling: 0.6,
  });

  // Vanes along the walls turn the air the plume draws in, so the column spins.
  const swirl = uniform(values['swirl']!);
  const vanes: Material = {
    build: ({ dt }) => ({
      postSolve: [
        Fn(() => {
          const p: Any = particles.positions.element(instanceIndex).xyz;
          const velocity: Any = particles.velocities.element(instanceIndex);
          const v: Any = velocity.xyz.toVar();
          const r: Any = vec2(p.x, p.z).length().toVar();
          const tangent: Any = vec3(p.z.negate(), 0, p.x).div(r.max(1e-4));
          const turn: Any = swirl.sub(v.dot(tangent)).mul(r.smoothstep(0.3, 0.45)).mul(dt.mul(4));
          velocity.assign(vec4(v.add(tangent.mul(turn)), velocity.w));
        })().compute(count),
      ],
    }),
  };

  // A lid keeps hot air from leaving through the top.
  const walls = tank(particles, HALF + 0.02, HALF + 0.02);
  walls.addPlane(new Vector3(0, -1, 0), new Vector3(0, TOP + 0.02, 0));
  const substeps = scaledSubsteps(2, ctx.particles),
    iterations = 2;
  const loop = new SimLoop(particles, {
    substeps,
    iterations,
    // Light gravity keeps the air settled, so gaps the plume leaves refill.
    gravity: new Vector3(0, -1, 0),
    colliders: [walls],
    materials: [smoke, air, vanes],
  });

  const sprites = new GasSpriteRenderer(smoke, {
    size: 0.04,
    initialOpacity: 0.5,
    opacityTau: 3,
    colorNode: () => color(0xe9eef2),
  });
  sprites.object.visible = false;
  const volume = new GasVolumeRenderer(smoke, {
    renderer: ctx.renderer,
    min: new Vector3(-HALF, 0, -HALF),
    max: new Vector3(HALF, SMOKE_TOP, HALF),
    resolution: [80, 128, 80],
    steps: detailed ? 96 : 80,
    density: values['density']!,
  });
  // Air particles glow with their temperature.
  const temperature = smoke.temperature!;
  const dots = particleView(particles, {
    radius: radius * 0.22,
    colorNode: () =>
      mix(
        color(0x6d8a9c),
        color(0xff8a3d),
        temperature.element(instanceIndex).clamp(0, 1).pow(0.4),
      ),
  });

  const vent = new Group();
  const recess = new Mesh(new CylinderGeometry(VENT, VENT, 0.006, 64), material(0x0b141d, 0.8, 0));
  recess.position.y = 0.005;
  vent.add(recess);
  const grilleMaterial = material(0x738894, 0.35, 0.7);
  for (let i = -5; i <= 5; i++) {
    const x = i * 0.028;
    const slat = new Mesh(
      new BoxGeometry(0.008, 0.008, Math.sqrt((VENT - 0.01) ** 2 - x ** 2) * 2),
      grilleMaterial,
    );
    slat.position.set(x, 0.012, 0);
    vent.add(slat);
  }

  let emissionCarry = 0;
  return {
    particles,
    loop,
    objects: [pedestal(0.58), vent, volume.object, sprites.object, dots],
    get particleCount() {
      return count + smoke.aliveCount;
    },
    substeps,
    iterations,
    prepareRender: () => volume.update(),
    update(dt) {
      emissionCarry += values['emission']! * dt;
      const n = Math.floor(emissionCarry);
      emissionCarry -= n;
      for (let i = 0; i < n; i++) {
        const angle = Math.random() * Math.PI * 2;
        const r = (VENT - 0.02) * Math.sqrt(Math.random());
        smoke.emit([Math.cos(angle) * r, 0.03 + Math.random() * 0.05, Math.sin(angle) * r]);
      }
    },
    setParameter(key, value) {
      values[key] = value;
      if (key === 'swirl') swirl.value = value;
      if (key === 'heat') smoke.buoyancy = value;
      if (key === 'density') volume.density = value;
    },
    setParticleView(enabled) {
      dots.visible = enabled;
      sprites.object.visible = enabled;
      volume.object.visible = !enabled;
    },
    dispose() {
      sprites.dispose();
      volume.dispose();
      particles.dispose();
      loop.dispose();
    },
  };
}
