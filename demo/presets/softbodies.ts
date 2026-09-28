import {
  Box3,
  Color,
  Euler,
  Group,
  MeshStandardMaterial,
  Quaternion,
  Vector3,
  type BufferGeometry,
} from 'three';
import {
  FluidSystem,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  SoftbodyMesh,
  SoftbodySystem,
  voxelize,
  type ParticleInit,
  type SDFData,
  type SoftbodyDef,
  type VoxelizeResult,
} from '../../src/index.js';
import { basin, block, glassTank, platform } from '../runtime/stage.js';
import type { BuildContext, Experiment, Values } from '../types.js';
import { compliance, loadElasticMaterial, loadElasticMesh } from './elastic.js';
import { loadBunny } from './honey.js';
import { liquidVisual } from './liquids.js';
import { lattice, particleView, scaledSubsteps, seededRandom } from './shared.js';

/**
 * Fill a distance field with close to `count` particles, keeping the largest
 * connected piece. Lattice counts fall as the spacing grows, so bisect on it.
 * A slight dilation keeps thin features such as ears attached.
 */
function sampleBody(sdf: SDFData, count: number): { shape: VoxelizeResult; radius: number } {
  const fill = (radius: number) =>
    voxelize(sdf, { particleRadius: radius, largestPiece: true, dilation: 0.4 * radius });
  let low = 0.0015,
    high = 0.05;
  for (let step = 0; step < 24; step++) {
    const mid = (low + high) / 2;
    if (fill(mid).count > count) low = mid;
    else high = mid;
  }
  return { shape: fill(low), radius: low };
}

const LINEUP = 5;
// Faces the camera, which looks down -z at the row.
const LINEUP_YAW = Math.PI / 2;
const FIRM = new Color(0x9fb3c8);
const JELLY = new Color(0xe8607e);

export async function buildBunnyLineup(ctx: BuildContext, values: Values): Promise<Experiment> {
  const bunny = await loadBunny();
  const { shape, radius } = sampleBody(bunny.sdf, Math.round(ctx.particles / LINEUP));
  const yaw = new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), LINEUP_YAW);
  const initial: ParticleInit[] = [];
  const bodies: SoftbodyDef[] = [];
  const geometries: BufferGeometry[] = [];
  for (let n = 0; n < LINEUP; n++) {
    const offset = new Vector3((n - (LINEUP - 1) / 2) * 0.56, values['height']!, 0);
    const start = initial.length;
    for (let i = 0; i < shape.count; i++) {
      const p = new Vector3()
        .fromArray(shape.positions, i * 3)
        .applyQuaternion(yaw)
        .add(offset);
      initial.push({ position: [p.x, p.y, p.z], invMass: shape.count / 1000 });
    }
    bodies.push({
      range: { start, count: shape.count },
      surfaceCount: shape.surfaceCount,
      // Left to right, firm rubber to loose jelly: each twice as soft as the last.
      compliance: 5e-5 * 2 ** (n - (LINEUP - 1)) * values['softness']!,
      edges: shape.edges,
    });
    geometries.push(
      bunny.mesh.geometry.clone().applyQuaternion(yaw).translate(offset.x, offset.y, offset.z),
    );
  }
  bunny.mesh.geometry.dispose();
  (bunny.mesh.material as MeshStandardMaterial).dispose();
  const particles = new ParticleSystem(ctx.renderer, initial.length, radius);
  particles.uploadParticles(initial);
  // Local shape matching keeps the thin ears from shearing off.
  const bunnies = new SoftbodySystem(particles, { bodies, shapeMatching: 'local' });
  const floor = new PrimitiveSet(particles);
  floor.addPlane(new Vector3(0, 1, 0), new Vector3(), { muS: 0.6, muK: 0.5 });
  const substeps = scaledSubsteps(6, ctx.particles),
    iterations = 2;
  const loop = new SimLoop(particles, {
    substeps,
    iterations,
    gravity: new Vector3(0, -values['gravity']!, 0),
    materials: [bunnies],
    colliders: [floor],
  });
  const materials = geometries.map(
    (_, n) =>
      new MeshStandardMaterial({
        color: FIRM.clone().lerp(JELLY, n / (LINEUP - 1)),
        roughness: 0.35 - (0.2 * n) / (LINEUP - 1),
        metalness: 0,
      }),
  );
  const meshes = geometries.map(
    (geometry, n) => new SoftbodyMesh(bunnies, n, geometry, materials[n]),
  );
  const dots = particleView(particles, { color: 0xd8bdd2 });
  return {
    particles,
    loop,
    objects: [basin(2.9, 0.9), ...meshes, dots],
    particleCount: initial.length,
    substeps,
    iterations,
    setParameter(key, value) {
      values[key] = value;
      if (key === 'gravity') loop.gravity.y = -value;
    },
    setParticleView(enabled) {
      for (const mesh of meshes) mesh.visible = !enabled;
      dots.visible = enabled;
    },
    dispose() {
      for (const material of materials) material.dispose();
      particles.dispose();
      loop.dispose();
    },
  };
}

const BANANAS = 8;
const BANANA_LENGTH = 0.41;
const JAR_HALF = 0.35;
const JAR_HEIGHT = 1.0;
const JAR_FLOOR = 0.26;
const FILL = 0.75;
// Short paddles up the shaft, each pointing opposite the one below it.
const PADDLE_REACH = 0.29;
const PADDLE_HALF_HEIGHT = 0.05;
const PADDLES = [0.1, 0.24, 0.38, 0.52, 0.66, 0.8];

export async function buildBlender(ctx: BuildContext, values: Values): Promise<Experiment> {
  const [template, sourceMaterial] = await Promise.all([
    loadElasticMesh('banana'),
    loadElasticMaterial(ctx.renderer),
  ]);
  // Liquid and bananas together fill 3/4 of the jar with the level's particles.
  // Every particle shares one size, so the bananas are voxelized at it too.
  const spacing = Math.cbrt(((JAR_HALF * 2) ** 2 * JAR_HEIGHT * FILL) / ctx.particles);
  const radius = spacing / 2;
  template.computeBoundingBox();
  const size = template.boundingBox!.getSize(new Vector3());
  const scale = BANANA_LENGTH / Math.max(size.x, size.y, size.z);
  template.scale(scale, scale, scale).center();
  const shape = voxelize(template, { particleRadius: radius });

  // Keep bananas inside the liquid and clear of the shaft and paddles.
  const fillTop = JAR_FLOOR + JAR_HEIGHT * FILL;
  const blocked = (x: number, y: number, z: number, margin: number) =>
    Math.abs(x) > JAR_HALF - margin ||
    Math.abs(z) > JAR_HALF - margin ||
    y < JAR_FLOOR + margin ||
    y > fillTop - margin ||
    Math.hypot(x, z) < 0.03 + margin ||
    PADDLES.some(
      (height, k) =>
        Math.abs(z) < 0.012 + margin &&
        Math.abs(y - JAR_FLOOR - height) < PADDLE_HALF_HEIGHT + margin &&
        (k % 2 ? -x : x) > -margin &&
        Math.abs(x) < PADDLE_REACH + margin,
    );
  const cell = (x: number, y: number, z: number) =>
    `${Math.floor(x / spacing)},${Math.floor(y / spacing)},${Math.floor(z / spacing)}`;
  const occupied = new Set<string>();
  const nearBanana = (x: number, y: number, z: number) => {
    for (let a = -1; a <= 1; a++)
      for (let b = -1; b <= 1; b++)
        for (let c = -1; c <= 1; c++)
          if (occupied.has(cell(x + a * spacing, y + b * spacing, z + c * spacing))) return true;
    return false;
  };
  const random = seededRandom(0x2545f491);
  const bananas: Vector3[][] = [];
  const geometries: BufferGeometry[] = [];
  for (let n = 0; n < BANANAS; n++) {
    // Scatter at random, rejecting placements that touch the jar, the rotor,
    // or a banana already placed.
    for (let attempt = 0; attempt < 400; attempt++) {
      const center = new Vector3(
        (random() - 0.5) * JAR_HALF * 1.6,
        JAR_FLOOR + 0.1 + random() * (fillTop - JAR_FLOOR - 0.2),
        (random() - 0.5) * JAR_HALF * 1.6,
      );
      const rotation = new Quaternion().setFromEuler(
        new Euler(random() * Math.PI * 2, random() * Math.PI * 2, random() * Math.PI * 2),
      );
      const points = Array.from({ length: shape.count }, (_, i) =>
        new Vector3()
          .fromArray(shape.positions, i * 3)
          .applyQuaternion(rotation)
          .add(center),
      );
      if (points.some((p) => blocked(p.x, p.y, p.z, radius) || nearBanana(p.x, p.y, p.z))) continue;
      for (const p of points) occupied.add(cell(p.x, p.y, p.z));
      bananas.push(points);
      geometries.push(
        template.clone().applyQuaternion(rotation).translate(center.x, center.y, center.z),
      );
      break;
    }
  }
  template.dispose();

  const water = lattice(
    [-JAR_HALF, JAR_FLOOR, -JAR_HALF],
    [JAR_HALF, fillTop, JAR_HALF],
    spacing,
    (x, y, z) => !nearBanana(x, y, z) && !blocked(x, y, z, -radius),
  );
  const waterCount = water.length;
  const initial: ParticleInit[] = [...water];
  const bodies: SoftbodyDef[] = bananas.map((banana) => {
    const start = initial.length;
    // As dense as the liquid, so the bananas hang in suspension.
    for (const p of banana)
      initial.push({ position: [p.x, p.y, p.z], invMass: 1 / (1000 * spacing ** 3) });
    return {
      range: { start, count: shape.count },
      surfaceCount: shape.surfaceCount,
      compliance: compliance(values['softness']!, shape.count),
      edges: shape.edges,
    };
  });
  const particles = new ParticleSystem(ctx.renderer, initial.length, radius);
  particles.uploadParticles(initial);
  const softbody = new SoftbodySystem(particles, { bodies, shapeMatching: 'local' });
  const fluid = new FluidSystem(particles, {
    range: { start: 0, count: waterCount },
    viscosity: values['viscosity']!,
    surfaceTension: 0.05,
    sortByCell: true,
  });
  for (let i = 0; i < bodies.length; i++) fluid.addBoundary(softbody.surfaceRange(i));

  const colliders = new PrimitiveSet(particles);
  const wall = { muS: 0.3, muK: 0.2 };
  colliders.addPlane(new Vector3(0, 1, 0), new Vector3(0, JAR_FLOOR, 0), wall);
  colliders.addPlane(new Vector3(0, -1, 0), new Vector3(0, JAR_FLOOR + JAR_HEIGHT, 0), wall);
  for (const axis of [new Vector3(1, 0, 0), new Vector3(0, 0, 1)])
    for (const side of [-1, 1])
      colliders.addPlane(
        axis.clone().multiplyScalar(-side),
        axis.clone().multiplyScalar(side * JAR_HALF),
        wall,
      );
  // Each paddle is a box that follows its mesh on the spinning rotor; the
  // shaft is a fixed capsule on the axis.
  const rotor = new Group();
  rotor.position.y = JAR_FLOOR;
  const steel = new MeshStandardMaterial({ color: 0xc9d1d6, roughness: 0.22, metalness: 0.9 });
  const shaft = block([0.06, JAR_HEIGHT, 0.06], [0, JAR_HEIGHT / 2, 0], 0x8b979e, 0.025);
  shaft.material = steel;
  rotor.add(shaft);
  for (const [k, height] of PADDLES.entries()) {
    const paddle = block(
      [PADDLE_REACH, PADDLE_HALF_HEIGHT * 2, 0.024],
      [((k % 2 ? -1 : 1) * PADDLE_REACH) / 2, height, 0],
      0xc9d1d6,
      0.008,
    );
    paddle.material = steel;
    rotor.add(paddle);
    colliders.attach(
      colliders.addBox(
        new Vector3(),
        new Vector3(PADDLE_REACH / 2, PADDLE_HALF_HEIGHT, 0.012),
        wall,
      ),
      paddle,
    );
  }
  colliders.addCapsule(
    new Vector3(0, JAR_FLOOR, 0),
    new Vector3(0, JAR_FLOOR + JAR_HEIGHT, 0),
    0.03,
    wall,
  );

  const substeps = scaledSubsteps(5, ctx.particles),
    iterations = 2;
  const loop = new SimLoop(particles, {
    substeps,
    iterations,
    gravity: new Vector3(0, -values['gravity']!, 0),
    materials: [fluid, softbody],
    colliders: [colliders],
    contact: { muS: 0.2, muK: 0.1 },
  });
  const meshes = geometries.map(
    (geometry, i) => new SoftbodyMesh(softbody, i, geometry, sourceMaterial),
  );
  const visual = liquidVisual(ctx, fluid, {
    bounds: new Box3(
      new Vector3(-JAR_HALF - 0.03, JAR_FLOOR - 0.02, -JAR_HALF - 0.03),
      new Vector3(JAR_HALF + 0.03, JAR_FLOOR + JAR_HEIGHT + 0.02, JAR_HALF + 0.03),
    ),
    colliders: [colliders],
    solids: { start: waterCount, count: initial.length - waterCount },
    appearance: { color: 0x9fd8ec, attenuationDistance: 1.2, scattering: 0.04, roughness: 0.06 },
    // Bent rays smear dark patches where bananas cross the surface.
    refraction: false,
  });
  const motor = block(
    [0.86, JAR_FLOOR - 0.01, 0.86],
    [0, (JAR_FLOOR - 0.01) / 2, 0],
    0x26343f,
    0.06,
  );
  const jar = glassTank(JAR_HALF * 2, JAR_HALF * 2, JAR_HEIGHT);
  jar.position.y = JAR_FLOOR;
  let angle = 0;
  return {
    particles,
    loop,
    objects: [platform(1.1, 1.1), motor, jar, rotor, ...meshes, visual.surface.mesh, visual.dots],
    particleCount: initial.length,
    substeps,
    iterations,
    update(dt) {
      angle = (angle + dt * (values['rpm']! / 60) * Math.PI * 2) % (Math.PI * 2);
      rotor.rotation.y = angle;
    },
    prepareRender: () => visual.update(),
    setReflections: (enabled) => (visual.surface.reflections = enabled),
    setParticleView(enabled) {
      visual.setParticleView(enabled);
      for (const mesh of meshes) mesh.visible = !enabled;
    },
    setParameter(key, value) {
      values[key] = value;
      if (key === 'gravity') loop.gravity.y = -value;
      if (key === 'viscosity') fluid.viscosity = value;
    },
    dispose() {
      visual.surface.dispose();
      sourceMaterial.map?.dispose();
      sourceMaterial.dispose();
      particles.dispose();
      loop.dispose();
    },
  };
}
