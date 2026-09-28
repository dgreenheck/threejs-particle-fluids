import {
  Box3,
  CanvasTexture,
  Mesh,
  RepeatWrapping,
  SRGBColorSpace,
  Vector3,
  type BufferGeometry,
  type MeshStandardMaterial,
  type Texture,
} from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import {
  FluidSystem,
  ParticleSystem,
  SimLoop,
  SoftbodyMesh,
  SoftbodySystem,
  voxelize,
  type ParticleInit,
  type SoftbodyDef,
} from '../../src/index.js';
import { basin } from '../runtime/stage.js';
import type { BuildContext, Experiment, Values } from '../types.js';
import { liquidVisual } from './liquids.js';
import { fitRadius, lattice, particleView, scaledSubsteps, tank } from './shared.js';

/** How many times heavier a duck's base particles are than the rest of it. */
const BASE_WEIGHT = 4;

export async function buildBuoyancy(ctx: BuildContext, values: Values): Promise<Experiment> {
  const fill = (r: number) => lattice([-0.76, r, -0.5], [0.76, 0.42, 0.5], r * 2);
  const radius = fitRadius(fill, ctx.particles, 0.021);
  const spacing = radius * 2;
  const initial: ParticleInit[] = fill(radius);
  const waterCount = initial.length;

  const model = await new GLTFLoader().loadAsync(
    `${import.meta.env.BASE_URL}models/buoyancy/rubber-duck.glb`,
  );
  model.scene.updateMatrixWorld(true);
  const source = model.scene.getObjectByProperty('isMesh', true) as
    | Mesh<BufferGeometry, MeshStandardMaterial>
    | undefined;
  if (!source) throw new Error('The rubber duck asset has no mesh.');
  const geometries = [0.92, 1, 0.86].map((size, i) =>
    source.geometry
      .clone()
      .applyMatrix4(source.matrixWorld)
      .scale(size, size, size)
      .rotateY([-0.5, 0.65, -0.8][i]!)
      .translate((i - 1) * 0.45, 0.47, [0.08, -0.12, 0.06][i]!),
  );
  // Each duck is a stiff body with the requested density relative to water.
  // Like a weighted toy, most of its mass sits in its lowest third, which
  // keeps it floating upright.
  const meanInvMass = 1 / (1000 * spacing ** 3 * values['density']!);
  const bodies: SoftbodyDef[] = geometries.map((geometry) => {
    const shape = voxelize(geometry, { particleRadius: radius });
    const start = initial.length;
    const heights = Array.from({ length: shape.count }, (_, i) => shape.positions[i * 3 + 1]!);
    const bottom = heights.reduce((a, b) => Math.min(a, b));
    const top = heights.reduce((a, b) => Math.max(a, b));
    const inBase = heights.map((y) => y < bottom + (top - bottom) / 3);
    const baseCount = inBase.filter(Boolean).length;
    // Scale the masses so the duck's mean density stays as requested.
    const invMass =
      (meanInvMass * (baseCount * BASE_WEIGHT + shape.count - baseCount)) / shape.count;
    for (let i = 0; i < shape.count; i++) {
      const [x, y, z] = shape.positions.subarray(i * 3, i * 3 + 3);
      initial.push({
        position: [x!, y!, z!],
        invMass: inBase[i] ? invMass / BASE_WEIGHT : invMass,
      });
    }
    return {
      range: { start, count: shape.count },
      surfaceCount: shape.surfaceCount,
      compliance: 1e-6,
    };
  });
  const particles = new ParticleSystem(ctx.renderer, initial.length, radius);
  particles.uploadParticles(initial);
  const ducks = new SoftbodySystem(particles, { bodies });
  const water = new FluidSystem(particles, {
    range: { start: 0, count: waterCount },
    viscosity: values['viscosity']!,
    surfaceTension: values['tension']!,
    sortByCell: true,
  });
  for (let i = 0; i < bodies.length; i++) water.addBoundary(ducks.surfaceRange(i));
  const walls = tank(particles, 0.8, 0.55);
  const substeps = scaledSubsteps(5, ctx.particles),
    iterations = 2;
  const loop = new SimLoop(particles, {
    substeps,
    iterations,
    gravity: new Vector3(0, -values['gravity']!, 0),
    materials: [water, ducks],
    colliders: [walls],
    contact: { muS: 0.35, muK: 0.2 },
  });

  const meshes = geometries.map(
    (geometry, i) => new SoftbodyMesh(ducks, i, geometry, source.material),
  );
  const visual = liquidVisual(ctx, water, {
    bounds: new Box3(new Vector3(-0.83, -0.02, -0.58), new Vector3(0.83, 0.85, 0.58)),
    colliders: [walls],
    solids: { start: waterCount, count: initial.length - waterCount },
    appearance: {
      color: 0x9fd8ec,
      attenuationDistance: 1.2,
      scattering: 0.04,
      roughness: 0.055,
    },
  });
  // Ceramic pool tiles make the water's transparency and refraction readable.
  const tileTexture = poolTiles();
  const tray = basin(1.65, 1.15, 0.39, tileTexture);
  const dots = particleView(particles, { color: 0xdabec8 });
  return {
    particles,
    loop,
    objects: [tray, ...meshes, visual.surface.mesh, dots],
    particleCount: initial.length,
    substeps,
    iterations,
    prepareRender: () => visual.update(),
    setReflections: (enabled) => (visual.surface.reflections = enabled),
    setParticleView(enabled) {
      // Show the ducks' particles along with the water's.
      visual.setParticleView(enabled);
      visual.dots.visible = false;
      dots.visible = enabled;
      for (const mesh of meshes) mesh.visible = !enabled;
    },
    setParameter(key, value) {
      values[key] = value;
      if (key === 'gravity') loop.gravity.y = -value;
      if (key === 'viscosity') water.viscosity = value;
      if (key === 'tension') water.surfaceTension = value;
    },
    dispose() {
      tileTexture.dispose();
      const textures = new Set<Texture>();
      for (const value of Object.values(source.material))
        if (value && typeof value === 'object' && 'isTexture' in value)
          textures.add(value as Texture);
      for (const texture of textures) texture.dispose();
      source.material.dispose();
      source.geometry.dispose();
      visual.surface.dispose();
      particles.dispose();
      loop.dispose();
    },
  };
}

function poolTiles(): CanvasTexture {
  const tile = document.createElement('canvas');
  tile.width = tile.height = 128;
  const paint = tile.getContext('2d')!;
  paint.fillStyle = '#b8d3df';
  paint.fillRect(0, 0, 128, 128);
  paint.strokeStyle = '#87acbf';
  paint.lineWidth = 2;
  paint.strokeRect(0, 0, 128, 128);
  const texture = new CanvasTexture(tile);
  texture.colorSpace = SRGBColorSpace;
  texture.wrapS = texture.wrapT = RepeatWrapping;
  texture.repeat.set(7, 7);
  return texture;
}
