import { describe, expect, it } from 'vitest';
import { Vector3 } from 'three';
import { instancedArray, uniform } from 'three/tsl';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';
import type { WebGPURenderer } from 'three/webgpu';
import {
  FluidSystem,
  HashGrid,
  ParticleSystem,
  PrimitiveSet,
  SimLoop,
  createParticleRenderer,
  type ParticleInit,
} from '../../../src/index.js';
import { buildCellSortKernels } from '../../../src/fluids/sim/cellSort.js';

// `sortByCell` moves a fluid's particles into the neighbor grid's cell order.
//
// 1. The permutation itself is exact: every per-particle buffer moves whole,
//    bit for bit, into the order the grid's counting sort gave the fluid's
//    particles, and nothing outside the fluid's range changes.
// 2. A sorted fluid behaves like an unsorted one. Summation order changes,
//    so the two runs are not bitwise equal (nor are two unsorted runs: the
//    grid's atomic scatter already varies neighbor order); their settled
//    state must agree to within the spread of ordinary runs.

function lcg(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

async function read(
  renderer: WebGPURenderer,
  node: { value: { array?: unknown } },
): Promise<ArrayBuffer> {
  // A buffer no kernel has used yet exists only on the CPU, where it is still current.
  const backend = renderer.backend as unknown as { get(o: object): { buffer?: unknown } };
  if (backend.get(node.value).buffer === undefined) {
    return (node.value.array as Float32Array).slice().buffer;
  }
  return renderer.getArrayBufferAsync(node.value as never);
}

describe('fluid cell sort', () => {
  it('permutes every per-particle buffer exactly into cell order', async () => {
    const renderer = await createParticleRenderer();
    try {
      const rand = lcg(0x5eed);
      const solids = 700;
      const fluidCount = 2300;
      const capacity = solids + fluidCount;
      const particles = new ParticleSystem(renderer, capacity, 0.01);
      const init: ParticleInit[] = [];
      for (let i = 0; i < capacity; i++) {
        init.push({
          position: [rand() * 0.6 - 0.3, rand() * 0.6, rand() * 0.6 - 0.3],
          velocity: [rand() - 0.5, rand() - 0.5, rand() - 0.5],
          invMass: i < solids ? 0 : 1 + rand(),
          collisionGroup: Math.floor(rand() * 7),
        });
      }
      particles.uploadParticles(init);
      // Distinct values in every buffer, so a lost or duplicated row shows.
      const fill = (node: { value: { array: unknown; needsUpdate: boolean } }, width: number) => {
        const array = node.value.array as Float32Array;
        for (let i = 0; i < capacity * width; i++) array[i] = rand();
        node.value.needsUpdate = true;
      };
      fill(particles.boundaryVolume as never, 1);
      fill(particles.rotation as never, 4);
      fill(particles.predictedRotation as never, 4);
      fill(particles.angularVelocity as never, 4);
      const range = { start: solids, count: fluidCount };
      // A range-local buffer holding each fluid particle's original local index.
      const tag = instancedArray(
        Float32Array.from({ length: fluidCount }, (_, i) => i),
        'float',
      );

      const grid = new HashGrid(particles, { cellSize: 0.04 });
      await grid.rebuild();
      const snapshot = await grid.readback();

      const buffers = [...particles.perParticleBuffers];
      const before = await Promise.all(buffers.map((b) => read(renderer, b.buffer)));
      const enabled = uniform(1, 'uint' as 'float') as unknown as UniformNode<'uint', number>;
      const { kernels } = buildCellSortKernels({
        particles,
        grid,
        range,
        data: [...buffers, { buffer: tag, type: 'float', local: true }],
        enabled,
      });
      await renderer.computeAsync(kernels);
      const after = await Promise.all(buffers.map((b) => read(renderer, b.buffer)));
      const tags = new Float32Array(await read(renderer, tag));

      // The grid's order, restricted to the fluid.
      const expected = Array.from(snapshot.sortedIndices.subarray(0, capacity))
        .filter((p) => p >= solids)
        .map((p) => p - solids);
      expect(Array.from(tags, Math.round)).toEqual(expected);

      buffers.forEach((b, n) => {
        const width = b.type === 'vec4' ? 4 : 1;
        const old = new Uint32Array(before[n]!);
        const now = new Uint32Array(after[n]!);
        // Outside the fluid: untouched.
        expect(now.subarray(0, solids * width)).toEqual(old.subarray(0, solids * width));
        // Inside: row i is the old row of the particle that moved there, bit for bit.
        for (let i = 0; i < fluidCount; i++) {
          const from = (solids + expected[i]!) * width;
          const to = (solids + i) * width;
          for (let c = 0; c < width; c++) {
            if (now[to + c] !== old[from + c]) {
              throw new Error(
                `buffer ${n}: row ${i} component ${c} did not move with its particle`,
              );
            }
          }
        }
      });

      // Nothing moved while disabled.
      enabled.value = 0;
      await renderer.computeAsync(kernels);
      const idle = await Promise.all(buffers.map((b) => read(renderer, b.buffer)));
      idle.forEach((bytes, n) =>
        expect(new Uint32Array(bytes)).toEqual(new Uint32Array(after[n]!)),
      );
    } finally {
      renderer.dispose();
    }
  });

  it('settles a tank of randomly ordered fluid the same with and without sorting', async () => {
    async function settle(sortByCell: boolean) {
      const renderer = await createParticleRenderer();
      try {
        const spacing = 0.025;
        const rand = lcg(0xc0ffee);
        const lattice: [number, number, number][] = [];
        for (let j = 0; j < 12; j++)
          for (let k = 0; k < 8; k++)
            for (let i = 0; i < 8; i++)
              lattice.push([
                -0.1 + (i + 0.5) * spacing,
                (j + 0.5) * spacing,
                -0.1 + (k + 0.5) * spacing,
              ]);
        // Shuffle, so memory order starts unrelated to space.
        for (let i = lattice.length - 1; i > 0; i--) {
          const j = Math.floor(rand() * (i + 1));
          [lattice[i], lattice[j]] = [lattice[j]!, lattice[i]!];
        }
        const particles = new ParticleSystem(renderer, lattice.length, spacing / 2);
        particles.uploadParticles(lattice.map((position) => ({ position })));
        const walls = new PrimitiveSet(particles);
        walls.addPlane(new Vector3(0, 1, 0), new Vector3(0, 0, 0));
        walls.addPlane(new Vector3(1, 0, 0), new Vector3(-0.1, 0, 0));
        walls.addPlane(new Vector3(-1, 0, 0), new Vector3(0.1, 0, 0));
        walls.addPlane(new Vector3(0, 0, 1), new Vector3(0, 0, -0.1));
        walls.addPlane(new Vector3(0, 0, -1), new Vector3(0, 0, 0.1));
        const fluid = new FluidSystem(particles, { viscosity: 0.01, sortByCell });
        const loop = new SimLoop(particles, {
          substeps: 4,
          iterations: 2,
          materials: [fluid],
          colliders: [walls],
        });
        // 9 s to settle, then kinetic energy averaged over the last second:
        // at a single frame it mostly measures where a slosh is in its cycle.
        for (let n = 0; n < 540; n++) await loop.step(1 / 60);
        let ke = 0;
        for (let n = 0; n < 60; n++) {
          await loop.step(1 / 60);
          if (n % 6 !== 5) continue;
          const v = (await particles.readback()).velocities;
          for (let i = 0; i < lattice.length; i++) {
            ke +=
              (0.5 * fluid.mass * (v[i * 4]! ** 2 + v[i * 4 + 1]! ** 2 + v[i * 4 + 2]! ** 2)) / 10;
          }
        }
        const snap = await particles.readback();
        const density = new Float32Array(await read(renderer, fluid.density));
        let sumY = 0;
        let maxDensity = 0;
        let escaped = 0;
        for (let i = 0; i < lattice.length; i++) {
          const x = snap.positions[i * 4]!;
          const y = snap.positions[i * 4 + 1]!;
          const z = snap.positions[i * 4 + 2]!;
          if (!(Math.abs(x) <= 0.1 + 1e-3 && Math.abs(z) <= 0.1 + 1e-3 && y >= -1e-3 && y < 1))
            escaped++;
          sumY += y;
          maxDensity = Math.max(maxDensity, density[i]!);
        }
        loop.dispose();
        walls.dispose();
        particles.dispose();
        return { meanY: sumY / lattice.length, ke, maxDensity, escaped };
      } finally {
        renderer.dispose();
      }
    }

    const plain = await settle(false);
    const sorted = await settle(true);
    expect(sorted.escaped).toBe(0);
    expect(plain.escaped).toBe(0);
    // Measured over nine 10 s runs (three each: unsorted shuffled, unsorted in
    // lattice order, sorted): mean height 0.12502–0.12537 m, peak density
    // 1048–1061 kg/m³, and averaged kinetic energy 0.0018–0.023 J, where the
    // top of that range is a slosh some runs of every kind still carry. The
    // bounds below are about three times those spreads.
    expect(Math.abs(sorted.meanY - plain.meanY)).toBeLessThan(1e-3);
    expect(Math.abs(sorted.maxDensity - plain.maxDensity) / plain.maxDensity).toBeLessThan(0.03);
    expect(sorted.ke).toBeLessThan(0.07);
    expect(plain.ke).toBeLessThan(0.07);
  }, 120_000);
});
