import { Fn, If, Return, instanceIndex, instancedArray, uint } from 'three/tsl';
import type ComputeNode from 'three/src/nodes/gpgpu/ComputeNode.js';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';
import type UniformNode from 'three/src/nodes/core/UniformNode.js';

import type { HashGrid, ParticleRange, ParticleSystem } from '../../core/index.js';
import {
  MAX_CELLS_SINGLE_LEVEL_SCAN,
  SCAN_WORKGROUP_SIZE,
  buildBlockScanKernels,
  padToScanWorkgroup,
} from '../../core/hashGrid/sort.js';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

/** A per-particle buffer that has to move with its particle when the fluid is re-sorted. */
export interface ParticleData {
  readonly buffer:
    | StorageBufferNode<'vec4'>
    | StorageBufferNode<'float'>
    | StorageBufferNode<'uint'>;
  readonly type: 'vec4' | 'float' | 'uint';
  /** Indexed from 0 at `range.start` rather than by global particle index. */
  readonly local?: boolean;
}

/**
 * Kernels that permute a fluid's particles into the neighbor grid's cell
 * order, so particles that are neighbors in space are neighbors in memory.
 *
 * The grid's last rebuild already sorted every particle by cell
 * (`sortedIndices`). Restricted to the fluid's range, that order is found
 * with a prefix sum over "is slot k one of this fluid's particles": a fluid
 * particle at slot k moves to `range.start + prefix(k)`. Every buffer in
 * `data` is then scattered through a scratch copy into its new place, so the
 * permutation is exact: each particle's state moves whole, bit for bit, and
 * nothing outside the range is touched.
 *
 * Run between steps, while no substep holds particle indices (neighbor
 * lists, contacts, and accumulators are all rebuilt each substep). The
 * permute kernels skip while `enabled` is 0, which the caller keeps until the
 * grid has been built once.
 */
export function buildCellSortKernels(args: {
  readonly particles: ParticleSystem;
  readonly grid: HashGrid;
  readonly range: ParticleRange;
  readonly data: readonly ParticleData[];
  readonly enabled: UniformNode<'uint', number>;
}): { readonly kernels: ComputeNode[]; readonly buffers: StorageBufferNode<Any>[] } {
  const { particles, grid, range, data, enabled } = args;
  const { capacity } = particles;
  const nPadded = padToScanWorkgroup(capacity);
  if (nPadded > MAX_CELLS_SINGLE_LEVEL_SCAN) {
    throw new Error(
      `FluidSystem: sortByCell supports at most ${MAX_CELLS_SINGLE_LEVEL_SCAN} particles ` +
        `in the ParticleSystem, got ${capacity}`,
    );
  }
  const W = SCAN_WORKGROUP_SIZE;
  const start = uint(range.start);
  const end = uint(range.start + range.count);
  const sorted = grid.sortedIndices;
  const inRange = (p: Any): Any => p.greaterThanEqual(start).and(p.lessThan(end));

  const prefix = instancedArray(nPadded, 'uint');
  const blockSums = instancedArray(W, 'uint');
  const destination = instancedArray(range.count, 'uint');
  const scratch = {
    vec4: instancedArray(range.count, 'vec4'),
    float: instancedArray(range.count, 'float'),
    uint: instancedArray(range.count, 'uint'),
  };

  const { blockScan, blockSumScan } = buildBlockScanKernels(
    (gi) => {
      const slot: Any = gi.min(uint(capacity - 1));
      return gi
        .lessThan(uint(capacity))
        .and(inRange(sorted.element(slot)))
        .select(uint(1), uint(0));
    },
    (gi, value) => prefix.element(gi).assign(value),
    blockSums,
    nPadded,
    'cellSort',
  );

  const locate = Fn(() => {
    If(enabled.equal(uint(0)), () => Return());
    const k: Any = instanceIndex;
    const p: Any = sorted.element(k).toVar();
    If(inRange(p), () => {
      const rank: Any = prefix.element(k).add(blockSums.element(k.div(uint(W))));
      destination.element(p.sub(start)).assign(rank);
    });
  })()
    .compute(capacity)
    .setName('cellSort.locate');

  const kernels: ComputeNode[] = [blockScan, blockSumScan, locate];
  for (const { buffer, type, local } of data) {
    const tmp: Any = scratch[type];
    const source: Any = buffer;
    const index = (i: Any): Any => (local ? i : i.add(start));
    const scatter = Fn(() => {
      If(enabled.equal(uint(0)), () => Return());
      const i: Any = instanceIndex;
      tmp.element(destination.element(i)).assign(source.element(index(i)));
    })()
      .compute(range.count)
      .setName('cellSort.scatter');
    const copyBack = Fn(() => {
      If(enabled.equal(uint(0)), () => Return());
      const i: Any = instanceIndex;
      source.element(index(i)).assign(tmp.element(i));
    })()
      .compute(range.count)
      .setName('cellSort.copyBack');
    kernels.push(scatter, copyBack);
  }

  return {
    kernels,
    buffers: [prefix, blockSums, destination, scratch.vec4, scratch.float, scratch.uint],
  };
}
