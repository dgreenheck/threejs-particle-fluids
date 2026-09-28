import { instancedArray } from 'three/tsl';
import type { WebGPURenderer } from 'three/webgpu';
import type StorageBufferNode from 'three/src/nodes/accessors/StorageBufferNode.js';

/** A contiguous block of particle slots, `[start, start + count)`. */
export interface ParticleRange {
  readonly start: number;
  readonly count: number;
}

/** Initial state for one particle. See {@link ParticleSystem.uploadParticles}. */
export interface ParticleInit {
  readonly position: readonly [number, number, number];
  /** Default `[0, 0, 0]`. */
  readonly velocity?: readonly [number, number, number];
  /** Inverse mass. `0` pins the particle in place. Default `1`. */
  readonly invMass?: number;
  /**
   * Particles that share a non-zero collision group never collide with each
   * other. Default `0`, which collides with everything. Materials such as
   * {@link SoftbodySystem} assign groups for their own particles.
   */
  readonly collisionGroup?: number;
}

/** CPU copy of the particle state, returned by {@link ParticleSystem.readback}. */
export interface ParticleSnapshot {
  readonly capacity: number;
  /** xyzw per particle. */
  readonly positions: Float32Array;
  readonly predictedPositions: Float32Array;
  readonly velocities: Float32Array;
  readonly invMass: Float32Array;
  /** Unit quaternion (x, y, z, w) per particle. */
  readonly rotation: Float32Array;
  readonly predictedRotation: Float32Array;
  readonly angularVelocity: Float32Array;
}

/**
 * GPU storage for every particle in a simulation. All particles share one
 * radius, which is what lets a single uniform grid find their neighbors.
 *
 * Materials ({@link FluidSystem}, {@link SoftbodySystem}, {@link ClothSystem},
 * …) each own a {@link ParticleRange} of this buffer, and {@link SimLoop}
 * advances all of them together, so different materials interact naturally.
 *
 * The storage buffers are public so custom TSL kernels can read and write
 * them. After writing to a buffer's `.value.array` on the CPU, set
 * `.value.needsUpdate = true`.
 */
export class ParticleSystem {
  readonly renderer: WebGPURenderer;
  readonly capacity: number;
  /** Radius shared by every particle, in metres. */
  readonly particleRadius: number;

  readonly positions: StorageBufferNode<'vec4'>;
  /** Positions being solved during a substep (`x*` in the PBD literature). */
  readonly predictedPositions: StorageBufferNode<'vec4'>;
  readonly velocities: StorageBufferNode<'vec4'>;
  readonly invMass: StorageBufferNode<'float'>;
  /** See {@link ParticleInit.collisionGroup}. */
  readonly collisionGroup: StorageBufferNode<'uint'>;
  /**
   * Boundary volume per particle. Non-zero for particles that a
   * {@link FluidSystem} treats as a solid boundary.
   */
  readonly boundaryVolume: StorageBufferNode<'float'>;
  /**
   * Orientation per particle as a unit quaternion. Only soft bodies that use
   * local shape matching rotate their particles; everything else stays at
   * identity.
   */
  readonly rotation: StorageBufferNode<'vec4'>;
  readonly predictedRotation: StorageBufferNode<'vec4'>;
  readonly angularVelocity: StorageBufferNode<'vec4'>;

  private isDisposed = false;

  constructor(renderer: WebGPURenderer, capacity: number, particleRadius: number) {
    if (!Number.isInteger(capacity) || capacity <= 0) {
      throw new Error(`ParticleSystem: capacity must be a positive integer, got ${capacity}`);
    }
    if (!Number.isFinite(particleRadius) || particleRadius <= 0) {
      throw new Error(
        `ParticleSystem: particleRadius must be a positive finite number, got ${particleRadius}`,
      );
    }
    this.renderer = renderer;
    this.capacity = capacity;
    this.particleRadius = particleRadius;

    this.positions = instancedArray(capacity, 'vec4');
    this.predictedPositions = instancedArray(capacity, 'vec4');
    this.velocities = instancedArray(capacity, 'vec4');
    this.invMass = instancedArray(capacity, 'float');
    this.collisionGroup = instancedArray(capacity, 'uint');
    this.boundaryVolume = instancedArray(capacity, 'float');
    this.rotation = instancedArray(capacity, 'vec4');
    this.predictedRotation = instancedArray(capacity, 'vec4');
    this.angularVelocity = instancedArray(capacity, 'vec4');

    // Zero-filled quaternions are not rotations; start every particle at identity.
    const rotation = this.rotation.value.array as Float32Array;
    const predictedRotation = this.predictedRotation.value.array as Float32Array;
    for (let i = 0; i < capacity; i++) {
      rotation[i * 4 + 3] = 1;
      predictedRotation[i * 4 + 3] = 1;
    }
    this.rotation.value.needsUpdate = true;
    this.predictedRotation.value.needsUpdate = true;
  }

  /**
   * Write initial particle state into slots `[start, start + data.length)`.
   * Slots that are never written keep `invMass = 0`, so they stay put.
   * Throws on a range outside the buffer, a negative or non-finite
   * `invMass`, or a collision group that isn't a `uint32`.
   */
  uploadParticles(data: readonly ParticleInit[], start = 0): void {
    this.assertAlive();
    if (data.length === 0) {
      // An empty upload still has to start at a slot boundary inside the buffer.
      if (!Number.isInteger(start) || start < 0 || start > this.capacity) {
        throw new Error(
          `ParticleSystem.uploadParticles: invalid particle range start=${start} count=0 (capacity ${this.capacity})`,
        );
      }
      return;
    }
    assertRange(this, { start, count: data.length }, 'ParticleSystem.uploadParticles');
    for (const p of data) {
      if (p.invMass !== undefined) assertInvMass(p.invMass, 'ParticleSystem.uploadParticles');
      if (p.collisionGroup !== undefined) {
        assertCollisionGroup(p.collisionGroup, 'ParticleSystem.uploadParticles');
      }
    }
    const positions = this.positions.value.array as Float32Array;
    const predicted = this.predictedPositions.value.array as Float32Array;
    const velocities = this.velocities.value.array as Float32Array;
    const invMass = this.invMass.value.array as Float32Array;
    const groups = this.collisionGroup.value.array as Uint32Array;
    for (let k = 0; k < data.length; k++) {
      const p = data[k]!;
      const i = start + k;
      const [x, y, z] = p.position;
      const [vx, vy, vz] = p.velocity ?? [0, 0, 0];
      positions.set([x, y, z, 0], i * 4);
      predicted.set([x, y, z, 0], i * 4);
      velocities.set([vx, vy, vz, 0], i * 4);
      invMass[i] = p.invMass ?? 1;
      groups[i] = p.collisionGroup ?? 0;
    }
    this.positions.value.needsUpdate = true;
    this.predictedPositions.value.needsUpdate = true;
    this.velocities.value.needsUpdate = true;
    this.invMass.value.needsUpdate = true;
    this.collisionGroup.value.needsUpdate = true;
  }

  /** Set the inverse mass of every particle in `range`. Must be finite and ≥ 0. */
  setInvMass(range: ParticleRange, invMass: number): void {
    this.assertAlive();
    assertRange(this, range, 'ParticleSystem.setInvMass');
    assertInvMass(invMass, 'ParticleSystem.setInvMass');
    (this.invMass.value.array as Float32Array).fill(
      invMass,
      range.start,
      range.start + range.count,
    );
    this.invMass.value.needsUpdate = true;
  }

  /** Set the collision group of every particle in `range`. Must be an integer in `[0, 2³² − 1]`. */
  setCollisionGroup(range: ParticleRange, group: number): void {
    this.assertAlive();
    assertRange(this, range, 'ParticleSystem.setCollisionGroup');
    assertCollisionGroup(group, 'ParticleSystem.setCollisionGroup');
    (this.collisionGroup.value.array as Uint32Array).fill(
      group,
      range.start,
      range.start + range.count,
    );
    this.collisionGroup.value.needsUpdate = true;
  }

  /**
   * Read the particle state back from the GPU. This stalls until the GPU is
   * idle, so use it for debugging and tests rather than every frame.
   */
  async readback(): Promise<ParticleSnapshot> {
    this.assertAlive();
    const backend = this.renderer.backend as unknown as { get(o: object): { buffer?: unknown } };
    const read = async (buffer: StorageBufferNode<'vec4'> | StorageBufferNode<'float'>) =>
      // A buffer no kernel has used yet exists only on the CPU, where it is still current.
      backend.get(buffer.value).buffer === undefined
        ? new Float32Array(buffer.value.array as Float32Array)
        : new Float32Array(await this.renderer.getArrayBufferAsync(buffer.value));
    const [
      positions,
      predictedPositions,
      velocities,
      invMass,
      rotation,
      predictedRotation,
      angularVelocity,
    ] = await Promise.all([
      read(this.positions),
      read(this.predictedPositions),
      read(this.velocities),
      read(this.invMass),
      read(this.rotation),
      read(this.predictedRotation),
      read(this.angularVelocity),
    ]);
    return {
      capacity: this.capacity,
      positions,
      predictedPositions,
      velocities,
      invMass,
      rotation,
      predictedRotation,
      angularVelocity,
    };
  }

  /** True after {@link dispose}. */
  get disposed(): boolean {
    return this.isDisposed;
  }

  /**
   * @internal Every per-particle buffer, indexed by particle, with its element
   * type. Code that moves particles between slots must move all of these.
   */
  get perParticleBuffers(): readonly (
    | { readonly buffer: StorageBufferNode<'vec4'>; readonly type: 'vec4' }
    | { readonly buffer: StorageBufferNode<'float'>; readonly type: 'float' }
    | { readonly buffer: StorageBufferNode<'uint'>; readonly type: 'uint' }
  )[] {
    return [
      { buffer: this.positions, type: 'vec4' },
      { buffer: this.predictedPositions, type: 'vec4' },
      { buffer: this.velocities, type: 'vec4' },
      { buffer: this.invMass, type: 'float' },
      { buffer: this.collisionGroup, type: 'uint' },
      { buffer: this.boundaryVolume, type: 'float' },
      { buffer: this.rotation, type: 'vec4' },
      { buffer: this.predictedRotation, type: 'vec4' },
      { buffer: this.angularVelocity, type: 'vec4' },
    ];
  }

  /**
   * Free the particle buffers on the GPU. Later uploads and readbacks throw.
   * Dispose every loop, material, collider, and mesh that reads these
   * buffers first; they can't run afterwards.
   */
  dispose(): void {
    if (this.isDisposed) return;
    this.isDisposed = true;
    releaseStorageBuffers(this.renderer, [
      this.positions,
      this.predictedPositions,
      this.velocities,
      this.invMass,
      this.collisionGroup,
      this.boundaryVolume,
      this.rotation,
      this.predictedRotation,
      this.angularVelocity,
    ]);
  }

  private assertAlive(): void {
    if (this.isDisposed) throw new Error('ParticleSystem has been disposed');
  }
}

function assertInvMass(invMass: number, context: string): void {
  if (!(invMass >= 0) || !Number.isFinite(invMass)) {
    throw new Error(`${context}: invMass must be a finite number ≥ 0, got ${invMass}`);
  }
}

function assertCollisionGroup(group: number, context: string): void {
  if (!Number.isInteger(group) || group < 0 || group > 0xffffffff) {
    throw new Error(`${context}: collisionGroup must be an integer in [0, 2³² − 1], got ${group}`);
  }
}

/**
 * Destroy the GPU buffers behind storage nodes. A buffer no kernel has used
 * yet has nothing on the GPU to free. Kernels that still reference a freed
 * buffer must not be dispatched again.
 */
export function releaseStorageBuffers(
  renderer: WebGPURenderer,
  buffers: readonly { readonly value: object }[],
): void {
  // three r184 has no public call that frees a storage buffer; attribute
  // disposal only reaches the renderer for geometry attributes. Its
  // attribute manager's `delete` destroys the GPUBuffer and updates
  // `renderer.info`.
  const internals = renderer as unknown as
    | {
        readonly _attributes?: { delete(attribute: object): unknown } | null;
        readonly backend?: { get(object: object): { buffer?: unknown } };
      }
    | undefined;
  // Without a backend (a renderer that never initialized) nothing was uploaded.
  if (!internals?.backend) return;
  for (const { value } of buffers) {
    if (internals.backend.get(value).buffer !== undefined) internals._attributes?.delete(value);
  }
}

/** Throw unless `range` is a non-empty block of whole slots inside `particles`. */
export function assertRange(
  particles: { readonly capacity: number },
  range: ParticleRange,
  context: string,
): void {
  const { start, count } = range;
  if (
    !Number.isInteger(start) ||
    !Number.isInteger(count) ||
    start < 0 ||
    count <= 0 ||
    start + count > particles.capacity
  ) {
    throw new Error(
      `${context}: invalid particle range start=${start} count=${count} (capacity ${particles.capacity})`,
    );
  }
}
