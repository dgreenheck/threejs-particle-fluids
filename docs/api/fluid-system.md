[Docs](../README.md) › [API](../README.md#api-reference) › FluidSystem

# FluidSystem

These are the low-level liquid classes. `FluidSystem` is the Position Based Fluids solver (Macklin & Müller 2013), `ViscositySolver` adds implicit viscosity for thick liquids, and `FluidSurfaceRenderer` draws the liquid as a ray-marched surface. [`Simulation.addFluid`](./fluid.md) builds these for you, and adds a `ViscositySolver` only when you give `thickness`. This page is for building them yourself.

```ts
import { FluidSystem, ViscositySolver, FluidSurfaceRenderer } from 'threejs-particle-fluids';
```

**Contents:** [FluidSystem](#fluidsystem-1) · [ViscositySolver](#viscositysolver) · [FluidSurfaceRenderer](#fluidsurfacerenderer) · [FluidAppearance](#fluidappearance) · [Emitting](#emitting) · [Sorting](#sorting) · [Limitations](#limitations)

## FluidSystem

A [`Material`](./extending.md#material) that keeps the particles in `range` at rest density. Add it to a [`SimLoop`](./core.md#simloop)'s `materials`. The optional effects `viscosity`, `vorticity`, `surfaceTension`, and `adhesion` are compiled into the solver only if you give them at construction.

```ts
particles.uploadParticles(points); // spaced particleSpacing apart
const water = new FluidSystem(particles, { range: waterRange, viscosity: 0.01, surfaceTension: 0 });
water.addBoundary(duckRange);
const loop = new SimLoop(particles, { materials: [water], colliders: [walls] });
water.surfaceTension = 0.1; // allowed: surfaceTension was given, even as 0
```

### Constructor

```ts
new FluidSystem(particles: ParticleSystem, options?: FluidSystemOptions)
```

| Parameter   | Type                                         | Description              |
| ----------- | -------------------------------------------- | ------------------------ |
| `particles` | [`ParticleSystem`](./core.md#particlesystem) | Particle storage.        |
| `options`   | [`FluidSystemOptions`](#fluidsystemoptions)  | See below. Default `{}`. |

The constructor sets the inverse mass of every particle in `range` to `1 / mass` and fills `density` over `range` with `restDensity`. `ParticleSystem.uploadParticles` resets inverse masses to `1` by default, so upload first, or call `particles.setInvMass(fluid.range, 1 / fluid.mass)` afterwards.

#### FluidSystemOptions

| Option            | Type                                       | Default                                   | Description                                                                                                                                                                                                             |
| ----------------- | ------------------------------------------ | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `range`           | [`ParticleRange`](./core.md#particlerange) | `{ start: 0, count: particles.capacity }` | Particles that make up the fluid.                                                                                                                                                                                       |
| `restDensity`     | `number`                                   | `1000`                                    | Rest density, kg/m³.                                                                                                                                                                                                    |
| `particleSpacing` | `number`                                   | `2 × particles.particleRadius`            | Distance between particles at rest, m. Sets `mass`.                                                                                                                                                                     |
| `smoothingRadius` | `number`                                   | `2 × particleSpacing`                     | Distance within which particles affect each other, m.                                                                                                                                                                   |
| `compliance`      | `number`                                   | `1e-4`                                    | How much the fluid can compress. `0` is incompressible. Fixed after construction.                                                                                                                                       |
| `viscosity`       | `number`                                   | off                                       | How strongly each particle's velocity is blended with its neighbors' (XSPH viscosity).                                                                                                                                  |
| `vorticity`       | `number`                                   | off                                       | Strength of vorticity confinement, which restores swirls the solver damps out.                                                                                                                                          |
| `surfaceTension`  | `number`                                   | off                                       | Pulls the fluid into drops and smooth sheets. Acts between fluid particles only.                                                                                                                                        |
| `adhesion`        | `number`                                   | off                                       | Attraction toward boundary particles (see [`addBoundary`](#addboundaryrange-options)). No effect without boundaries.                                                                                                    |
| `sortByCell`      | `boolean`                                  | `false`                                   | Re-sort the fluid's particles into neighbor-grid order at the start of every step, which keeps the solver fast once the fluid has mixed. Particle indices within `range` change between steps. See [Sorting](#sorting). |

#### Errors

| Throws                                                       | When                                                                                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `FluidSystem: invalid particle range`                        | `range` has a negative or non-integer value, is empty, or runs past `particles.capacity`.                                 |
| `FluidSystem: restDensity must be positive`                  | `restDensity` is zero, negative, or not finite. `particleSpacing` and `smoothingRadius` throw the same way.               |
| `FluidSystem: compliance must be ≥ 0`                        | `compliance` is negative or `NaN`.                                                                                        |
| `FluidSystem: <option> must be finite`                       | `viscosity`, `vorticity`, `surfaceTension`, or `adhesion` is `NaN` or infinite.                                           |
| `FluidSystem: sortByCell supports at most 1048576 particles` | `sortByCell` is on and the particle system holds more than 1,048,576 particles. Thrown when a `SimLoop` builds the fluid. |

### Properties

| Property          | Type                                         | Access     | Description                                                                                                             |
| ----------------- | -------------------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------- |
| `particles`       | [`ParticleSystem`](./core.md#particlesystem) | read-only  | Particle storage.                                                                                                       |
| `range`           | [`ParticleRange`](./core.md#particlerange)   | read-only  | Particles that make up the fluid.                                                                                       |
| `restDensity`     | `number`                                     | read-only  | Rest density, kg/m³.                                                                                                    |
| `particleSpacing` | `number`                                     | read-only  | Rest spacing, m.                                                                                                        |
| `smoothingRadius` | `number`                                     | read-only  | Interaction distance, m.                                                                                                |
| `neighborRadius`  | `number`                                     | read-only  | Equals `smoothingRadius`. `SimLoop` sizes its hash grid from the largest `neighborRadius`.                              |
| `mass`            | `number`                                     | read-only  | Mass of one particle, kg. Equals `restDensity × particleSpacing³`.                                                      |
| `density`         | `StorageBufferNode<'float'>`                 | read-only  | Density of each particle, kg/m³, with one entry per particle in the system. Updated every solver iteration for `range`. |
| `viscosity`       | `number`                                     | read/write | XSPH viscosity. Requires `viscosity` at construction.                                                                   |
| `vorticity`       | `number`                                     | read/write | Vorticity confinement strength. Requires `vorticity` at construction.                                                   |
| `surfaceTension`  | `number`                                     | read/write | Surface tension strength. Requires `surfaceTension` at construction.                                                    |
| `adhesion`        | `number`                                     | read/write | Adhesion strength. Requires `adhesion` at construction.                                                                 |

| Throws                                                                        | When                                                                                                  |
| ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| ``FluidSystem: pass `<name>` in the options to enable it before changing it`` | You read or set `viscosity`, `vorticity`, `surfaceTension`, or `adhesion` without giving that option. |
| `FluidSystem: <name> must be finite`                                          | You set one of those properties to `NaN` or an infinite value.                                        |

### Methods

#### `addBoundary(range, options?)`

```ts
addBoundary(range: ParticleRange, options?: { readonly dynamic?: boolean }): void
```

Makes the particles in `range` a solid boundary (Akinci et al. 2012). The fluid can't pass through them, and it pushes on them, which gives them buoyancy. It also clings to them when `adhesion` is set. Use it for the surface particles of a soft body or cloth, and call it before you create the `SimLoop`.

| Parameter         | Type                                       | Description                                                                                                                                                          |
| ----------------- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `range`           | [`ParticleRange`](./core.md#particlerange) | Boundary particles. Must not overlap the fluid's `range`.                                                                                                            |
| `options.dynamic` | `boolean`                                  | Default `true`, which recomputes boundary volumes every substep from predicted positions. `false` computes them once, before the first step, from current positions. |

- A boundary particle's volume is computed from its neighbors in the same `addBoundary` range only.
- The fluid pushes only on particles in its own boundary ranges, and only on those with `invMass > 0`.
- Boundary volumes are stored in the shared `ParticleSystem.boundaryVolume`, which matters when you have several fluids. See [Limitations](#limitations).

| Throws                                                                | When                                                                              |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `FluidSystem.addBoundary: add boundaries before creating the SimLoop` | You called it after a `SimLoop` was created with this fluid.                      |
| `FluidSystem.addBoundary: invalid particle range`                     | `range` has a negative or non-integer value, is empty, or runs past the capacity. |
| `FluidSystem.addBoundary: a boundary cannot overlap the fluid`        | `range` overlaps the fluid's own `range`.                                         |
| `FluidSystem.addBoundary: boundaries cannot overlap each other`       | `range` overlaps a boundary you added earlier.                                    |

#### `readbackOverflow()`

```ts
readbackOverflow(): Promise<boolean>
```

Resolves to `true` if, at the last neighbor rebuild, some fluid particle had more than 64 neighbors within `smoothingRadius` and lost the extras. It reads back from the GPU, so use it for debugging and tests. [`SimLoop.readbackOverflow()`](./core.md#simloop) doesn't include this check.

| Throws                                          | When                                              |
| ----------------------------------------------- | ------------------------------------------------- |
| `FluidSystem: add the fluid to a SimLoop first` | You called it before a `SimLoop` built the fluid. |

#### `build(context)`

```ts
build(context: SolverContext): MaterialKernels
```

`SimLoop` calls this once and runs the kernels it returns. See [`Material`](./extending.md#material).

## ViscositySolver

Adds implicit viscosity for thick liquids like honey. It stays stable at viscosities where `FluidSystem`'s `viscosity` option would blow up. It's a [`Material`](./extending.md#material), so list it after its fluid in `materials`.

```ts
const honey = new FluidSystem(particles, { viscosity: 0.03 });
const thick = new ViscositySolver(honey, { viscosity: 20 });
const loop = new SimLoop(particles, { materials: [honey, thick] }); // order matters
```

### Constructor

```ts
new ViscositySolver(fluid: FluidSystem, options: ViscositySolverOptions)
```

| Parameter | Type                                                | Description                        |
| --------- | --------------------------------------------------- | ---------------------------------- |
| `fluid`   | [`FluidSystem`](#fluidsystem-1)                     | Fluid whose velocities are solved. |
| `options` | [`ViscositySolverOptions`](#viscositysolveroptions) | Required.                          |

#### ViscositySolverOptions

| Option       | Type     | Default  | Description                                                                                                                                                                                                      |
| ------------ | -------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `viscosity`  | `number` | required | Kinematic viscosity, m²/s. Its effect scales with `1 / smoothingRadius²`, so values don't match real liquids. Tune it by eye; honey is about 20. Unrelated to `FluidSystem`'s unitless `viscosity`. Must be ≥ 0. |
| `iterations` | `number` | `12`     | Jacobi sweeps per substep. Integer, 1–64. Fixed after construction.                                                                                                                                              |

#### Errors

| Throws                                                             | When                                                                                                    |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `ViscositySolver: viscosity must be ≥ 0`                           | `viscosity` is negative or not finite, in the options or when you set it later.                         |
| `ViscositySolver: iterations must be an integer from 1 to 64`      | `iterations` isn't an integer from 1 to 64.                                                             |
| ``ViscositySolver: list it after its FluidSystem in `materials` `` | Thrown by the `SimLoop` constructor when you listed the solver before its fluid, or left the fluid out. |

### Properties

| Property    | Type                            | Access     | Description                                        |
| ----------- | ------------------------------- | ---------- | -------------------------------------------------- |
| `fluid`     | [`FluidSystem`](#fluidsystem-1) | read-only  | Fluid whose velocities are solved.                 |
| `viscosity` | `number`                        | read/write | Kinematic viscosity, m²/s. Must be ≥ 0 and finite. |

### Methods

#### `build()`

```ts
build(): MaterialKernels
```

`SimLoop` calls this once. It returns `iterations + 2` dispatches over the fluid's range, which run after the solve.

## FluidSurfaceRenderer

Draws a `FluidSystem` as a smooth liquid surface. Each `update()` splats the particles into a voxel field, and the mesh's shader ray-marches that field when drawn. The liquid refracts the opaque scene behind it with Beer–Lambert absorption. It adds Fresnel-weighted environment and screen-space reflections, plus a GGX highlight from the scene's [key light](#key-light). It writes depth.

```ts
const surface = new FluidSurfaceRenderer(water, {
  renderer,
  scene,
  camera,
  bounds,
  colliders: [walls],
});
scene.add(surface.mesh); // not added automatically

// each frame
await loop.step(dt);
await surface.update();
renderer.render(scene, camera);
```

### Constructor

```ts
new FluidSurfaceRenderer(fluid: FluidSystem, options: FluidSurfaceRendererOptions)
```

| Parameter | Type                                                          | Description    |
| --------- | ------------------------------------------------------------- | -------------- |
| `fluid`   | [`FluidSystem`](#fluidsystem-1)                               | Fluid to draw. |
| `options` | [`FluidSurfaceRendererOptions`](#fluidsurfacerendereroptions) | Required.      |

#### FluidSurfaceRendererOptions

| Option          | Type                                               | Default                                                      | Description                                                                                                                                                                                               |
| --------------- | -------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `renderer`      | `WebGPURenderer`                                   | required                                                     | Runs the field kernels and `pick()`.                                                                                                                                                                      |
| `scene`         | `Scene`                                            | required                                                     | `scene.environment`, `scene.environmentIntensity`, and the key light are read every `update()`.                                                                                                           |
| `camera`        | `Camera`                                           | required                                                     | Used only by `pick()`, which casts rays from it. Drawing uses whichever camera renders the mesh.                                                                                                          |
| `bounds`        | `Box3`                                             | required                                                     | World box the liquid can occupy, m. Particles outside are ignored.                                                                                                                                        |
| `colliders`     | `readonly (PrimitiveSet \| SDFCollider)[]`         | `[]`                                                         | Colliders the surface is cut against. The liquid curves up where it meets them. See [`PrimitiveSet`](./colliders.md#primitiveset), [`SDFCollider`](./colliders.md#sdfcollider).                           |
| `carve`         | [`PrimitiveSet`](./colliders.md#primitiveset)      | none                                                         | Shapes cut out of the surface every frame. The liquid doesn't curve up against them.                                                                                                                      |
| `solids`        | [`ParticleRange`](./core.md#particlerange)         | none                                                         | Non-fluid particles, such as floating bodies, that the liquid curves up against.                                                                                                                          |
| `motionStretch` | `number`                                           | `0`                                                          | Stretches each fluid particle back along its velocity by this many seconds of travel, up to `3 × particleSpacing`. `0` turns it off. Must be ≥ 0.                                                         |
| `voxelBudget`   | `number`                                           | `FluidSurfaceRenderer.defaultVoxelBudget(fluid.range.count)` | Upper bound on voxels in the field, about 76 bytes each. Must be ≥ 1.                                                                                                                                     |
| `appearance`    | `Partial<`[`FluidAppearance`](#fluidappearance)`>` | all defaults                                                 | Initial look.                                                                                                                                                                                             |
| `cavities`      | `{ smokeColor: number; smokeDensity: number }`     | none                                                         | Draws pockets cut by `carve` with a reflective rim, filled with smoke. `smokeColor` is sRGB hex. `smokeDensity` is how quickly the smoke blocks light, 1/m, ≥ 0. Adds 16 steps to the transmission march. |
| `refraction`    | `boolean`                                          | `true`                                                       | Bend transmitted light. `false` samples the scene behind without bending it.                                                                                                                              |

Each voxel is as large as it needs to be for `bounds` to fit within `voxelBudget`, but never smaller than `particles.particleRadius`.

#### Errors

| Throws                                                    | When                                               |
| --------------------------------------------------------- | -------------------------------------------------- |
| `FluidSurfaceRenderer: bounds must have positive extent`  | `bounds` has zero or negative size on some axis.   |
| `FluidSurfaceRenderer: motionStretch must be ≥ 0`         | `motionStretch` is negative or not finite.         |
| `FluidSurfaceRenderer: voxelBudget must be ≥ 1`           | `voxelBudget` is less than 1 or not finite.        |
| `FluidSurfaceRenderer: smokeDensity must be ≥ 0`          | `cavities.smokeDensity` is negative or not finite. |
| `FluidSurfaceRenderer: appearance.<field> must be finite` | A field of `appearance` is `NaN` or infinite.      |

### Properties

| Property       | Type                            | Access     | Description                                                                                                                                        |
| -------------- | ------------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mesh`         | `Mesh`                          | read-only  | Box around the field, drawn from its back faces. Named `'FluidSurface'`, with `renderOrder = -1` and `frustumCulled = false`. Add it to the scene. |
| `fluid`        | [`FluidSystem`](#fluidsystem-1) | read-only  | Fluid being drawn.                                                                                                                                 |
| `reflections`  | `boolean`                       | read/write | Screen-space reflections. Default `true`. When `false`, the surface reflects only the environment.                                                 |
| `smokeDensity` | `number`                        | read/write | How quickly cavity smoke blocks light, 1/m, ≥ 0. Starts at `cavities.smokeDensity`. Requires `cavities` at construction.                           |

| Throws                                                                                             | When                                                            |
| -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| ``FluidSurfaceRenderer: pass `cavities` in the options to enable smokeDensity before changing it`` | You read or set `smokeDensity` without passing `cavities`.      |
| `FluidSurfaceRenderer: smokeDensity must be ≥ 0`                                                   | You set `smokeDensity` to a negative, `NaN`, or infinite value. |

### Methods

#### `update()`

```ts
update(): Promise<void>
```

Rebuilds the surface field from the current particle positions. Call it once per frame, after stepping and before rendering. Each call also picks up changes to the scene:

- It re-reads the key light and `scene.environmentIntensity`.
- It follows `scene.environment`. A new map replaces the old one in place, and gaining or losing a map rebuilds the surface shader once.
- It computes where the liquid meets `colliders` on the first call, and again whenever one of them changes its `version` by moving or changing shape.

Throws `FluidSurfaceRenderer: already disposed` if you call it after `dispose()`.

#### `setAppearance(appearance)`

```ts
setAppearance(appearance: Partial<FluidAppearance>): void
```

Changes any [`FluidAppearance`](#fluidappearance) fields. Fields you leave out or set to `undefined` keep their value. A `NaN` or infinite field throws `FluidSurfaceRenderer: appearance.<field> must be finite`, and changes nothing.

#### `pick(uv)`

```ts
pick(uv: Vector2): Promise<Vector3 | null>
```

Finds the liquid under a point on screen. It casts a ray from `options.camera` through viewport coordinate `uv`, which runs from 0 to 1 with y pointing down. It resolves to the world-space hit point, or `null` on a miss. It traces the field from the last `update()` and reads back from the GPU. Throws `FluidSurfaceRenderer: already disposed` after `dispose()`.

#### `dispose()`

```ts
dispose(): void
```

Removes `mesh` from its parent and frees its geometry and material, the field texture, the field's GPU buffers, and the compiled kernels. Calling it twice is safe. After it, `update()` and `pick()` throw.

#### `FluidSurfaceRenderer.defaultVoxelBudget(particleCount)`

```ts
static defaultVoxelBudget(particleCount: number): number
```

The default `voxelBudget` is `1_200_000` for 50 000 particles or more, `900_000` for 20 000 or more, and `600_000` otherwise.

### Key light

The key light gives the liquid its specular highlight, and there's none without one. It's the last `DirectionalLight` in `scene`, in traversal order, that has `castShadow` set. If no directional light casts shadows, it's the first one found. The renderer looks for it at construction and on each `update()` until it finds one, then keeps it. Its direction and `color × intensity` are read every `update()`.

## FluidAppearance

Controls how the liquid looks. Pass it as `options.appearance` or to `setAppearance()`. Every field can change at any time and must be finite.

| Field                 | Type     | Default    | Description                                                                                                                    |
| --------------------- | -------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `color`               | `number` | `0x2a8fb0` | sRGB hex. The colour white light turns after passing through `attenuationDistance` of liquid. Channels are clamped to ≥ 0.001. |
| `attenuationDistance` | `number` | `0.6`      | Path length at which transmitted light reaches `color`, m. Clamped to ≥ 1e-4.                                                  |
| `scattering`          | `number` | `0.08`     | Light scattered back out of the body. `0` is clear, `1` is milky.                                                              |
| `ior`                 | `number` | `1.333`    | Index of refraction. Also sets Fresnel reflectance.                                                                            |
| `roughness`           | `number` | `0.04`     | Blurs reflections and widens the highlight. Clamped to 0.02–1.                                                                 |
| `envIntensity`        | `number` | `1`        | Multiplier on environment reflection.                                                                                          |
| `metalness`           | `number` | `0`        | Blend from dielectric (`0`) to metal (`1`).                                                                                    |
| `metalColor`          | `number` | `0xc8d2da` | sRGB hex. Reflectance tint at `metalness: 1`.                                                                                  |

## Emitting

`FluidSystem` has no emit method, because `range` fixes its particle count. To pour liquid, park the fluid's particles out of view and pin them in place. Then release a batch each frame from a compute kernel:

1. Call `particles.setInvMass(fluid.range, 0)` to pin them. Do this after constructing the fluid, because the constructor sets masses.
2. For each particle `i` you release, write the nozzle position to `particles.positions[i]` and `particles.predictedPositions[i]`, and set `particles.velocities[i]`. Then set `particles.invMass[i]` to `1 / fluid.mass` and `fluid.density[i]` to `fluid.restDensity`.
3. Release the next batch only once the stream has moved one `particleSpacing`, so new particles don't overlap.

[`demo/presets/honey.ts`](../../demo/presets/honey.ts) does this. Leave [`sortByCell`](#sorting) off for a fluid you emit this way, because sorting moves parked and released particles between slots.

## Sorting

The solver runs its fluid kernels in the neighbor grid's order, so each workgroup handles particles that are close together. The particles' data, though, stays where it was uploaded, and every kernel reads each neighbor's position, mass, and density by particle index. Once the fluid has mixed, those reads land all over the buffers. Sorting keeps them close together. On an Apple M1 Pro at the demo's Ultra level, the liquid presets simulate 13–28% faster with `sortByCell`, and a shuffled 100,000-particle column takes 12.2 ms per step instead of 15.1 ms.

With `sortByCell: true`, the fluid puts its particles back into the neighbor grid's cell order at the start of every step. The order comes from the grid's last rebuild, so the first step runs unsorted. Everything stored per particle moves with it: every buffer in the `ParticleSystem`, the fluid's `density`, and a `GasSystem`'s air temperature. Nothing outside `range` changes. [`Simulation`](./simulation.md) turns it on for every fluid.

Because a particle's index within `range` changes from step to step:

- Leave it off if you read or write particular fluid particles by index across steps, such as an emitter that releases particles by slot (see [Emitting](#emitting)), or code that follows one particle through `readback()`.
- Per-particle buffers you keep yourself for fluid particles aren't moved.

Each step, the sort runs a prefix sum over every particle in the system, then two passes over the fluid for each per-particle buffer: ten for a liquid, eleven for air with a `GasSystem`.

## Limitations

- You can't change `compliance`, `restDensity`, `particleSpacing`, `smoothingRadius`, or `range` after construction. You can change `viscosity`, `vorticity`, `surfaceTension`, and `adhesion` only if you gave them in the options.
- A static boundary (`dynamic: false`) isn't updated if it moves, because its volumes are computed once before the first step. Each dynamic boundary costs one dispatch over its particles per substep.
- Any boundary adds a buffer of three 32-bit integers per particle in the system, which collects the fluid's push on boundaries. It's cleared every substep and applied every solver iteration, over the span from the first boundary particle to the last.
- Boundary volumes are stored in the shared `ParticleSystem`, so when several fluids share one particle system, give them all the same boundaries, as `Simulation` does. If you don't:
  - every `FluidSystem` treats a boundary added to any of them as a boundary, but pushes only on its own;
  - a range added to fluids with different `smoothingRadius` gets the volume from whichever fluid ran last.
- The fluid can't push back on nearby particles that aren't its boundaries, such as another fluid's particles or a soft body's interior. They still count toward its density with their own mass and push the fluid away, which keeps two fluids on one particle system apart. Surface tension ignores them.
- Each fluid particle can have at most 64 neighbors. Extras are dropped, which [`readbackOverflow()`](#readbackoverflow) reports.
- You can't change `FluidSurfaceRenderer` options after construction, except through `appearance`, `reflections`, and `smokeDensity`.
- `FluidSurfaceRenderer.update()` runs four dispatches over the full voxel grid, so its cost grows with voxel count. It also grows with fluid particle count, and particles are splatted three times when `motionStretch` is above 0.
- Every `carve` primitive is evaluated at every voxel on every `update()`.
