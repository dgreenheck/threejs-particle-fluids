[Docs](../README.md) › [API](../README.md#api-reference) › Simulation

# Simulation

`Simulation` is the main way to use the library. You add liquid, soft bodies, cloth, smoke, and obstacles to it, and it builds and connects everything the first time you call `step()`.

```ts
import { Simulation, createParticleRenderer } from 'threejs-particle-fluids';
```

- [`createParticleRenderer`](#createparticlerenderer)
- [`Simulation`](#simulation-1)
  - [Constructor](#constructor) · [`SimulationOptions`](#simulationoptions)
  - [Properties](#properties)
  - [Methods](#methods): [`addFluid`](#addfluidoptions) · [`addSoftbody`](#addsoftbodyoptions) · [`addCloth`](#addclothoptions) · [`addSmoke`](#addsmokeoptions) · [`addFloor`](#addflooroptions) · [`addSphere`](#addsphereoptions) · [`addBox`](#addboxoptions) · [`addCapsule`](#addcapsuleoptions) · [`addMesh`](#addmeshoptions) · [`start`](#start) · [`step`](#stepdt) · [`dispose`](#dispose)
- [Errors](#errors)
- [Limitations](#limitations)

---

## createParticleRenderer

```ts
createParticleRenderer(options?: Partial<WebGPURendererParameters>): Promise<WebGPURenderer>
```

Creates a `WebGPURenderer` and waits for it to be ready. Use it instead of `new WebGPURenderer()`, because it asks the GPU for the larger limits the simulation needs.

| Parameter | Type                                | Description                                                                                 |
| --------- | ----------------------------------- | ------------------------------------------------------------------------------------------- |
| `options` | `Partial<WebGPURendererParameters>` | Passed to `WebGPURenderer`. Any `requiredLimits` you give are combined with the ones below. |

The limits it requests:

| Limit                               | Value |
| ----------------------------------- | ----- |
| `maxComputeInvocationsPerWorkgroup` | 1024  |
| `maxComputeWorkgroupSizeX`          | 1024  |
| `maxStorageBuffersPerShaderStage`   | 10    |

| Throws                                             | When                                                                   |
| -------------------------------------------------- | ---------------------------------------------------------------------- |
| `createParticleRenderer: WebGPU is unavailable, …` | The browser doesn't support WebGPU. The library has no WebGL fallback. |

---

## Simulation

### Constructor

```ts
new Simulation(options: SimulationOptions)
```

### SimulationOptions

| Option           | Type             | Default                                  | Description                                                                     |
| ---------------- | ---------------- | ---------------------------------------- | ------------------------------------------------------------------------------- |
| `renderer`       | `WebGPURenderer` | required                                 | From [`createParticleRenderer`](#createparticlerenderer).                       |
| `scene`          | `Scene`          | required                                 | Scene the surfaces are added to. Liquid reflects `scene.environment`.           |
| `camera`         | `Camera`         | required                                 | The camera you render with.                                                     |
| `particles`      | `number`         | one of these two                         | Total particle budget. See [Sizing particles](#sizing-particles).               |
| `particleRadius` | `number`         | one of these two                         | Radius of every particle, m. See [Sizing particles](#sizing-particles).         |
| `container`      | `Box3`           | none                                     | A box with a floor and four walls that keeps everything in. Required for smoke. |
| `closed`         | `boolean`        | `false`                                  | Puts a lid on `container`. Smoke always gets one.                               |
| `gravity`        | `Vector3`        | `(0, -9.81, 0)`; `(0, -1, 0)` with smoke | m/s². To change it later, use the [`gravity`](#properties) property.            |
| `substeps`       | `number`         | chosen for you                           | How many pieces each 1/60 s step is split into. See [Substeps](#substeps).      |

#### Sizing particles

Every particle in a simulation is the same size. You choose it in one of two ways.

With `particles`, you set a budget and the simulation picks the size. When it starts, it lays everything out and adjusts the size until the total fits within the budget, usually within a few percent of it. Big objects get more particles than small ones.

With `particleRadius`, you set the size in metres and the count follows from what you add. A box of water 0.4 × 0.5 × 0.6 m holds about 4,800 particles at a radius of 0.014 m.

More particles make every frame slower. After `start()`, `particleCount` and `particleRadius` show what was chosen, and each object's own `particleCount` shows how many it got. The console warns you about a soft body with fewer than 100 particles or a cloth with fewer than 10 along a side, since those can't keep their shape.

#### Substeps

Each 1/60 s step is split into smaller substeps. More substeps stop fast or thin obstacles from being passed through and keep stiff objects stiff, but each one costs about as much as a whole step. If you don't set `substeps`, the simulation picks between 2 and 24, using more when the particles are small or the scene has soft bodies or cloth.

### Properties

| Property         | Type                                         | Access     | Description                                                                                                                                                                        |
| ---------------- | -------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gravity`        | `Vector3`                                    | read       | Gravity, m/s². Change it in place, for example `sim.gravity.set(0, -3, 0)`.                                                                                                        |
| `particleRadius` | `number`                                     | read       | Radius of every particle, m. When you use `particles`, it's `0` until the simulation starts.                                                                                       |
| `particleCount`  | `number`                                     | read       | Particles in use. `0` until the simulation starts.                                                                                                                                 |
| `showParticles`  | `boolean`                                    | read/write | Draws the particles instead of the surfaces, for debugging. Default `false`.                                                                                                       |
| `particleSystem` | [`ParticleSystem`](./core.md#particlesystem) | read       | The particle storage underneath. Available after `start()`. Liquid and air particles are re-sorted every step (see [Sorting](./fluid-system.md#sorting)), so their indices change. |
| `loop`           | [`SimLoop`](./core.md#simloop)               | read       | The solver underneath. Available after `start()`. You can change `loop.substeps` while running.                                                                                    |

### Methods

Call the `add` methods before the first `start()` or `step()`. All positions are in world space, in metres.

#### `addFluid(options)`

```ts
addFluid(options: FluidOptions): Fluid
```

Fills a box or a closed mesh with liquid. See [Fluid](./fluid.md).

#### `addSoftbody(options)`

```ts
addSoftbody(options: SoftbodyOptions): Softbody
```

Turns a closed mesh into a soft body. See [Softbody](./softbody.md).

#### `addCloth(options)`

```ts
addCloth(options: ClothOptions): Cloth
```

Adds a rectangular cloth. See [Cloth](./cloth.md).

#### `addSmoke(options?)`

```ts
addSmoke(options?: SmokeOptions): Smoke
```

Adds a heated source that releases smoke. The simulation needs a `container`. See [Smoke](./smoke.md).

#### `addFloor(options?)`

```ts
addFloor(options?: { height?: number; friction?: number }): void
```

| Option     | Type     | Default | Description                            |
| ---------- | -------- | ------- | -------------------------------------- |
| `height`   | `number` | `0`     | Floor height, m.                       |
| `friction` | `number` | `0.5`   | From 0 (slippery) to about 1 (sticky). |

#### `addSphere(options)`

```ts
addSphere(options: { radius: number; center?: Vector3; follow?: Object3D; friction?: number }): void
```

| Option     | Type       | Default     | Description                                              |
| ---------- | ---------- | ----------- | -------------------------------------------------------- |
| `radius`   | `number`   | required    | Radius, m.                                               |
| `center`   | `Vector3`  | `(0, 0, 0)` | Ignored when `follow` is set.                            |
| `follow`   | `Object3D` | none        | An object to move with. The sphere stays centered on it. |
| `friction` | `number`   | `0.5`       |                                                          |

#### `addBox(options)`

```ts
addBox(options: { size: Vector3; center?: Vector3; rotation?: Euler; follow?: Object3D; friction?: number }): void
```

| Option     | Type       | Default     | Description                                                      |
| ---------- | ---------- | ----------- | ---------------------------------------------------------------- |
| `size`     | `Vector3`  | required    | Full edge lengths, m.                                            |
| `center`   | `Vector3`  | `(0, 0, 0)` | Ignored when `follow` is set.                                    |
| `rotation` | `Euler`    | none        | Ignored when `follow` is set.                                    |
| `follow`   | `Object3D` | none        | An object to move with. The box takes its position and rotation. |
| `friction` | `number`   | `0.5`       |                                                                  |

#### `addCapsule(options)`

```ts
addCapsule(options: { start: Vector3; end: Vector3; radius: number; follow?: Object3D; friction?: number }): void
```

| Option     | Type       | Default  | Description                                                                                 |
| ---------- | ---------- | -------- | ------------------------------------------------------------------------------------------- |
| `start`    | `Vector3`  | required | First end point, m.                                                                         |
| `end`      | `Vector3`  | required | Second end point, m.                                                                        |
| `radius`   | `number`   | required | Radius, m.                                                                                  |
| `follow`   | `Object3D` | none     | An object to move with. The capsule's middle stays on it, and it turns as the object turns. |
| `friction` | `number`   | `0.5`    |                                                                                             |

#### `addMesh(options)`

```ts
addMesh(options: { mesh: Mesh; resolution?: number; friction?: number }): void
```

Makes a closed mesh into a solid obstacle. It moves and turns with the mesh. The shape is prepared on the CPU when the simulation starts, which takes longer for detailed meshes.

| Option       | Type     | Default  | Description                                                                             |
| ------------ | -------- | -------- | --------------------------------------------------------------------------------------- |
| `mesh`       | `Mesh`   | required | A closed mesh with no holes.                                                            |
| `resolution` | `number` | `64`     | Detail of the shape along its longest side. Higher is more exact but slower to prepare. |
| `friction`   | `number` | `0.5`    |                                                                                         |

#### `start()`

```ts
start(): Promise<void>
```

Builds everything you've added. The first `step()` calls it for you, so you only need it to control when the work happens. It blocks the page while it runs, so show any loading screen before you call it. If it throws, your meshes are shown again and you can fix the problem and call it again.

#### `step(dt?)`

```ts
step(dt?: number): Promise<void>
```

Moves the simulation forward and updates what it draws. Await it before each `renderer.render()`.

| `dt`     | Behavior                                                                                                                                              |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| omitted  | Keeps the simulation in step with real time by running up to four 1/60 s steps. After a pause of more than 250 ms, it carries on without catching up. |
| `number` | Advances exactly `dt` seconds. Pass a constant such as `1 / 60`, not the time since the last frame.                                                   |

If you call `step()` while a step is still running, you get the running step's promise back.

#### `dispose()`

```ts
dispose(): void
```

Removes everything the simulation added to your scene, frees its GPU memory, and shows your original meshes again. A `material` you passed to `addCloth` is left for you to dispose. The simulation can't be used afterwards.

---

## Errors

| Message                                                               | When                                                                                                    |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| ``Simulation: give exactly one of `particles` or `particleRadius` ``  | You passed both options, or neither.                                                                    |
| `Simulation: particles must be a positive integer, …`                 | `particles` isn't a whole number above 0.                                                               |
| `Simulation: particleRadius must be a positive number of metres, …`   | `particleRadius` isn't a number above 0.                                                                |
| `Simulation: <call> must happen before the first step() or start()`   | You called an `add` method after the simulation started.                                                |
| `Simulation: add a fluid, smoke, soft body, or cloth before stepping` | You added only obstacles.                                                                               |
| ``addFluid: give either `box` or `mesh` ``                            | You passed both options, or neither.                                                                    |
| `addFluid: the fluid has no room; check its box and the container`    | The box is outside the container, or obstacles and solids fill it.                                      |
| `addSoftbody: the mesh is too small for the particle size`            | Not one particle fits inside the mesh. Raise `particles`, lower `particleRadius`, or scale the mesh up. |
| ``addSmoke: smoke needs a `container` for the air to fill``           | The simulation has no `container`.                                                                      |
| `addSmoke: a simulation can have one smoke source`                    | You called `addSmoke` twice.                                                                            |
| `<call>: gas and liquid can’t be simulated together, …`               | You mixed smoke with liquid, soft bodies, or cloth.                                                     |
| `Simulation.particleSystem is created on the first step`              | You read `particleSystem` or `loop` before `start()`.                                                   |
| `bakeMeshToSdf: mesh appears non-watertight — …`                      | A mesh passed to `addMesh` has holes.                                                                   |
| `SDFCollider.setTransform: …`                                         | A mesh passed to `addMesh` is scaled differently along different axes, scaled to zero, or mirrored.     |

Errors from building the simulation reject the promise that `start()` or `step()` returns.

---

## Limitations

- You can't add or remove anything after the simulation starts. To change what's in the scene, build a new `Simulation`.
- All particles share one size, so small objects next to a lot of liquid get few particles.
- Smoke can't share a simulation with liquid, soft bodies, or cloth, and each simulation has one smoke source. Use a second `Simulation` for smoke.
- The container can't move. To slosh liquid around, tilt `gravity` or move a box obstacle with `follow`.
- `follow` ignores the object's scale.
- An `addMesh` obstacle keeps the shape the mesh had when the simulation started. It can move and turn, but not bend.
- You can't add your own [materials](./extending.md) to a `Simulation`. Build the scene with the [low-level API](./core.md) instead.
- Friction between soft bodies and cloth is fixed. So are cloth air drag and damping, but you can change those on [`cloth.clothSystem`](./cloth-system.md) after `start()`.
- Without a `container`, liquid is only drawn near where it started, and the console warns you.
