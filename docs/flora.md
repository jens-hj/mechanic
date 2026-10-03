# Flora

Trees grow from a small **species genome**. One recursive branching process,
in `crates/mechanic-world/src/generation/flora`, makes every species. Spruce,
oak, birch, willow, bamboo, and poplar differ only in the numbers in
[`worldgen/flora.ron`](../crates/mechanic-world/worldgen/flora.ron).

![Every species at four seeds, with a top view](flora/sheet.png)

The rows are spruce, oak, birch, willow, bamboo, and poplar, each at four seeds.
The last panel in each row is a top view. Grid lines are 1 m apart, with a
heavier line every 5 m.

## The genome

| Field | Type | What it does | Measure it owns |
|---|---|---|---|
| `height` | `(lo, hi)` m | Mature height. Each tree draws its own from the range. | measured height |
| `width` | ratio | Crown diameter ÷ height. It bounds every branch through the crown envelope. | crown width ÷ height |
| `girth` | ratio | Trunk base diameter ÷ height, shared among the stems. Every other radius follows the pipe model. | base diameter ÷ height |
| `stems` | `(min, max)` | Stems rising from the base, spread on a disc. | stem count |
| `crown_base` | 0..1 | Fraction of the height left as bare trunk before the first split. | lowest foliage ÷ height |
| `dominance` | 0..1 | Apical dominance. It sets the chance the leader survives a split; otherwise the leader forks into co-dominant limbs. It also blends the crown envelope from a dome (0) to a cone (1). | how high the original leader reaches ÷ height |
| `split_chance` | 0..1 | Chance that a node splits. | fraction of nodes that split on laterals |
| `split_count` | `(min, max)` | Children per split: whorls on stems, at most one per side on laterals. | children per split |
| `split_angle` | degrees | Angle between a child and its parent. | first-order branch angle from vertical |
| `tropism` | −1..1 | Bend per node after a lateral leaves its parent. Positive seeks the sun and negative hangs. Thin, flexible branches bend most. | mean rise of unbranched twigs |
| `wobble` | 0..1 | Random kink per segment. | stem path length ÷ chord |
| `foliage` | `(look, size, density)` | `look` is the surface look. `size` is the sleeve radius around terminal wood. `density` is the filled fraction of the sleeve. | foliage volume; filled fraction |
| `bark` | look | Surface look of the wood. | — |
| `roots` | `(spread, depth)` | `spread` is root radius ÷ crown radius; `depth` is how deep the roots go before levelling off, in metres. | root radius; root depth |

`cargo test -p mechanic-world flora` checks the table in
`every_genome_field_moves_its_own_metric`. It sweeps every field across its
range on an oak, using 8 seeds at each of 7 steps, and requires two things:
- each field moves its own measure monotonically (Spearman ρ ≥ 0.9);
- no two fields have the same effect, i.e. their scaled effects on every
  measure must not point the same way (|cos| < 0.9).

The sweeps are defined once, in `GenomeSweep::ALL`, and the gallery draws them.

## How a tree grows

1. **Draw the tree.**
   - Height `H` comes from `height`, crown radius `R = width·H/2`, and trunk
     radius `r0 = girth·H/2`.
   - Each of the `n` stems starts with radius `r0/√n`, standing on a
     golden-angle spiral.
   - With several stems, each leans outward by `split_angle·(1 − dominance)/2`.
   - A short flare cone, 1.4× wider over 0.3 m, roots each stem.
2. **Grow each axis.**
   - Each axis runs in `clamp(round(length·20/H), 2, 20)` segments.
   - Radius tapers linearly toward the twig radius (half a 5 cm cell).
   - Wobble kinks every segment.
   - Tropism bends laterals from their second node on, weighted by
     `(1 − r/r0)²`, so leaders hold their line.
3. **Split.**
   - A node splits with probability `split_chance`. Stems start splitting only
     above `crown_base`.
   - **The leader survives** (probability `dominance`): laterals leave at
     `split_angle`, with radius `r·lerp(0.7, 0.35, dominance)`, and the
     leader keeps what the pipe model leaves it.
     - On stems they spiral by the golden angle.
     - On laterals they are flat: at most one to each side, in the branch's
       horizontal plane.
   - **The leader forks** (probability `1 − dominance`): two or more
     co-dominant children of radius `r/√m` take the remaining length, at up to
     30° apart.
4. **Lengths.**
   - A stem's lateral is as long as it must be for its **tip** to land on the
     crown envelope. The envelope is `lerp(dome, cone, dominance)·R`, less the
     foliage radius.
   - Deeper laterals are 0.6 of what remains of their parent.
   - Branches are at most 240× their base radius long.
5. **Bounds.**
   - Laterals that leave the crown envelope are pruned. Co-dominant forks turn
     back in along its edge instead.
   - Roots stop at `spread·R`, never climb, and level off at `depth`.
   - Wood stops 0.4 m above the ground, so hanging twigs end short of it.
6. **Foliage.**
   - Every non-stem axis carries a capsule sleeve of radius `size` beyond its
     last split, and a ball 1.3× wider at its tip.
   - Holes dent the blobs' union from outside: smooth noise sets how deep,
     from nothing to the full blob radius. The noise is remapped through its
     own distribution, and its target is biased so that `density` is the
     filled fraction (`foliage_density_sets_filled_fraction`).
   - The noise's wavelength is 1.25 × its steepest slope × the dent depth, so
     a dent never changes faster than distance. Walking from any leaf toward
     its twig, the foliage only gets denser: no leaf clump floats free.
7. **Roots.** Five main roots, plus a taproot when `depth > spread·R/2`, grow by
   the same process with fixed branching numbers.

Recursion needs no depth limit in the genome. A branch thinner than half a
cell can no longer split. `MAX_ORDER = 6` and a 20,000-segment budget are
safety stops that no preset reaches.

A `TreeModel` holds tapered capsules for wood and roots and capsules for
foliage, indexed by 1 m buckets. `TreeModel::sample` returns a signed
density, exact to 0.3 m outside every primitive, together with the part that
dominates: wood, root, or foliage.

Every lattice draws wood at least 0.9 lattice spacings thick
(`sample_at_stride`): 4.5 cm at the finest level, 18 cm at the 20 cm stride.
No point on a branch's axis lies farther than half a cube diagonal (0.87
spacings) from a lattice corner, so every point of the axis has a solid
corner beside it, and those corners join edge to edge. Thinner wood broke
into floating shards: 933 pieces on one spruce at the finest level.
`every_branch_holds_together_at_every_grown_stride` and
`no_wood_or_crown_floats_at_coarse_levels_of_detail` hold both the model and
the meshed field to this.

## Gallery

```
cargo run -p mechanic-bench --release --bin flora-gallery -- --out <dir> [--sweep all|<field>] [--species oak] [--flora <file>] [--seeds 4] [--no-voxels]
```

It writes these images:
- `sheet.png`: the picture above.
- `species-<name>.png`: two trees, larger.
- `voxels-<name>.png`: the tree as terrain cells.
  - the front-most solid 5 cm cells;
  - the same at the 20 cm stride of the second level of detail;
  - a slice through the trunk that shows the holes in the foliage.
- `sweep-<field>.png`: seven steps of one field on the base species, two seeds
  each.

It prints one JSONL line per tree, with every measure, `grow_ms`,
`sample_ns_per_cell` and `filled_fraction`, and one per sweep step.

In the release build on the M1 Pro, every preset grows in under 2 ms with at
most about 3,000 segments. A density sample costs 15–130 ns.

## Tuning log

These are visual verdicts on `sheet.png`, phase 1. Each species must be
recognisable by silhouette alone.

| Round | Change | Verdict |
|---|---|---|
| 1 | First presets | Oak is an umbrella, willow hangs through the ground, the spruce top droops and forks, roots climb into the air. |
| 2 | Tropism only on laterals, roots never climb, wood stops above ground, 20 internodes | Spruce is a cone, but its low whorls make a ball. Oak, bamboo, and poplar read correctly. Poplar grows whips above the crown. |
| 3 | Slenderness cap, flat laterals | Poplar is a column. The willow's short twigs do not hang. |
| 4 | Tropism per node, not per metre | Willow weeps. |
| 5 | Laterals split at most once to each side | The spruce ball is gone: clean whorls to the ground. |
| 6 | Crown envelope prunes laterals; forks bend back in; tips aim at the envelope | `width` and `dominance` now control the crown. Birch is a slender oval on pale stems. Every species reads correctly. |
| 7 (phase 2) | Foliage holes widened from 15 to 40 cm for terrain cost | Leaves read as clumps with gaps rather than lace. Every species still reads correctly. |

Final verdicts:
- **Spruce:** a narrow cone of flat whorls to the ground, dark needles.
- **Oak:** a broad dome on a short, gnarled bole with heavy surface roots.
- **Birch:** one to three pale, slender stems under a light, oval crown.
- **Willow:** a rounded crown with curtains hanging almost to the ground.
- **Bamboo:** a dense clump of straight culms, leafy in the upper half.
- **Poplar:** a tall, narrow column.

## Trees in the world

**Placement.**
- Biomes list their trees in `flora`; see [world generation](world-generation.md#flora).
- Each tree is a pure function of the world seed, its layer and its grid
  cell. Placing one costs about 25 ground samples and is cached per thread.
- Growing one takes about a millisecond. Grown trees live in a shared cache
  with a 192 MB budget, behind a small per-thread table.

**In the terrain field.**
- The field's density is `max(ground, tree)`, so trees are ordinary terrain.
- Wood is the `Wood` material, painted with the species' `bark` look. Bark
  uses the wood texture, recoloured per species.
- Leaves are `Foliage`, painted with its `foliage` look on the recoloured
  grass texture.
- Digging, breakage, clumps and the matter books treat both like any other
  ground:
  - **Wood** is strong, light (600 kg/m³) and stays in pieces when cut, like
    ore.
  - **Foliage** gives way under a few kPa (`foliage_breaks_where_soil_holds`)
    and crumbles into clods.
- Neither erodes. The terrain brush does not lay them: the hotbar and
  material wheel offer `TerrainMaterial::BRUSHABLE`.

**Roots.**
- Roots add no ground; they paint as wood the ground they run through, below
  its top 0.3 m.
- Grass, water and erosion therefore keep their topsoil, and digging near a
  tree turns up wood.

**Water.**
- Water runs through leaves but not wood (`water_runs_through_canopies_but_not_trunks`).
- `TerrainField::water_density` and `water_density_lattice` see the ground
  that way, and native grass is found under trees.

**Distance.**
- Terrain sampled at up to a 20 cm stride (levels 0 to 2, out to 22–64 m)
  shows grown trees.
- Coarser levels show impostors: the trunk under a solid crown shaped by the
  species' envelope (cone to dome by `dominance`), which needs no growing.

**Bounds.**
- `classify` and `interval` are tree-aware from cheap placement bounds, so no
  box holding a tree is ever judged empty.
- Meshing culls blocks by the ground alone and raises the trees over every
  block afterwards.

**Ground queries.** `topmost_surface` and `surface_height` find the ground
under any canopy.

### Cost

`terrain-cut` (release, seed 42, 10 workers, M1 Pro) compares each biome's
full streaming cut with and without its flora. CPU is sampling plus
extraction.

| Biome | CPU without trees | CPU with trees | Ratio | Triangles without → with |
|---|---|---|---|---|
| Verdant Hills | 161.6 s | 192.7 s | 1.19× | 9.7 M → 19.0 M |
| Titan Crags | 220.4 s | 239.2 s | 1.09× | 10.6 M → 14.0 M |
| Shelf Mire | 145.3 s | 153.6 s | 1.06× | 7.3 M → 8.4 M |

Selection takes about 1 s longer per cut, from tree placement.

Measure a change by copying `crates/mechanic-world/worldgen` with the
`flora` lists removed and running `terrain-cut --worldgen <dir>` against
both.

### Known limits

- Twigs are drawn as thick as the lattice can hold them, so distant trees
  look stouter: twigs are 36 cm across at the 20 cm stride.
- Bark and leaf looks recolour the wood and grass textures; there are no
  dedicated bark or leaf maps.
- There is no growth, felling, or tree-specific harvesting: trees are part of
  the seed's ground.
- Saved worlds made before trees are outdated by the new worldgen digest and
  brick format, and are not migrated.

## Decisions

- **Measures.**
  - `dominance` owns how far the leader reaches, not where the crown is widest.
    Widest height also depends on `split_angle`, because a 45° branch cannot
    reach out far inside a narrowing cone.
  - `split_chance` owns the fraction of nodes that split, counted on laterals
    thick enough to split. Total segment count also grows with crown size, so it
    tracked `crown_base`.
  - `wobble` owns stem tortuosity, which tropism cannot touch.
- **The cross-field check** is the plan's pruning rule: no two fields may have
  the same effect. A first version required that no field move another
  field's measure more than that field does. Crown width and shape emerge from
  several fields by design, so that version failed for reasons that were not
  defects.
- **No genome field was pruned.** All fifteen pass monotonicity and
  distinctness, `wobble` and `crown_base` included.
- **Preset changes from the plan's starting values:**
  - spruce: `dominance` 1.0, `girth` 0.028
  - oak: `tropism` 0.12
  - birch: `girth` 0.028, `split_chance` 0.6, `split_count` (2, 3), `split_angle` 40, `tropism` −0.35, `foliage` size 0.55 and density 0.55
  - willow: `crown_base` 0.3, `tropism` −0.95
  - bamboo: `width` 0.3, `girth` 0.03, `dominance` 0.9, `split_chance` 0.3
  - poplar: `dominance` 0.85, `girth` 0.036
  - Girth was raised on the three slenderest species after the first look in
    the app: their trunks read as sticks at 5 cm cells.
- **Validation errors use the existing `WorldgenError::Invalid`**, naming the
  species. A dedicated variant would add nothing.
- **Images:** the sheet lives in `docs/flora/`, following the per-feature image
  folders used elsewhere in `docs/`.
- **Roots paint, they do not build.**
  - First version: roots added solid ground. Shore oaks then pushed roots out
    of the soil into lakes and channels, damming them.
  - Second version: roots repainted the topsoil as wood, so it stopped
    eroding and soaking.
  - Now roots only repaint ground below its top 0.3 m.
- **`topmost_surface` passes through trees.**
  - Its callers place spawns, pits, vehicles and benchmark probes on the
    ground.
  - Tests of the ground's own shape use `testing::treeless_field`. These
    include LOD smoothness, and two water tests whose helpers stood on
    canopies.
- **`TreeModel::sample` reports only true densities within 0.3 m.** A floor
  of −0.3 lifted the ground's own density wherever a tree's bucket overlapped
  it, moving the terrain surface.
- **Trees paint the open side of their surface too.** A mesh crossing takes
  its open corner's look, so air corners beside a trunk must know the trunk.
- **Placement asks for open sky up to the tallest tree and a level 1 m
  footing.**
  - Without the sky check, trees grew through carve roofs.
  - Without the footing check, trees stood on ledge edges.
  - The footing check replaced a local slope probe.
- **Bark uses the shipped wood texture** (`TextureSet::Wood`) rather than
  recoloured stone.
- **Foliage holes are as wide as the leaves are deep.**
  - First version: 0.15 m holes. The finer sponge tripled a woodland's
    triangles.
  - Second version: 0.4 m holes, cut anywhere in the blob. These left leaf
    clumps floating beside the crowns.
  - Third version: radial holes, read where each direction from a twig
    meets its blob. These cannot float, but overlapping blobs each fill
    independently: a density of 0.2 filled 75 %.
  - Now one dent field, gentler than distance, cuts the blobs' union:
    holes span 1.6 m on a spruce and 5.7 m on an oak.
- **Wood is never thinner than its lattice can hold.** Without this floor,
  twigs and the tops of slender trunks broke into floating shards at every
  level. Impostor trunks were 10 cm thick on 40 cm lattices. Edits keep the
  finest level's densities, so dug wood far away is drawn at its true
  thickness.
- **Grown trees end at the 20 cm stride; impostors take over.** With grown
  trees at 40 cm, Verdant Hills cost 2.5×. Most of that was culling by the
  tree-aware interval and painting by per-vertex tree lookups. Both are now
  gathered once per lattice.

