# Star systems, moons, and the night sky

Every world belongs to a star system chosen by its seed
(`mechanic_world::CelestialSystem`). The sky is a pure function of that system
and the world's elapsed solar time, so nothing beyond the day and time of day is
saved, and every body moves on its own orbit rather than being pinned to the sun.

## What a seed chooses

| Choice | Range |
| --- | --- |
| Stars | One (45%), a close pair the planet orbits (20%), one star with a distant companion (15%), a close pair with a distant companion (12%), or one star with a distant pair (8%). |
| Star masses | Main sequence; luminosity, radius, and temperature follow from mass, and temperature sets each star's colour. |
| Planet orbit | Circular, at the distance where the inner stars give Earth's sunlight. The year lasts 48 to 96 solar days. |
| Observer | Latitude 25° to 48° north; axial tilt 8° to 28°, always at least 6° below the latitude, so the noon sun stands to the south. |
| Moons | None (10%), one (50%), two (28%), or three (12%), 10 to 60 planet radii out, each with its own inclination of up to 6°, reflectance, tint, and markings. |

Periods follow Kepler's third law. Star orbits are scaled so that the planet's
orbit takes the chosen year, and a close pair is kept 3.6 to 6 of its
separations inside the planet's orbit, so it circles in days and its two suns
part and meet. Moon periods follow from their distance about an Earth-mass
planet, from about two days to a third of a year. Distant companions sit 60 to
400 AU away; they barely move against the stars, give several to a few hundred
lux, and show as brilliant points that can light the night and cast shadows.

## How the sky moves

- The planet turns once per solar day relative to its sun and one extra time per
  year relative to the stars, so the stars rise a little earlier each night.
- Local mean noon is at 12:00. The axial tilt moves the sun along the ecliptic:
  noon height and day length follow the seasons, and true noon wanders by a few
  minutes.
- Each moon's position is computed from the observer on the surface, not the
  planet's centre, so near moons shift with the observer's turning. A moon rises
  at a different hour every night and runs through its phases.
- A moon in front of a sun hides part of its disk, dimming that sun's light by
  the covered share; a nearer star does the same to a farther one. A moon in the
  planet's umbra or penumbra loses that star's light.

Directions are in the horizon frame, which is the world frame: `x` east, `y` up,
`z` south.

## Rendering

The app turns the system into lights and drawables each frame (`sky` module):

- Each star is a directional light carrying a Bevy sun disk sized to its
  apparent diameter. Disks smaller than 0.2° are drawn at that size with the
  same total light.
- Moons are camera-facing quads 30 km away, beyond the finite world but inside
  the atmosphere's aerial-perspective range, so daylight veils them. Their
  shader (`assets/shaders/moon.wgsl`) lights the sphere from every star, adds
  planetshine to the near side, and eases a dark-adapted eye's response so a
  full moon does not blow out at night.
- One moonlight carries every risen moon's reflected light from the brightest
  one's direction. Moonlight is boosted 13× over photometry, which makes a full
  moon like Earth's four lux.
- Only the brightest light above the horizon casts shadows.
- Exposure follows the suns through twilight and day, rises a little for two
  bright suns, and adapts at night to bright moons and distant stars.
- The background stars and the galaxy's band are baked once into a cubemap in
  the ecliptic frame. Star counts follow N(>L) ∝ L^-1.5 and crowd into the band,
  which has a brighter core and a dust lane.

## Verification

`cargo test -p mechanic-world celestial` covers system variety, sunrise and noon
direction, noon height through the year, sidereal drift, moon phases and
elongation, lunar eclipses, circumbinary suns, distant companions, and disk
overlap. The app tests cover shadow selection, exposure, moon shading, and the
lights and disks kept per system.

The opt-in GPU capture renders twin suns at dusk and by day, a moon at three
phases, a moon by day, a moonless sky, the galaxy's core, and a distant pair at
night, and measures a three-star, three-moon system against a lone sun:

```sh
cargo xtask test --exclude mechanic-bench celestial_captures_show_suns_moons_and_stars -- --ignored --nocapture
```

## Captured evidence, 2026-10-03

Apple M1 Pro, Metal, 768 × 512, 4× MSAA, development profile with optimized
dependencies.

- Suns: [twin suns at dusk](celestial/2026-10-03/twin-suns-dusk.png), [twin suns by day](celestial/2026-10-03/twin-suns-day.png).
- Moon phases (6° field): [10%](celestial/2026-10-03/moon-10-percent.png), [50%](celestial/2026-10-03/moon-50-percent.png), [90%](celestial/2026-10-03/moon-90-percent.png); [moon by day](celestial/2026-10-03/moon-by-day.png).
- Night: [moonless sky](celestial/2026-10-03/night-sky.png), [galaxy core](celestial/2026-10-03/galaxy.png), [distant pair lighting the night](celestial/2026-10-03/distant-pair-at-night.png).
- [Shot times and timings](celestial/2026-10-03/celestial.json).

The frame timings in that file were taken while other sessions were compiling
on the same machine, and two runs disagreed by more than 2×. They do not
establish the system's cost; a quiet machine must repeat them. Each extra star
adds one directional light to the atmosphere's scattering loops and each moon
one quad; only one light ever renders shadows.
