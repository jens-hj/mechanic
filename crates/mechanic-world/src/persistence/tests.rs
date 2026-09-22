use std::{
    fs,
    sync::atomic::{AtomicU32, Ordering},
    time::Duration,
};

use mechanic_core::{CREATION_FORMAT_VERSION, CreationDocument, DimensionLinkId};

use super::{
    AutosaveState, FrozenCreationDoc, OpenWorldOutcome, SavedWorldStatus, WORLD_FORMAT_VERSION,
    WorldCreationInstanceDoc, WorldDocument, WorldPoseDoc, WorldSaveError, WorldStore,
};
use crate::{TerrainField, TerrainOctree, WorldPosition, WorldSeed};

struct TempDir(std::path::PathBuf);

impl TempDir {
    fn new() -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        Self(std::env::temp_dir().join(format!(
            "mechanic-world-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn empty_creation() -> CreationDocument {
    CreationDocument {
        version: CREATION_FORMAT_VERSION,
        name: "Anchor".to_owned(),
        frames: vec![mechanic_core::ConstructionFrameDoc {
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
        }],
        part_frames: Vec::new(),
        region_frames: Vec::new(),
        parts: Vec::new(),
        welds: Vec::new(),
        rigid_links: Vec::new(),
        gear_links: Vec::new(),
        bearings: Vec::new(),
        drive_links: Vec::new(),
        input_seat_links: Vec::new(),
        physical_inputs: Vec::new(),
        seat_controller_links: Vec::new(),
        gearbox_configs: Vec::new(),
        regions: Vec::new(),
        shape_features: Vec::new(),
        sockets: Vec::new(),
    }
}

#[test]
fn world_instance_and_brick_round_trip() {
    let temporary = TempDir::new();
    let store = WorldStore::new(&temporary.0);
    let field = TerrainField::new(WorldSeed(84));
    let world = WorldDocument::new("Violet Reach", field.seed(), field.safe_spawn());
    let world_path = store.save_world(&world).unwrap();
    assert_eq!(
        store.load_world(world_path.parent().unwrap()).unwrap(),
        world
    );

    let instance = WorldCreationInstanceDoc {
        id: 17,
        creation: empty_creation(),
        root_pose: WorldPoseDoc::default(),
        joint_coordinates: vec![0.25, -0.5],
    };
    let instance_path = store.save_instance(&world.name, &instance).unwrap();
    assert_eq!(store.load_instance(&instance_path).unwrap(), instance);

    let mut edits = TerrainOctree::default();
    edits.promote(&field, crate::BrickCoord::new(0, 0, 0));
    let brick = edits.brick(crate::BrickCoord::new(0, 0, 0)).unwrap();
    let brick_path = store.save_brick(&world.name, brick).unwrap();
    assert_eq!(store.load_brick(&brick_path).unwrap(), *brick);
}

#[test]
fn material_snapshot_keeps_terrain_and_clumps_together() {
    let temporary = TempDir::new();
    let store = WorldStore::new(&temporary.0);
    let field = TerrainField::new(WorldSeed(84));
    let mut terrain = TerrainOctree::default();
    let centre = crate::WorldPosition(bevy_math::DVec3::new(0.025, 200.025, 0.025));
    terrain
        .add_sphere(&field, centre, 0.1, crate::TerrainMaterial::Iron)
        .unwrap();
    let cell = centre.cell().unwrap();
    let source = crate::ExtractionCell {
        cell,
        sample: terrain.sample_cell(&field, cell),
        throw: bevy_math::DVec3::ZERO,
    };
    let mut broken = crate::ClumpCollection::default();
    broken
        .extract(&mut terrain, &field, &[source], false)
        .unwrap();
    store
        .save_material_state("material", &terrain, &broken)
        .unwrap();
    let (loaded, clumps) = store.load_material_state("material").unwrap();
    assert_eq!(clumps, broken);
    assert!(!loaded.sample_cell(&field, cell).is_solid());
    let path = store.directory_for("material").join("material.bin");
    let mut bytes = fs::read(&path).unwrap();
    bytes.pop();
    fs::write(&path, bytes).unwrap();
    assert!(store.load_material_state("material").is_err());
}

#[test]
fn paired_spaces_publish_and_reload_one_generation() {
    let temporary = TempDir::new();
    let store = WorldStore::new(&temporary.0);
    let field = TerrainField::new(WorldSeed(91));
    let mut world = WorldDocument::new("Paired", field.seed(), field.safe_spawn());
    let world_space = WorldCreationInstanceDoc {
        id: 1,
        creation: empty_creation(),
        root_pose: WorldPoseDoc::default(),
        joint_coordinates: Vec::new(),
    };
    let garage_space = WorldCreationInstanceDoc {
        id: 2,
        creation: CreationDocument {
            name: "Garage".to_owned(),
            ..empty_creation()
        },
        root_pose: WorldPoseDoc::default(),
        joint_coordinates: Vec::new(),
    };

    store
        .save_space_pair(&mut world, &world_space, &garage_space)
        .unwrap();
    assert_eq!(world.construction_generation, 1);
    let published = store.load_world(&store.directory_for("Paired")).unwrap();
    assert_eq!(published.construction_generation, 1);
    let interrupted = store.directory_for("Paired").join("generations").join("2");
    fs::create_dir_all(&interrupted).unwrap();
    fs::write(interrupted.join("world.ron"), b"incomplete generation").unwrap();
    assert_eq!(
        store.load_space_pair(&published).unwrap(),
        Some((world_space, garage_space))
    );
}

fn frozen_world() -> WorldDocument {
    let mut world = WorldDocument::new("Frozen", WorldSeed(42), WorldPosition::default());
    world.active_dimension_link = Some(DimensionLinkId(3));
    world.construction_generation = 7;
    world.frozen_creation = Some(FrozenCreationDoc {
        link: DimensionLinkId(3),
        target: WorldPosition(bevy_math::DVec3::new(10_000.0, 1.25, -20_000.0)),
        heading: 3,
        construction_generation: 7,
    });
    world
}

#[test]
fn frozen_target_round_trips_with_matching_published_generation() {
    let temporary = TempDir::new();
    let store = WorldStore::new(&temporary.0);
    let mut world = frozen_world();
    let space = WorldCreationInstanceDoc {
        id: 1,
        creation: empty_creation(),
        root_pose: WorldPoseDoc::default(),
        joint_coordinates: Vec::new(),
    };
    let original = world.frozen_creation.unwrap();
    store.save_space_pair(&mut world, &space, &space).unwrap();
    assert_eq!(world.construction_generation, 8);
    assert_eq!(
        world.frozen_creation.unwrap(),
        FrozenCreationDoc {
            construction_generation: 8,
            ..original
        }
    );
    assert_eq!(
        store.load_world(&store.directory_for(&world.name)).unwrap(),
        world
    );
}

#[test]
fn invalid_frozen_state_is_rejected_on_save_and_load() {
    let temporary = TempDir::new();
    let store = WorldStore::new(&temporary.0);
    for case in 0..5 {
        let mut world = frozen_world();
        match case {
            0 => {
                world
                    .frozen_creation
                    .as_mut()
                    .unwrap()
                    .construction_generation = 6;
            }
            1 => {
                world.construction_generation = 0;
                world
                    .frozen_creation
                    .as_mut()
                    .unwrap()
                    .construction_generation = 0;
            }
            2 => world.frozen_creation.as_mut().unwrap().heading = 4,
            3 => world.active_dimension_link = None,
            _ => world.frozen_creation.as_mut().unwrap().target.0.x = f64::INFINITY,
        }
        assert!(matches!(
            store.save_world(&world),
            Err(WorldSaveError::InvalidFrozenCreation { .. })
        ));
        let directory = store.directory_for(&world.name);
        fs::create_dir_all(&directory).unwrap();
        fs::write(directory.join("world.ron"), ron::to_string(&world).unwrap()).unwrap();
        assert!(matches!(
            store.load_world(&directory),
            Err(WorldSaveError::InvalidFrozenCreation { .. })
        ));
    }
}

#[test]
fn failed_manifest_publication_restores_frozen_and_construction_generations() {
    let temporary = TempDir::new();
    let store = WorldStore::new(&temporary.0);
    let mut world = frozen_world();
    let original = world.clone();
    let space = WorldCreationInstanceDoc {
        id: 1,
        creation: empty_creation(),
        root_pose: WorldPoseDoc::default(),
        joint_coordinates: Vec::new(),
    };
    // A directory at the manifest path forces the final atomic rename to fail.
    fs::create_dir_all(store.directory_for(&world.name).join("world.ron")).unwrap();
    assert!(matches!(
        store.save_space_pair(&mut world, &space, &space),
        Err(WorldSaveError::Io { .. })
    ));
    assert_eq!(world, original);
}

#[test]
fn corrupt_brick_reports_exact_file_and_preserves_it() {
    let temporary = TempDir::new();
    let store = WorldStore::new(&temporary.0);
    let path = temporary.0.join("broken.bin");
    fs::create_dir_all(&temporary.0).unwrap();
    fs::write(&path, b"original recovery data").unwrap();
    let error = store.load_brick(&path).unwrap_err();
    assert!(matches!(error, WorldSaveError::Brick { path: ref failed, .. } if failed == &path));
    assert_eq!(fs::read(&path).unwrap(), b"original recovery data");
}

#[test]
fn autosave_debounces_but_never_waits_more_than_thirty_seconds() {
    let mut autosave = AutosaveState::default();
    autosave.mutate(Duration::ZERO);
    autosave.mutate(Duration::from_secs(1));
    assert!(!autosave.due(Duration::from_millis(2_999)));
    assert!(autosave.due(Duration::from_secs(3)));
    autosave.mutate(Duration::from_secs(29));
    assert!(autosave.due(Duration::from_secs(30)));
    autosave.saved();
    assert!(!autosave.is_dirty());
}

#[test]
fn list_classifies_version_two_outdated_and_corrupt_manifests() {
    let temporary = TempDir::new();
    let store = WorldStore::new(&temporary.0);
    let field = TerrainField::new(WorldSeed(11));
    let current = WorldDocument::new("Current", field.seed(), field.safe_spawn());
    store.save_world(&current).unwrap();

    let outdated_path = store.directory_for("Outdated").join("world.ron");
    fs::create_dir_all(outdated_path.parent().unwrap()).unwrap();
    let mut outdated = WorldDocument::new("Outdated", field.seed(), field.safe_spawn());
    outdated.version = WORLD_FORMAT_VERSION - 1;
    fs::write(&outdated_path, ron::to_string(&outdated).unwrap()).unwrap();

    let corrupt_path = store.directory_for("Corrupt").join("world.ron");
    fs::create_dir_all(corrupt_path.parent().unwrap()).unwrap();
    fs::write(
            &corrupt_path,
            format!(
                "(version:{WORLD_FORMAT_VERSION},name:\"Corrupt\",generator_version:1,seed:11,last_played_unix_seconds:3,player_pose:broken)"
            ),
        )
        .unwrap();

    let listed = store.list();
    assert_eq!(listed.len(), 3);
    assert!(
        listed
            .iter()
            .any(|world| world.status == SavedWorldStatus::Current)
    );
    assert!(
        listed
            .iter()
            .any(|world| world.status == SavedWorldStatus::Outdated)
    );
    assert!(listed.iter().any(|world| matches!(
        world.status,
        SavedWorldStatus::Corrupt { ref file, .. } if file == &corrupt_path
    )));
}

#[test]
fn opening_outdated_deletes_only_exact_child_and_corrupt_current_is_preserved() {
    let temporary = TempDir::new();
    let store = WorldStore::new(&temporary.0);
    let field = TerrainField::new(WorldSeed(12));
    let mut outdated = WorldDocument::new("Old", field.seed(), field.safe_spawn());
    outdated.version = WORLD_FORMAT_VERSION - 1;
    let directory = store.directory_for(&outdated.name);
    let manifest = directory.join("world.ron");
    fs::create_dir_all(&directory).unwrap();
    fs::write(&manifest, ron::to_string(&outdated).unwrap()).unwrap();
    let entry = store
        .list()
        .into_iter()
        .find(|entry| entry.path == directory)
        .unwrap();
    assert_eq!(
        store.open_entry(&entry).unwrap(),
        OpenWorldOutcome::OutdatedRemoved {
            path: directory.clone()
        }
    );
    assert!(!directory.exists());

    let corrupt_directory = store.directory_for("Current corrupt");
    let corrupt_manifest = corrupt_directory.join("world.ron");
    fs::create_dir_all(&corrupt_directory).unwrap();
    let bytes = format!(
        "(version:{WORLD_FORMAT_VERSION},name:\"Current corrupt\",generator_version:1,seed:12,last_played_unix_seconds:4,player_pose:broken)"
    );
    fs::write(&corrupt_manifest, &bytes).unwrap();
    let corrupt = store
        .list()
        .into_iter()
        .find(|entry| entry.path == corrupt_directory)
        .unwrap();
    assert!(matches!(
        store.open_entry(&corrupt),
        Err(WorldSaveError::CorruptCurrent { ref path, .. }) if path == &corrupt_manifest
    ));
    assert_eq!(fs::read_to_string(corrupt_manifest).unwrap(), bytes);
}

#[test]
fn dirty_leaf_save_and_hierarchy_reconstruction_are_deterministic() {
    let temporary = TempDir::new();
    let store = WorldStore::new(&temporary.0);
    let field = TerrainField::new(WorldSeed(13));
    let world = WorldDocument::new("Octree", field.seed(), field.safe_spawn());
    store.save_world(&world).unwrap();
    let mut terrain = TerrainOctree::default();
    terrain
        .excavate_sphere(
            &field,
            WorldPosition(bevy_math::DVec3::new(
                0.0,
                field.surface_height(0.0, 0.0),
                0.0,
            )),
            0.3,
        )
        .unwrap();
    let saved = store.save_dirty_leaves(&world.name, &mut terrain).unwrap();
    assert!(!saved.is_empty());
    assert_eq!(terrain.dirty_leaves().count(), 0);
    let rebuilt = store.load_octree(&world.name).unwrap();
    assert_eq!(
        rebuilt.node(crate::TerrainNodeId::ROOT),
        terrain.node(crate::TerrainNodeId::ROOT)
    );
    assert_eq!(
        rebuilt.brick_coordinates().collect::<Vec<_>>(),
        terrain.brick_coordinates().collect::<Vec<_>>()
    );
}
