use core::fmt;

macro_rules! define_id {
    ($name:ident) => {
        #[doc = concat!("Stable generational handle for a `", stringify!($name), "` row.")]
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name {
            index: u32,
            generation: u32,
        }

        impl $name {
            /// Dense slot index. It may be reused after removal with a new generation.
            pub const fn index(self) -> u32 {
                self.index
            }

            /// Generation guarding against use-after-remove.
            pub const fn generation(self) -> u32 {
                self.generation
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    formatter,
                    "{}({}:{})",
                    stringify!($name),
                    self.index,
                    self.generation
                )
            }
        }

        impl Handle for $name {
            fn from_parts(index: u32, generation: u32) -> Self {
                Self { index, generation }
            }

            fn parts(self) -> (u32, u32) {
                (self.index, self.generation)
            }
        }
    };
}

define_id!(PartId);
define_id!(WeldId);
define_id!(RegionId);
define_id!(ShapeFeatureId);
define_id!(RigidLinkId);
define_id!(BearingId);
define_id!(DriveLinkId);
define_id!(InputSeatLinkId);
define_id!(SeatControllerLinkId);

pub(crate) trait Handle: Copy {
    fn from_parts(index: u32, generation: u32) -> Self;
    fn parts(self) -> (u32, u32);
}

#[derive(Clone, Debug)]
struct Slot<T> {
    generation: u32,
    value: Option<T>,
}

/// Small typed arena used to keep public handles stable across unrelated edits.
#[derive(Clone, Debug)]
pub(crate) struct Arena<T, I> {
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
    len: usize,
    marker: core::marker::PhantomData<I>,
}

impl<T, I> Default for Arena<T, I> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            len: 0,
            marker: core::marker::PhantomData,
        }
    }
}

impl<T, I: Handle> Arena<T, I> {
    pub(crate) fn insert(&mut self, value: T) -> I {
        self.len += 1;
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            debug_assert!(slot.value.is_none());
            slot.value = Some(value);
            return I::from_parts(index, slot.generation);
        }

        let index = u32::try_from(self.slots.len()).expect("construction arena exhausted u32 IDs");
        self.slots.push(Slot {
            generation: 0,
            value: Some(value),
        });
        I::from_parts(index, 0)
    }

    pub(crate) fn get(&self, id: I) -> Option<&T> {
        let (index, generation) = id.parts();
        let slot = self.slots.get(index as usize)?;
        (slot.generation == generation)
            .then_some(slot.value.as_ref())
            .flatten()
    }

    pub(crate) fn get_mut(&mut self, id: I) -> Option<&mut T> {
        let (index, generation) = id.parts();
        let slot = self.slots.get_mut(index as usize)?;
        (slot.generation == generation)
            .then_some(slot.value.as_mut())
            .flatten()
    }

    pub(crate) fn remove(&mut self, id: I) -> Option<T> {
        let (index, generation) = id.parts();
        let slot = self.slots.get_mut(index as usize)?;
        if slot.generation != generation {
            return None;
        }
        let value = slot.value.take()?;
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(index);
        self.len -= 1;
        Some(value)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (I, &T)> {
        self.slots.iter().enumerate().filter_map(|(index, slot)| {
            slot.value.as_ref().map(|value| {
                (
                    I::from_parts(
                        index.try_into().expect("arena index fits u32"),
                        slot.generation,
                    ),
                    value,
                )
            })
        })
    }

    pub(crate) const fn len(&self) -> usize {
        self.len
    }

    pub(crate) const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::{Arena, PartId};

    #[test]
    fn stale_handle_does_not_alias_reused_slot() {
        let mut arena = Arena::<u32, PartId>::default();
        let stale = arena.insert(10);
        assert_eq!(arena.remove(stale), Some(10));
        let fresh = arena.insert(20);

        assert_eq!(stale.index(), fresh.index());
        assert_ne!(stale.generation(), fresh.generation());
        assert_eq!(arena.get(stale), None);
        assert_eq!(arena.get(fresh), Some(&20));
    }
}
