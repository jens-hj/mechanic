//! Union-find over part indices, used to collapse welds into compounds.

pub(super) const fn ordered_pair(a: u32, b: u32) -> [u32; 2] {
    if a < b { [a, b] } else { [b, a] }
}

#[derive(Clone, Debug)]
pub(super) struct DisjointSet {
    pub(super) parent: Vec<usize>,
    pub(super) rank: Vec<u8>,
}

impl DisjointSet {
    pub(super) fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            rank: vec![0; len],
        }
    }

    pub(super) fn find(&mut self, mut item: usize) -> usize {
        let mut root = item;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        while self.parent[item] != item {
            let parent = self.parent[item];
            self.parent[item] = root;
            item = parent;
        }
        root
    }

    pub(super) fn union(&mut self, a: usize, b: usize) {
        let mut root_a = self.find(a);
        let mut root_b = self.find(b);
        if root_a == root_b {
            return;
        }
        if self.rank[root_a] < self.rank[root_b] {
            core::mem::swap(&mut root_a, &mut root_b);
        }
        self.parent[root_b] = root_a;
        if self.rank[root_a] == self.rank[root_b] {
            self.rank[root_a] += 1;
        }
    }
}
