//! Packed median-split bounds trees, following the world's construction BVH.
use super::overlaps;
use bevy_math::DVec3;

type Bounds = [DVec3; 2];

pub(super) struct Tree {
    nodes: Vec<Node>,
}
struct Node {
    // Equal children mark a leaf. Internal nodes store their descendant count
    // in `item`; only leaves need a collider/body row. Keep each node compact.
    children: [usize; 2],
    item: usize,
}
impl Node {
    fn children(&self) -> Option<[usize; 2]> {
        (self.children[0] != self.children[1]).then_some(self.children)
    }
}
impl Tree {
    pub(super) fn new(mut items: Vec<usize>, bounds: &[Bounds]) -> Self {
        let mut tree = Self { nodes: Vec::new() };
        if !items.is_empty() {
            tree.build(&mut items, bounds);
        }
        tree
    }
    fn build(&mut self, items: &mut [usize], bounds: &[Bounds]) -> usize {
        let index = self.nodes.len();
        self.nodes.push(Node {
            children: [0; 2],
            item: items[0],
        });
        if items.len() > 1 {
            let aggregate = items
                .iter()
                .fold([DVec3::INFINITY, DVec3::NEG_INFINITY], |[lo, hi], &row| {
                    [lo.min(bounds[row][0]), hi.max(bounds[row][1])]
                });
            let extent = aggregate[1] - aggregate[0];
            let axis = if extent.x >= extent.y && extent.x >= extent.z {
                0
            } else if extent.y >= extent.z {
                1
            } else {
                2
            };
            items.sort_unstable_by(|&a, &b| {
                (bounds[a][0][axis] + bounds[a][1][axis])
                    .total_cmp(&(bounds[b][0][axis] + bounds[b][1][axis]))
                    .then(a.cmp(&b))
            });
            let (a, b) = items.split_at_mut(items.len() / 2);
            let children = [self.build(a, bounds), self.build(b, bounds)];
            self.nodes[index].children = children;
            self.nodes[index].item = items.len();
        }
        index
    }
    pub(super) fn refit(&self, bounds: &[Bounds], output: &mut Vec<Bounds>) {
        output.resize(self.nodes.len(), [DVec3::ZERO; 2]);
        for (index, node) in self.nodes.iter().enumerate().rev() {
            output[index] = if let Some([a, b]) = node.children() {
                [
                    output[a][0].min(output[b][0]),
                    output[a][1].max(output[b][1]),
                ]
            } else {
                bounds[node.item]
            };
        }
    }
    /// Traverse overlapping nodes of both trees; ties split the first tree.
    /// Returns how many node pairs were tested.
    pub(super) fn pairs(
        &self,
        bounds: &[Bounds],
        other: &Self,
        other_bounds: &[Bounds],
        stack: &mut Vec<[usize; 2]>,
        output: &mut Vec<[usize; 2]>,
    ) -> usize {
        stack.clear();
        if self.nodes.is_empty() || other.nodes.is_empty() {
            return 0;
        }
        // Each pair is tested as it is reached, so only overlapping pairs are
        // stacked; they are still visited depth first, left before right.
        let mut tests = 1;
        if overlaps(bounds[0], other_bounds[0]) {
            stack.push([0, 0]);
        }
        while let Some([a, b]) = stack.pop() {
            let first = &self.nodes[a];
            let second = &other.nodes[b];
            let children = match (first.children(), second.children()) {
                (None, None) => {
                    output.push([first.item.min(second.item), first.item.max(second.item)]);
                    continue;
                }
                (Some([left, right]), _)
                    if second.children().is_none() || first.item >= second.item =>
                {
                    [[right, b], [left, b]]
                }
                (_, Some([left, right])) => [[a, right], [a, left]],
                _ => unreachable!("a non-leaf first node was handled above"),
            };
            for [a, b] in children {
                tests += 1;
                if overlaps(bounds[a], other_bounds[b]) {
                    stack.push([a, b]);
                }
            }
        }
        tests
    }
    pub(super) fn query(
        &self,
        bounds: &[Bounds],
        query: Bounds,
        stack: &mut Vec<usize>,
        output: &mut Vec<usize>,
    ) -> usize {
        let mut tests = 0;
        stack.clear();
        if !self.nodes.is_empty() {
            stack.push(0);
        }
        while let Some(index) = stack.pop() {
            tests += 1;
            if !overlaps(bounds[index], query) {
                continue;
            }
            let node = &self.nodes[index];
            if let Some([a, b]) = node.children() {
                stack.extend([b, a]);
            } else {
                output.push(node.item);
            }
        }
        tests
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every node pair popped by a traversal that stacks children untested, and
    // the leaf pairs it reports in order: the count and order `Tree::pairs`
    // keeps while testing children before stacking them.
    fn traverse(
        first: &Tree,
        bounds: &[Bounds],
        second: &Tree,
        other_bounds: &[Bounds],
    ) -> (usize, Vec<[usize; 2]>) {
        let mut stack = Vec::new();
        if !first.nodes.is_empty() && !second.nodes.is_empty() {
            stack.push([0, 0]);
        }
        let (mut tests, mut output) = (0, Vec::new());
        while let Some([a, b]) = stack.pop() {
            tests += 1;
            if !overlaps(bounds[a], other_bounds[b]) {
                continue;
            }
            let (one, two) = (&first.nodes[a], &second.nodes[b]);
            match (one.children(), two.children()) {
                (None, None) => output.push([one.item.min(two.item), one.item.max(two.item)]),
                (Some([left, right]), _) if two.children().is_none() || one.item >= two.item => {
                    stack.extend([[right, b], [left, b]]);
                }
                (_, Some([left, right])) => stack.extend([[a, right], [a, left]]),
                _ => unreachable!(),
            }
        }
        (tests, output)
    }

    fn visited(first: &Tree, a: &[Bounds], second: &Tree, b: &[Bounds]) -> usize {
        traverse(first, a, second, b).0
    }

    fn stack_order(first: &Tree, a: &[Bounds], second: &Tree, b: &[Bounds]) -> Vec<[usize; 2]> {
        traverse(first, a, second, b).1
    }

    #[test]
    fn paired_trees_match_exhaustive_overlaps_for_large_uneven_chassis() {
        let bounds = (0..2049)
            .map(|i| {
                let x = f64::from(i % 1024) * 0.25;
                let y = f64::from(i / 1024) * 0.2;
                [DVec3::new(x, y, 0.0), DVec3::new(x + 0.25, y + 0.25, 0.25)]
            })
            .collect::<Vec<_>>();
        let first = Tree::new((0..1024).collect(), &bounds);
        let mut stack = Vec::new();
        let mut pairs = Vec::new();
        for last in [1024, 1025, 2049] {
            let second = Tree::new((1024..last).collect(), &bounds);
            for travel in [0.0, 0.01, 0.8, 5.0] {
                let swept = bounds
                    .iter()
                    .map(|[lo, hi]| [*lo - DVec3::splat(travel), *hi + DVec3::splat(travel)])
                    .collect::<Vec<_>>();
                let mut a = Vec::new();
                let mut b = Vec::new();
                first.refit(&swept, &mut a);
                second.refit(&swept, &mut b);
                pairs.clear();
                let tests = first.pairs(&a, &second, &b, &mut stack, &mut pairs);
                assert_eq!(
                    (tests, &pairs),
                    (
                        visited(&first, &a, &second, &b),
                        &stack_order(&first, &a, &second, &b)
                    )
                );
                pairs.sort_unstable();
                let expected = (0..1024)
                    .flat_map(|a| {
                        let swept = &swept;
                        (1024..last)
                            .filter_map(move |b| overlaps(swept[a], swept[b]).then_some([a, b]))
                    })
                    .collect::<Vec<_>>();
                assert_eq!(pairs, expected);
            }
        }
    }
}
