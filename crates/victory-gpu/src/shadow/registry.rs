use std::collections::{BTreeMap, BTreeSet};

use super::Entry;

type Key = (u64, usize);

pub(super) struct Registry {
    pub(super) entries: BTreeMap<Key, Entry>,
    by_length: BTreeMap<u32, BTreeSet<Key>>,
}

impl Registry {
    pub(super) const fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            by_length: BTreeMap::new(),
        }
    }

    pub(super) fn values(&self) -> impl Iterator<Item = &Entry> {
        self.entries.values()
    }

    pub(super) fn insert(&mut self, key: Key, entry: Entry) {
        let class = (entry.end - key.0).checked_ilog2();
        let old = self.entries.insert(key, entry);
        if let Some(old) = old {
            let previous = (old.end - key.0).checked_ilog2();
            if previous == class {
                return;
            }
            self.remove_from_class(key, previous);
        }
        if let Some(class) = class {
            self.by_length.entry(class).or_default().insert(key);
        }
    }

    pub(super) fn remove(&mut self, key: &Key) {
        if let Some(old) = self.entries.remove(key) {
            self.remove_from_class(*key, (old.end - key.0).checked_ilog2());
        }
    }

    fn remove_from_class(&mut self, key: Key, class: Option<u32>) {
        if let Some(class) = class {
            let keys = self.by_length.get_mut(&class).unwrap();
            keys.remove(&key);
            if keys.is_empty() {
                self.by_length.remove(&class);
            }
        }
    }

    /// Prefer the latest start, then the greatest resource id at the same start.
    pub(super) fn containing(&self, address: u64, end: u64) -> Option<Key> {
        let (&nearest, entry) = self.entries.range(..=(address, usize::MAX)).next_back()?;
        if entry.end >= end {
            return Some(nearest);
        }

        // Earlier starts require at least this many bytes to reach the requested end.
        let minimum = (end - nearest.0).ilog2();
        let mut best = None;
        for (&class, keys) in self.by_length.range(minimum..) {
            // Class k holds lengths [2^k, 2^(k+1)-1]. Everything before
            // this lower bound is too short, including far-away giant buffers.
            let largest = u64::MAX >> (63 - class);
            let first = (end.saturating_sub(largest), 0).max(best.unwrap_or((0, 0)));
            if first <= nearest
                && let Some(&key) = keys
                    .range(first..=nearest)
                    .rev()
                    .find(|key| self.entries[key].end >= end)
            {
                best = Some(key);
            }
        }
        best
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn buffer(end: u64) -> Entry {
        Entry { end, shadow: None }
    }

    fn exhaustive(registry: &Registry, address: u64, end: u64) -> Option<Key> {
        registry
            .entries
            .range(..=(address, usize::MAX))
            .rev()
            .find(|(_, entry)| entry.end >= end)
            .map(|(&key, _)| key)
    }

    #[test]
    fn overlapping_registrations_follow_replacements_and_removals() {
        let mut r = Registry::new();
        for (key, end) in [
            ((100, 1), 640),
            ((100, 4), 140),
            ((150, 7), 210),
            ((200, 8), 225),
        ] {
            r.insert(key, buffer(end));
        }
        assert_eq!(r.containing(110, 140), Some((100, 4)));
        assert_eq!(r.containing(110, 141), Some((100, 1)));
        assert_eq!(r.containing(172, 210), Some((150, 7)));
        assert_eq!(r.containing(202, 225), Some((200, 8)));
        assert_eq!(r.containing(205, 226), Some((100, 1)));
        assert_eq!(r.containing(639, 640), Some((100, 1)));
        assert_eq!(r.containing(640, 641), None);
        r.insert((100, 1), buffer(130));
        assert_eq!(r.containing(205, 226), None);
        r.remove(&(200, 8));
        assert_eq!(r.containing(205, 210), Some((150, 7)));
        r.remove(&(150, 7));
        assert_eq!(r.containing(205, 210), None);
        r.remove(&(100, 4));
        assert_eq!(r.containing(110, 130), Some((100, 1)));
        r.remove(&(100, 1));
        assert_eq!(r.containing(110, 111), None);
        assert!(r.by_length.is_empty());
    }

    #[test]
    fn empty_buffers_and_maximum_exclusive_end() {
        let mut r = Registry::new();
        r.insert((0, 0), buffer(0));
        r.insert((u64::MAX - 127, 1), buffer(u64::MAX));
        r.insert((u64::MAX - 2, 2), buffer(u64::MAX - 1));
        assert_eq!(r.containing(0, 1), None);
        assert_eq!(
            r.containing(u64::MAX - 2, u64::MAX - 1),
            Some((u64::MAX - 2, 2))
        );
        assert_eq!(
            r.containing(u64::MAX - 2, u64::MAX),
            Some((u64::MAX - 127, 1))
        );
        assert_eq!(
            r.containing(u64::MAX - 1, u64::MAX),
            Some((u64::MAX - 127, 1))
        );
        r.insert((u64::MAX - 127, 1), buffer(u64::MAX - 127));
        assert_eq!(r.containing(u64::MAX - 1, u64::MAX), None);
        r.remove(&(0, 0));
        r.remove(&(u64::MAX - 2, 2));
        r.remove(&(u64::MAX - 127, 1));
        assert!(r.by_length.is_empty());
    }

    #[test]
    fn indexed_ranges_match_a_full_scan_through_churn() {
        let mut r = Registry::new();
        let mut value = 85682731u64;
        let mut next = || {
            value ^= value << 13;
            value ^= value >> 7;
            value ^= value << 17;
            value
        };
        for _ in 0..5000 {
            let start = (next() & 4095) + (0x7f_0000_u64 << 32);
            let key = (start, (next() & 3) as usize);
            if next().is_multiple_of(5) {
                r.remove(&key);
            } else {
                let end = start + (next() & 255);
                r.insert(key, buffer(end));
            }
            for _ in 0..4 {
                let a = (start & !4095) + (next() & 8191);
                let end = a + 1 + (next() & 511);
                assert_eq!(r.containing(a, end), exhaustive(&r, a, end));
            }
        }
    }
}
