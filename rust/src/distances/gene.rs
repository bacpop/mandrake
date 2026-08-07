use hashbrown::HashMap;
use roaring::RoaringBitmap;

#[derive(Clone, Debug, Default)]
pub(crate) struct SampleBases {
    a: RoaringBitmap,
    c: RoaringBitmap,
    g: RoaringBitmap,
    t: RoaringBitmap,
    gap: RoaringBitmap,
}

impl SampleBases {
    // Encodes a sequence into per-base bitmap positions.
    // Offset provides position of gene in whole pangenome
    pub(crate) fn from_sequence_at(sequence: &[u8], offset: u32) -> Self {
        let mut bases = SampleBases::default();

        for (position, &base) in sequence.iter().enumerate() {
            bases.insert_base(position as u32 + offset, base);
        }

        bases
    }

    #[cfg(test)]
    pub(crate) fn from_sequence(sequence: &[u8]) -> Self {
        Self::from_sequence_at(sequence, 0)
    }

    // Combines two bitmaps that have been offset
    pub(crate) fn union_assign(&mut self, other: SampleBases) {
        self.a |= &other.a;
        self.c |= &other.c;
        self.g |= &other.g;
        self.t |= &other.t;
        self.gap |= &other.gap;
    }

    pub(crate) fn matching_sites(&self, other: &Self) -> RoaringBitmap {
        let mut matches = &self.a & &other.a;
        matches |= &self.c & &other.c;
        matches |= &self.g & &other.g;
        matches |= &self.t & &other.t;
        matches
    }

    pub(crate) fn both_gap_sites(&self, other: &Self) -> RoaringBitmap {
        &self.gap & &other.gap
    }

    pub(crate) fn either_gap_sites(&self, other: &Self) -> RoaringBitmap {
        &self.gap | &other.gap
    }

    // Records one alignment position under its matching base set.
    fn insert_base(&mut self, position: u32, base: u8) {
        match base.to_ascii_uppercase() {
            b'A' => {
                self.a.insert(position);
            }
            b'C' => {
                self.c.insert(position);
            }
            b'G' => {
                self.g.insert(position);
            }
            b'T' => {
                self.t.insert(position);
            }
            b'M' => {
                self.a.insert(position);
                self.c.insert(position);
            }
            b'R' => {
                self.a.insert(position);
                self.g.insert(position);
            }
            b'W' => {
                self.a.insert(position);
                self.t.insert(position);
            }
            b'S' => {
                self.c.insert(position);
                self.g.insert(position);
            }
            b'Y' => {
                self.c.insert(position);
                self.t.insert(position);
            }
            b'K' => {
                self.g.insert(position);
                self.t.insert(position);
            }
            b'V' => {
                self.a.insert(position);
                self.c.insert(position);
                self.g.insert(position);
            }
            b'H' => {
                self.a.insert(position);
                self.c.insert(position);
                self.t.insert(position);
            }
            b'D' => {
                self.a.insert(position);
                self.g.insert(position);
                self.t.insert(position);
            }
            b'B' => {
                self.c.insert(position);
                self.g.insert(position);
                self.t.insert(position);
            }
            b'-' => {
                self.gap.insert(position);
            }
            _ => {
                // includes N
                self.a.insert(position);
                self.c.insert(position);
                self.g.insert(position);
                self.t.insert(position);
            }
        }
    }

    // Counts positions encoded as any non-gap base.
    pub(crate) fn non_gap_count(&self, alignment_len: usize) -> usize {
        alignment_len - self.gap.len() as usize
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct GeneMetadata {
    name: String,
    paralog_count: usize,
}

impl GeneMetadata {
    pub(crate) fn new(name: String, paralog_count: usize) -> Self {
        GeneMetadata {
            name,
            paralog_count,
        }
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn paralog_count(&self) -> Option<usize> {
        if self.paralog_count == 0 {
            None
        } else {
            Some(self.paralog_count)
        }
    }
}

#[derive(Debug)]
pub(crate) struct ParsedGeneAlignment {
    pub(crate) gene_index: usize,
    pub(crate) metadata: GeneMetadata,
    pub(crate) alignment_len: usize,
    pub(crate) offset: u32,
    pub(crate) sequences: HashMap<usize, SampleBases>,
}

impl ParsedGeneAlignment {
    pub(crate) fn new(
        gene_index: usize,
        metadata: GeneMetadata,
        alignment_len: usize,
        offset: u32,
        sequences: HashMap<usize, SampleBases>,
    ) -> Self {
        ParsedGeneAlignment {
            gene_index,
            metadata,
            alignment_len,
            offset,
            sequences,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snp_count(sequence_a: &[u8], sequence_b: &[u8], gaps: bool) -> (usize, usize) {
        let left = SampleBases::from_sequence(sequence_a);
        let right = SampleBases::from_sequence(sequence_b);
        let matches = left.matching_sites(&right);
        let both_gap = left.both_gap_sites(&right);
        let length = if gaps {
            sequence_a.len() - both_gap.len() as usize
        } else {
            sequence_a.len() - left.gap.union_len(&right.gap) as usize
        };

        (length - matches.len() as usize, length)
    }

    fn one_base_snp_count(left: u8, right: u8) -> (usize, usize) {
        sample_snp_count(&[left], &[right], false)
    }

    // Asserts bitmap membership for one encoded base.
    fn assert_membership(
        base: u8,
        expected_a: bool,
        expected_c: bool,
        expected_g: bool,
        expected_t: bool,
        expected_gap: bool,
    ) {
        let bases = SampleBases::from_sequence(&[base]);

        assert_eq!(bases.a.contains(0), expected_a, "{} A", base as char);
        assert_eq!(bases.c.contains(0), expected_c, "{} C", base as char);
        assert_eq!(bases.g.contains(0), expected_g, "{} G", base as char);
        assert_eq!(bases.t.contains(0), expected_t, "{} T", base as char);
        assert_eq!(bases.gap.contains(0), expected_gap, "{} gap", base as char);
    }

    #[test]
    fn offset_sequence_encoding_shifts_bitmap_positions() {
        let bases = SampleBases::from_sequence_at(b"AC-G", 7);

        assert!(bases.a.contains(7));
        assert!(bases.c.contains(8));
        assert!(bases.gap.contains(9));
        assert!(bases.g.contains(10));
        assert!(!bases.a.contains(0));
    }

    #[test]
    // Verifies exact DNA bases only match identical bases.
    fn exact_bases_match_only_themselves() {
        assert_eq!(sample_snp_count(b"ACGT", b"ACGT", false), (0, 4));
        assert_eq!(sample_snp_count(b"ACGT", b"TGCA", false), (4, 4));
    }

    #[test]
    // Verifies IUPAC ambiguity codes map to the expected base sets.
    fn iupac_bases_encode_expected_memberships() {
        let expectations = [
            (b'A', true, false, false, false, false),
            (b'C', false, true, false, false, false),
            (b'G', false, false, true, false, false),
            (b'T', false, false, false, true, false),
            (b'M', true, true, false, false, false),
            (b'R', true, false, true, false, false),
            (b'W', true, false, false, true, false),
            (b'S', false, true, true, false, false),
            (b'Y', false, true, false, true, false),
            (b'K', false, false, true, true, false),
            (b'V', true, true, true, false, false),
            (b'H', true, true, false, true, false),
            (b'D', true, false, true, true, false),
            (b'B', false, true, true, true, false),
            (b'N', true, true, true, true, false),
        ];

        for (base, expected_a, expected_c, expected_g, expected_t, expected_gap) in expectations {
            assert_membership(
                base,
                expected_a,
                expected_c,
                expected_g,
                expected_t,
                expected_gap,
            );
        }
    }

    #[test]
    // Verifies ambiguous bases match when their base sets overlap.
    fn iupac_bases_match_when_any_membership_overlaps() {
        assert_eq!(one_base_snp_count(b'R', b'A'), (0, 1));
        assert_eq!(one_base_snp_count(b'R', b'G'), (0, 1));
        assert_eq!(one_base_snp_count(b'R', b'C'), (1, 1));
        assert_eq!(one_base_snp_count(b'B', b'A'), (1, 1));
        assert_eq!(one_base_snp_count(b'B', b'T'), (0, 1));
        assert_eq!(
            sample_snp_count(b"MRWSYKVHDB", b"AATCTGCATG", false),
            (0, 10)
        );
    }

    #[test]
    // Verifies N and unknown non-gap bases match ordinary bases.
    fn n_and_unknown_non_gap_characters_match_any_ordinary_base() {
        assert_eq!(sample_snp_count(b"NX?z", b"ACGT", false), (0, 4));
        assert_eq!(one_base_snp_count(b'n', b't'), (0, 1));
        assert_eq!(one_base_snp_count(b'?', b'A'), (0, 1));
        assert_eq!(one_base_snp_count(b'X', b'C'), (0, 1));
    }

    #[test]
    // Verifies gaps are ignored by default, and counted only in gap-inclusive mode.
    fn gap_handling_depends_on_gap_mode() {
        assert_membership(b'-', false, false, false, false, true);

        assert_eq!(
            sample_snp_count(b"A--CGGTTT-", b"ACCCTG----", false),
            (1, 4)
        );
        assert_eq!(sample_snp_count(b"A--CGGTTT-", b"ACCCTG----", true), (6, 9));
    }

    #[test]
    fn union_assign_merges_base_bitmaps() {
        let mut bases = SampleBases::from_sequence_at(b"A-", 0);
        bases.union_assign(SampleBases::from_sequence_at(b"CT", 2));

        assert!(bases.a.contains(0));
        assert!(bases.gap.contains(1));
        assert!(bases.c.contains(2));
        assert!(bases.t.contains(3));
    }
}
