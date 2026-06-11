# Scan-result evaluation and plotting

Tooling to evaluate and visualise ASM motif predictions produced by a per-threshold
proteome scan, comparing them against curated ASM references and PFAM domain references.

## Proteome directory layout

Each tool operates on a *proteome directory* with the following contents:

```
<proteome>/
  <name>.fasta              proteome sequences (used for protein lengths)
  asm_reference.tsv         true ASM motifs: seq_id, asm_beg, asm_end (positions optional)
  pfam_references.tsv       PFAM domains:   seq_id, pfam_beg, pfam_end, pfam_acc, pfam_name
  scan_results/
    <threshold>/*.json      predicted motifs at each probability threshold (e.g. 070, ..., 095)
```

A prediction JSON is a list of objects with `protein`, `location` (`"beg-end"`) and
`probability` fields.

In `asm_reference.tsv`, a row whose begin/end is blank or one of `NA`/`None`/`-` is
treated as "present but unpositioned": the protein is known to carry an ASM, but the
exact coordinates are unknown.

## Reference scope

Only a fixed allow-list of PFAM accessions is treated as a reference (see
`scan_common.ALLOWED_PFAM`): NLR domains, where the ASM is expected in the N-terminus,
and effector domains, where it is expected in the C-terminus. All other PFAM accessions
in `pfam_references.tsv` are ignored.

## Tools

### `evaluate_scan_results.py`

Prints a per-threshold table and writes `scan_evaluation.tsv` into the proteome directory.

```
python3 scripts/tools/evaluate_scan_results.py <proteome_dir>
```

Columns:

| column | meaning |
| --- | --- |
| `threshold` | scan threshold subfolder name |
| `total_found` | number of predicted motifs |
| `n_term_hits` | predictions starting within `TERM_MARGIN` (50 aa) of the N-terminus |
| `c_term_hits` | predictions ending within `TERM_MARGIN` of the C-terminus |
| `asm_refs_total` | total ASM reference motifs |
| `found_in_asm_refs_>=45%` | ASM references covered by a prediction (>=45% of the region, or any hit when unpositioned) |
| `pfam_refs_same_protein` | PFAM references whose protein contains at least one prediction |
| `sensitivity` | `found_in_asm_refs_>=45%` / `asm_refs_total` |

Note: `pfam_refs_same_protein` counts a PFAM reference as matched whenever a prediction
falls anywhere in the same protein, regardless of distance to the domain.

### `plot_scan_results.py`

Draws one figure per threshold under `scan_results/`, showing every protein that has at
least one prediction. Each protein row shows the protein length bar, its PFAM domains and
true ASM motifs (filled boxes), and the predictions (dashed boxes) coloured as:

- green  -- prediction covers a true ASM (>=45% of the region)
- blue   -- prediction is in a protein that carries a PFAM reference
- red    -- neither of the above

```
# all thresholds, output into <proteome_dir>/scan_plots/
python3 scripts/tools/plot_scan_results.py <proteome_dir>

# a single threshold into a custom directory
python3 scripts/tools/plot_scan_results.py <proteome_dir> \
    --only-thresholds 095 --out-dir results/pfam_fix/plots/<proteome>
```

Options: `--out-dir`, `--page-size` (proteins per figure; extra proteins paginate into
`_p02`, `_p03`, ...), `--only-thresholds`.

### `scan_common.py`

Shared loaders and constants imported by both tools: `ALLOWED_PFAM`, `load_asm_refs`,
`load_pfam_refs`, `load_protein_lengths`, `load_scan`, `overlap_len`,
`find_proteome_fasta`. Not run directly.

### `interproscan_to_pfam_tsv.py`

Converts an InterProScan TSV output into the `pfam_references.tsv` format expected above.

## Batch example

Evaluate every proteome under `proteomes/` and collect the tables in one place:

```
mkdir -p results/pfam_fix
for d in proteomes/*/; do
  name=$(basename "$d")
  [ -d "$d/scan_results" ] || continue
  python3 scripts/tools/evaluate_scan_results.py "$d" > "results/pfam_fix/${name}_evaluation.txt"
done
```
