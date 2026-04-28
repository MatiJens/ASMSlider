import argparse
import csv
import json
from collections import defaultdict

ASM_PATTERNS = ("HET-s", "sigma", "inhouse006", "PUASM", "HET-S", "NACHT_sigma", "RHIM", "PF17046")


def _matches_asm(value):
    return any(p in value for p in ASM_PATTERNS)


def count_asm_types(input_csv, output_json):
    results = defaultdict(lambda: defaultdict(list))

    with open(input_csv) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            accession = row[0]
            gca = row[1]
            pfam_name = row[7]
            pfam_id = row[6]
            start = row[3]
            end = row[4]

            if _matches_asm(pfam_name) or _matches_asm(pfam_id):
                results[gca][accession].append({
                    "type": pfam_name,
                    "start": int(start),
                    "end": int(end),
                })

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    total_hits = sum(len(h) for accs in results.values() for h in accs.values())
    total_proteins = sum(len(accs) for accs in results.values())
    print(f"Found {total_hits} ASM hits across {total_proteins} proteins in {len(results)} assemblies.")
    print(f"Saved to {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Count ASM-related domain hits per assembly and protein.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    count_asm_types(args.input, args.output)


if __name__ == "__main__":
    main()
