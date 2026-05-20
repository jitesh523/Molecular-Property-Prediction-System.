# molprop Examples

Worked examples demonstrating the molprop Python SDK and CLI.

## Prerequisites

Start a local server (the examples assume `http://localhost:8000`):

```bash
uvicorn molprop.serving.api:app --reload
```

## Scripts

### `01_sdk_walkthrough.py`

End-to-end Python tour exercising every major cheminformatics endpoint:
health/version, scaffold + SAScore, functional groups, compare + MCS,
R-group decomposition, reaction SMARTS, structural alerts, compound library
CRUD, and the aggregated Markdown report.

```bash
python examples/01_sdk_walkthrough.py
```

## CLI mini-recipes

```bash
# A) Quick PAINS check on a screening list
while read smi; do
  echo -n "$smi → "
  molprop --compact alerts "$smi" | jq -r '.is_clean'
done < hits.smi

# B) Substructure search across a saved project library
molprop --compact substructure "c1ccc(N)cc1" --project drug-x

# C) Generate a Markdown report per molecule in a series
for s in $(cat series.smi); do
  molprop report "$s" -o "reports/${s//[^a-zA-Z0-9]/_}.md"
done

# D) Apply a built-in named reaction to substrate pairs (TAB-separated)
echo -e "CC(=O)O\tNCC\nc1ccccc1C(=O)O\tNCC" > substrates.tsv
molprop react --named amide_coupling substrates.tsv

# E) Browse the built-in named-reaction catalog
molprop react-list
```
