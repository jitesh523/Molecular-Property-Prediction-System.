"""
molprop CLI — thin command-line wrapper over `MolpropClient`.

Examples:

    molprop health
    molprop predict "CC(=O)Oc1ccccc1C(=O)O"
    molprop scaffold "CC(=O)NC1=CC=C(O)C=C1"
    molprop compare "CCO" "CCN"
    molprop library list --project drug-x
    molprop library save "CCO" --name Ethanol --tag solvent --tag flammable
    molprop substructure "c1ccccc1" --project drug-x
    molprop report "CC(=O)NC1=CC=C(O)C=C1" -o aspirin.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click

from molprop.client import MolpropAPIError, MolpropClient

DEFAULT_URL = "http://localhost:8000"


def _client(ctx: click.Context) -> MolpropClient:
    return MolpropClient(ctx.obj["url"], timeout=ctx.obj["timeout"])


def _emit(ctx: click.Context, data: object) -> None:
    if ctx.obj["pretty"]:
        click.echo(json.dumps(data, indent=2, default=str))
    else:
        click.echo(json.dumps(data, default=str))


def _handle_errors(fn):
    """Decorator to convert MolpropAPIError into a clean CLI exit."""

    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except MolpropAPIError as e:
            click.secho(f"API error ({e.status_code}): {e.detail}", fg="red", err=True)
            sys.exit(1)
        except Exception as e:  # noqa: BLE001
            click.secho(f"Error: {e}", fg="red", err=True)
            sys.exit(2)

    wrapper.__name__ = fn.__name__
    return wrapper


# ── root group ────────────────────────────────────────────────────────────────


@click.group()
@click.option(
    "--url",
    default=DEFAULT_URL,
    envvar="MOLPROP_URL",
    show_default=True,
    help="Base URL of the molprop API.",
)
@click.option(
    "--timeout", default=60.0, type=float, show_default=True, help="HTTP timeout in seconds."
)
@click.option("--pretty/--compact", default=True, help="Pretty-print JSON output.")
@click.pass_context
def cli(ctx: click.Context, url: str, timeout: float, pretty: bool) -> None:
    """molprop command-line interface."""
    ctx.ensure_object(dict)
    ctx.obj["url"] = url
    ctx.obj["timeout"] = timeout
    ctx.obj["pretty"] = pretty


# ── meta ──────────────────────────────────────────────────────────────────────


@cli.command()
@click.pass_context
@_handle_errors
def health(ctx: click.Context) -> None:
    """Check if the API is running and models are loaded.
    
    Returns the API health status including whether the GNN model and VAE are available.
    """
    _emit(ctx, _client(ctx).health())


@cli.command()
@click.pass_context
@_handle_errors
def version(ctx: click.Context) -> None:
    """Show API and dependency versions.
    
    Returns molprop version, API version, PyTorch version, and model/VAE availability.
    """
    _emit(ctx, _client(ctx).version())


# ── prediction ────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("smiles")
@click.option(
    "--explain",
    is_flag=True,
    help="Generate atom-level explanation scores (GNNExplainer) and feature importance."
)
@click.option(
    "--uncertainty",
    type=int,
    default=0,
    show_default=True,
    help="Number of MC Dropout samples for uncertainty estimation (0-50)."
)
@click.pass_context
@_handle_errors
def predict(ctx: click.Context, smiles: str, explain: bool, uncertainty: int) -> None:
    """Predict molecular properties from a SMILES string.
    
    Standardizes input SMILES, generates graph representation, and runs GNN inference.
    Optionally generates explanations and Bayesian uncertainty estimates.
    
    Example:
        molprop predict "CC(=O)Oc1ccccc1C(=O)O" --explain --uncertainty 10
    """
    _emit(ctx, _client(ctx).predict(smiles))


# ── cheminformatics ───────────────────────────────────────────────────────────


@cli.command()
@click.argument("smiles")
@click.pass_context
@_handle_errors
def scaffold(ctx: click.Context, smiles: str) -> None:
    """Analyze Bemis–Murcko scaffold and compute SAScore.
    
    Returns the molecular scaffold (side chains removed), ring information,
    and synthetic accessibility score (1=easy, 10=hard).
    
    Example:
        molprop scaffold "CC(=O)NC1=CC=C(O)C=C1"
    """
    _emit(ctx, _client(ctx).scaffold(smiles))


@cli.command(name="fg")
@click.argument("smiles")
@click.pass_context
@_handle_errors
def functional_groups(ctx: click.Context, smiles: str) -> None:
    """Detect functional groups in a SMILES."""
    _emit(ctx, _client(ctx).functional_groups(smiles))


@cli.command()
@click.argument("smiles")
@click.option(
    "--max-tautomers",
    default=25,
    show_default=True,
    type=int,
    help="Maximum number of tautomers to enumerate (prevents combinatorial explosion)."
)
@click.option(
    "--max-stereoisomers",
    default=16,
    show_default=True,
    type=int,
    help="Maximum number of stereoisomers to enumerate."
)
@click.pass_context
@_handle_errors
def isomers(ctx: click.Context, smiles: str, max_tautomers: int, max_stereoisomers: int) -> None:
    """Enumerate tautomers and stereoisomers for a molecule.
    
    Returns canonical tautomer, all tautomer variants, and stereoisomer variants.
    Limits prevent combinatorial explosion on heavily substituted molecules.
    
    Example:
        molprop isomers "CCO" --max-tautomers 25 --max-stereoisomers 16
    """
    _emit(ctx, _client(ctx).isomers(smiles, max_tautomers, max_stereoisomers))


@cli.command()
@click.argument("smiles_a")
@click.argument("smiles_b")
@click.option(
    "--metric",
    type=click.Choice(["tanimoto", "cosine", "euclidean"]),
    default="tanimoto",
    show_default=True,
    help="Similarity metric to use."
)
@click.pass_context
@_handle_errors
def compare(ctx: click.Context, smiles_a: str, smiles_b: str, metric: str) -> None:
    """Compare two molecules using structural similarity.
    
    Computes MACCS fingerprint similarity between two SMILES strings.
    Supports multiple distance metrics (Tanimoto, Cosine, Euclidean).
    
    Example:
        molprop compare "CCO" "CCN" --metric tanimoto
    """
    """Compare two molecules side-by-side."""
    _emit(ctx, _client(ctx).compare(smiles_a, smiles_b))


@cli.command()
@click.argument("smiles_a")
@click.argument("smiles_b")
@click.pass_context
@_handle_errors
def mcs(ctx: click.Context, smiles_a: str, smiles_b: str) -> None:
    """Maximum common substructure between two molecules."""
    _emit(ctx, _client(ctx).mcs(smiles_a, smiles_b))


@cli.command()
@click.argument("core")
@click.argument("smiles_file", type=click.Path(exists=True))
@click.pass_context
@_handle_errors
def rgroups(ctx: click.Context, core: str, smiles_file: str) -> None:
    """R-group decomposition. Reads one SMILES per line from SMILES_FILE."""
    smiles_list = [
        line.strip() for line in Path(smiles_file).read_text().splitlines() if line.strip()
    ]
    _emit(ctx, _client(ctx).rgroups(core, smiles_list))


@cli.command()
@click.option("--smarts", default=None, help="Reaction SMARTS string.")
@click.option("--named", default=None, help="Named reaction key (see `molprop react-list`).")
@click.argument("substrates_file", type=click.Path(exists=True))
@click.pass_context
@_handle_errors
def react(
    ctx: click.Context, smarts: Optional[str], named: Optional[str], substrates_file: str
) -> None:
    """
    Apply a reaction SMARTS to substrate tuples.

    SUBSTRATES_FILE has one tuple per line; reactants in a tuple are
    separated by a TAB or a single space. Example for amide coupling::

        CC(=O)O\tNCC
        CCCC(=O)O\tNc1ccccc1
    """
    rows: list[list[str]] = []
    for line in Path(substrates_file).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p for p in line.replace("\t", " ").split(" ") if p]
        rows.append(parts)
    _emit(ctx, _client(ctx).react(rows, smarts=smarts, named=named))


@cli.command(name="react-list")
@click.pass_context
@_handle_errors
def react_list(ctx: click.Context) -> None:
    """List the built-in named reactions."""
    _emit(ctx, _client(ctx).react_named())


@cli.command(name="freewilson")
@click.option("--core", required=True, help="Core scaffold SMARTS or SMILES.")
@click.argument("data_file", type=click.Path(exists=True))
@click.option(
    "--min-occurrences",
    default=1,
    show_default=True,
    help="Drop R-group occupants seen in fewer than N analogues.",
)
@click.pass_context
@_handle_errors
def freewilson(ctx: click.Context, core: str, data_file: str, min_occurrences: int) -> None:
    """
    Free-Wilson additive R-group SAR analysis.

    DATA_FILE is a TAB- or comma-separated file with two columns:
    SMILES and the measured activity (e.g. pIC50). Header optional.
    """
    smiles_list: list[str] = []
    activities: list[float] = []
    for line in Path(data_file).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # split on tab or comma; take first two non-empty parts
        parts = [p for p in line.replace("\t", ",").split(",") if p.strip()]
        if len(parts) < 2:
            continue
        try:
            activities.append(float(parts[1]))
        except ValueError:
            # likely a header row — skip
            continue
        smiles_list.append(parts[0].strip())
    if not smiles_list:
        raise click.ClickException("No (SMILES, activity) rows found in data file.")
    _emit(
        ctx,
        _client(ctx).free_wilson(core, smiles_list, activities, min_occurrences=min_occurrences),
    )


@cli.command()
@click.argument("smiles_file", type=click.Path(exists=True))
@click.option(
    "--max-sub-atoms",
    "max_substituent_atoms",
    default=10,
    show_default=True,
    help="Drop cuts whose substituent has more heavy atoms than this.",
)
@click.option("--max-pairs", default=500, show_default=True)
@click.pass_context
@_handle_errors
def mmp(ctx: click.Context, smiles_file: str, max_substituent_atoms: int, max_pairs: int) -> None:
    """Single-cut Matched Molecular Pairs analysis. One SMILES per line."""
    smiles_list = [
        line.strip() for line in Path(smiles_file).read_text().splitlines() if line.strip()
    ]
    _emit(
        ctx,
        _client(ctx).mmp(
            smiles_list,
            max_substituent_atoms=max_substituent_atoms,
            max_pairs=max_pairs,
        ),
    )


@cli.command()
@click.argument("smiles")
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Write the SVG to this file instead of stdout.",
)
@click.option("--width", default=400, show_default=True)
@click.option("--height", default=400, show_default=True)
@click.option(
    "--highlight-smarts",
    default=None,
    help="SMARTS pattern; every matching atom will be highlighted.",
)
@click.pass_context
@_handle_errors
def depict(
    ctx: click.Context,
    smiles: str,
    output: Optional[str],
    width: int,
    height: int,
    highlight_smarts: Optional[str],
) -> None:
    """Render a 2D SVG depiction of a molecule."""
    data = _client(ctx).depict(
        smiles, width=width, height=height, highlight_smarts=highlight_smarts
    )
    if output:
        Path(output).write_text(data["svg"])
        click.secho(f"Wrote SVG to {output}", fg="green")
    else:
        click.echo(data["svg"])


@cli.command()
@click.argument("smiles")
@click.option(
    "--catalog",
    "catalogs",
    multiple=True,
    help="Catalog(s) to apply: PAINS, BRENK, NIH, ZINC (repeatable).",
)
@click.pass_context
@_handle_errors
def alerts(ctx: click.Context, smiles: str, catalogs) -> None:
    """Run structural alerts (PAINS, Brenk, NIH, ZINC)."""
    _emit(ctx, _client(ctx).alerts(smiles, list(catalogs) or None))


@cli.command()
@click.argument("smiles")
@click.pass_context
@_handle_errors
def standardize(ctx: click.Context, smiles: str) -> None:
    """Run standardisation with a detailed report."""
    _emit(ctx, _client(ctx).standardize(smiles))


@cli.command()
@click.argument("smiles")
@click.pass_context
@_handle_errors
def admet(ctx: click.Context, smiles: str) -> None:
    """ADMET prediction for a SMILES."""
    _emit(ctx, _client(ctx).admet(smiles))


@cli.command()
@click.argument("query")
@click.option(
    "--candidates",
    "candidates_file",
    type=click.Path(exists=True),
    help="Path to a text file with one candidate SMILES per line.",
)
@click.option("--project", default=None, help="Restrict library search to project.")
@click.option("--limit", default=100, show_default=True)
@click.pass_context
@_handle_errors
def substructure(
    ctx: click.Context,
    query: str,
    candidates_file: Optional[str],
    project: Optional[str],
    limit: int,
) -> None:
    """Substructure search (SMARTS / SMILES) over candidates or the library."""
    cands: Optional[list[str]] = None
    if candidates_file:
        cands = [
            line.strip() for line in Path(candidates_file).read_text().splitlines() if line.strip()
        ]
    _emit(ctx, _client(ctx).substructure(query, candidates=cands, project=project, limit=limit))


# ── report ────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("smiles")
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Write the markdown report to this file.",
)
@click.option("--no-admet", is_flag=True, default=False)
@click.option("--no-scaffold", is_flag=True, default=False)
@click.option("--no-fg", is_flag=True, default=False)
@click.option("--no-descriptors", is_flag=True, default=False)
@click.pass_context
@_handle_errors
def report(
    ctx: click.Context,
    smiles: str,
    output: Optional[str],
    no_admet: bool,
    no_scaffold: bool,
    no_fg: bool,
    no_descriptors: bool,
) -> None:
    """Generate an aggregated Markdown report for a molecule."""
    data = _client(ctx).report(
        smiles,
        include_admet=not no_admet,
        include_scaffold=not no_scaffold,
        include_functional_groups=not no_fg,
        include_descriptors=not no_descriptors,
    )
    md = data["markdown"]
    if output:
        Path(output).write_text(md)
        click.secho(f"Wrote report to {output}", fg="green")
    else:
        click.echo(md)


# ── library subcommands ───────────────────────────────────────────────────────


@cli.group()
def library() -> None:
    """Manage the persistent compound library."""


@library.command(name="save")
@click.argument("smiles")
@click.option("--name", default=None)
@click.option("--project", default="default", show_default=True)
@click.option("--tag", "tags", multiple=True, help="Add a tag (repeatable).")
@click.option("--notes", default=None)
@click.pass_context
@_handle_errors
def library_save(ctx: click.Context, smiles: str, name, project, tags, notes) -> None:
    """Save / upsert a compound."""
    _emit(
        ctx,
        _client(ctx).library_save(smiles, name=name, project=project, tags=list(tags), notes=notes),
    )


@library.command(name="list")
@click.option("--project", default=None)
@click.option("--tag", default=None)
@click.option("--search", default=None)
@click.option("--limit", default=100, show_default=True)
@click.pass_context
@_handle_errors
def library_list(ctx, project, tag, search, limit) -> None:
    """List compounds (with filters)."""
    _emit(ctx, _client(ctx).library_list(project=project, tag=tag, search=search, limit=limit))


@library.command(name="get")
@click.argument("compound_id", type=int)
@click.pass_context
@_handle_errors
def library_get(ctx, compound_id: int) -> None:
    """Fetch a single compound by id."""
    _emit(ctx, _client(ctx).library_get(compound_id))


@library.command(name="delete")
@click.argument("compound_id", type=int)
@click.confirmation_option(prompt="Really delete this compound?")
@click.pass_context
@_handle_errors
def library_delete(ctx, compound_id: int) -> None:
    """Delete a compound."""
    _emit(ctx, _client(ctx).library_delete(compound_id))


@library.command(name="projects")
@click.pass_context
@_handle_errors
def library_projects(ctx) -> None:
    """List all projects + tags in the library."""
    _emit(ctx, _client(ctx).library_projects())


def main() -> None:  # entry-point for `python -m molprop.cli`
    cli(obj={})


if __name__ == "__main__":
    main()
