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
    """Check the API health endpoint."""
    _emit(ctx, _client(ctx).health())


@cli.command()
@click.pass_context
@_handle_errors
def version(ctx: click.Context) -> None:
    """Show the API version."""
    _emit(ctx, _client(ctx).version())


# ── prediction ────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("smiles")
@click.pass_context
@_handle_errors
def predict(ctx: click.Context, smiles: str) -> None:
    """Predict properties for a SMILES."""
    _emit(ctx, _client(ctx).predict(smiles))


# ── cheminformatics ───────────────────────────────────────────────────────────


@cli.command()
@click.argument("smiles")
@click.pass_context
@_handle_errors
def scaffold(ctx: click.Context, smiles: str) -> None:
    """Bemis–Murcko scaffold + SAScore for a SMILES."""
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
@click.option("--max-tautomers", default=25, show_default=True)
@click.option("--max-stereoisomers", default=16, show_default=True)
@click.pass_context
@_handle_errors
def isomers(ctx: click.Context, smiles: str, max_tautomers: int, max_stereoisomers: int) -> None:
    """Enumerate tautomers + stereoisomers."""
    _emit(ctx, _client(ctx).isomers(smiles, max_tautomers, max_stereoisomers))


@cli.command()
@click.argument("smiles_a")
@click.argument("smiles_b")
@click.pass_context
@_handle_errors
def compare(ctx: click.Context, smiles_a: str, smiles_b: str) -> None:
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
