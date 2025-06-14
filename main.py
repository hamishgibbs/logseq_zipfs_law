#!/usr/bin/env python3
"""
Zipf analysis of Logseq page‑links (uses Click for CLI).

Example
-------
python logseq_zipf.py --root /path/to/graph --plot zipf.png
"""
import collections
import glob
import re
from pathlib import Path

import click
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ------------- configuration ------------------------------------------------
LINK_RE = re.compile(r"\[\[([^\[\]]+?)\]\]")  # matches [[Page Name]]
# ---------------------------------------------------------------------------


def find_markdown_files(root: Path):
    """Return list of .md files inside journals/ and pages/."""
    md_files = glob.glob(str(root / "journals" / "**" / "*.md"), recursive=True)
    md_files += glob.glob(str(root / "pages" / "**" / "*.md"), recursive=True)
    return md_files


def count_page_links(files):
    """Count occurrences of each [[Page]]."""
    counts = collections.Counter()
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            text = fh.read()
            for m in LINK_RE.finditer(text):
                counts[m.group(1).strip()] += 1
    return counts


def make_zipf_plot(counts, out_png="zipf_plot.png"):
    """Create Zipf scatterplot, fit power‑law, return exponent & R²."""
    df = (
        pd.DataFrame(counts.items(), columns=["page", "freq"])
        .sort_values("freq", ascending=False)
        .reset_index(drop=True)
    )
    df["rank"] = np.arange(1, len(df) + 1)

    # linear regression in log–log space
    slope, intercept, r, p, stderr = linregress(
        np.log10(df["rank"]), np.log10(df["freq"])
    )
    exponent = -slope

    # plotting
    sns.set_theme(style="whitegrid", context="talk")
    
    # Create the plot with the line first (to put it in the background)
    fig, ax = plt.subplots()
    x_fit = np.linspace(df["rank"].min(), df["rank"].max(), 200)
    y_fit = 10 ** (intercept + slope * np.log10(x_fit))
    line = plt.plot(x_fit, y_fit, color='black', linestyle='--', linewidth=0.8,
             label=r"$f(r) = {:.2f} \cdot r^{{{:.2f}}}$".format(10**intercept, -slope))
    
    # Add points with smaller size and specific color
    sns.scatterplot(data=df, x="rank", y="freq", s=9, linewidth=0, color="#00bbf9", ax=ax)
    
    # Create legend with only the equation text (no line)
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='', linestyle='none', 
                             label=r"$f(r) = {:.2f} \cdot r^{{{:.2f}}}$".format(10**intercept, -slope))]
    ax.legend(handles=legend_elements)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Frequency")
    ax.set_title("Zipf's plot of Logseq page links")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

    return exponent, r ** 2, out_png


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--root",
    type=click.Path(file_okay=False, dir_okay=True, exists=True, path_type=Path),
    required=True,
    help="Path to Logseq graph root containing journals/ and pages/",
)
@click.option(
    "--plot",
    "plot_file",
    default="zipf_plot.png",
    show_default=True,
    help="Filename for the output PNG plot",
)
def cli(root: Path, plot_file: str):
    """Count Logseq page‑links and make a Zipf plot."""
    md_files = find_markdown_files(root)
    if not md_files:
        click.echo("❌  No markdown files found – is the path correct?", err=True)
        raise SystemExit(1)

    counts = count_page_links(md_files)
    exponent, r2, plot_path = make_zipf_plot(counts, plot_file)

    click.echo(f"Estimated Zipf exponent : {exponent:.3f}")
    click.echo(f"R² of log‑log fit       : {r2:.3f}")
    click.echo(f"Plot saved to           : {plot_path}")


if __name__ == "__main__":
    cli()