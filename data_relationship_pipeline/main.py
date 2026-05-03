"""
╔══════════════════════════════════════════════════════════════╗
║         DATA RELATIONSHIP PIPELINE  v1.0                    ║
║         Built with First Principles Architecture             ║
║         Author: Professional Python Pipeline                 ║
╚══════════════════════════════════════════════════════════════╝

CLI Usage:
    python main.py --data path/to/file.csv
    python main.py --data https://example.com/data.csv
    python main.py --data data.csv --columns Age,Salary,Department
    python main.py --data data.csv --rows 0,5,10,15
    python main.py --data data.csv --output my_report.html
"""

import argparse
import sys
import os
from pathlib import Path

# ── Rich for beautiful CLI output ──────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.prompt import Prompt, Confirm
    from rich import print as rprint
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


def print_banner():
    """Print the startup banner."""
    if RICH_AVAILABLE:
        banner = """
[bold cyan]██████╗  █████╗ ████████╗ █████╗     ██████╗ ██╗██████╗ ███████╗██╗     ██╗███╗   ██╗███████╗[/bold cyan]
[bold cyan]██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗    ██╔══██╗██║██╔══██╗██╔════╝██║     ██║████╗  ██║██╔════╝[/bold cyan]
[bold cyan]██║  ██║███████║   ██║   ███████║    ██████╔╝██║██████╔╝█████╗  ██║     ██║██╔██╗ ██║█████╗  [/bold cyan]
[bold cyan]██║  ██║██╔══██║   ██║   ██╔══██║    ██╔═══╝ ██║██╔═══╝ ██╔══╝  ██║     ██║██║╚██╗██║██╔══╝  [/bold cyan]
[bold cyan]██████╔╝██║  ██║   ██║   ██║  ██║    ██║     ██║██║     ███████╗███████╗██║██║ ╚████║███████╗[/bold cyan]
[bold cyan]╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝    ╚═╝     ╚═╝╚═╝     ╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝[/bold cyan]
        """
        console.print(Panel(
            "[bold white]🔬 DATA RELATIONSHIP PIPELINE[/bold white]\n"
            "[dim]Discover hidden patterns · Statistical analysis · Visual reports[/dim]\n"
            "[cyan]v1.0 — Built with First Principles Architecture[/cyan]",
            border_style="cyan",
            padding=(1, 4)
        ))
    else:
        print("=" * 60)
        print("   DATA RELATIONSHIP PIPELINE v1.0")
        print("=" * 60)


def parse_args():
    """Parse command-line arguments with smart defaults."""
    parser = argparse.ArgumentParser(
        description="🔬 Data Relationship Pipeline — Find & Visualize Data Relationships",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python main.py --data sales.csv
  python main.py --data sales.csv --columns Age,Salary,Department
  python main.py --data https://raw.githubusercontent.com/.../iris.csv
  python main.py --data data.xlsx --output report.html --rows 0,10,50
        """
    )
    parser.add_argument(
        "--data", "-d",
        help="Path to data file (CSV, Excel, JSON) or URL",
        required=False
    )
    parser.add_argument(
        "--columns", "-c",
        help="Comma-separated column names to analyze (e.g. Age,Salary,Dept)",
        default=None
    )
    parser.add_argument(
        "--rows", "-r",
        help="Comma-separated row indices to filter (e.g. 0,5,10)",
        default=None
    )
    parser.add_argument(
        "--output", "-o",
        help="Output HTML report filename (default: report.html)",
        default="report.html"
    )
    parser.add_argument(
        "--sample", "-s",
        help="Max rows to sample for analysis (default: all)",
        type=int,
        default=None
    )
    parser.add_argument(
        "--no-interactive",
        help="Skip interactive column selection",
        action="store_true",
        default=False
    )
    return parser.parse_args()


def interactive_data_input():
    """Interactive CLI for data input when no --data flag is given."""
    if RICH_AVAILABLE:
        console.print("\n[bold yellow]📂 DATA SOURCE[/bold yellow]")
        console.print("[dim]Enter a file path (CSV/Excel/JSON) or a URL to a dataset[/dim]")
        data_path = Prompt.ask("[cyan]→ Data path or URL[/cyan]")
    else:
        print("\nEnter data path or URL:")
        data_path = input("→ ").strip()
    return data_path


def interactive_column_selection(df):
    """Let user select which columns to analyze interactively."""
    if RICH_AVAILABLE:
        # Show available columns in a table
        table = Table(title="📊 Available Columns", border_style="cyan")
        table.add_column("#", style="dim", width=4)
        table.add_column("Column Name", style="bold white")
        table.add_column("Data Type", style="cyan")
        table.add_column("Non-Null", style="green")
        table.add_column("Sample Values", style="dim")

        for i, col in enumerate(df.columns):
            dtype = str(df[col].dtype)
            non_null = f"{df[col].notna().sum()} / {len(df)}"
            sample = str(list(df[col].dropna().head(3).values)).replace("'", "")[:40]
            table.add_row(str(i), col, dtype, non_null, sample)

        console.print(table)
        console.print(
            "\n[dim]Enter column names separated by commas, "
            "or press Enter to analyze ALL columns[/dim]"
        )
        selection = Prompt.ask(
            "[cyan]→ Columns to analyze[/cyan]",
            default=""
        )
    else:
        print("\nAvailable columns:")
        for i, col in enumerate(df.columns):
            print(f"  [{i}] {col} ({df[col].dtype})")
        print("\nEnter column names (comma-separated) or press Enter for ALL:")
        selection = input("→ ").strip()

    if not selection:
        return list(df.columns)

    selected = [c.strip() for c in selection.split(",")]
    # Validate
    valid = [c for c in selected if c in df.columns]
    invalid = [c for c in selected if c not in df.columns]

    if invalid and RICH_AVAILABLE:
        console.print(f"[yellow]⚠ Columns not found (ignored): {invalid}[/yellow]")

    return valid if valid else list(df.columns)


def interactive_row_selection(df):
    """Let user optionally filter specific rows."""
    if RICH_AVAILABLE:
        console.print("\n[bold yellow]🔢 ROW SELECTION[/bold yellow]")
        console.print(f"[dim]Dataset has {len(df)} rows (indices 0–{len(df)-1})[/dim]")
        use_rows = Confirm.ask(
            "[cyan]→ Filter specific rows? (No = use all rows)[/cyan]",
            default=False
        )
        if not use_rows:
            return None
        row_input = Prompt.ask(
            "[cyan]→ Enter row indices (comma-separated, e.g. 0,5,10,50)[/cyan]"
        )
    else:
        print(f"\nDataset has {len(df)} rows. Filter specific rows? (y/N)")
        ans = input("→ ").strip().lower()
        if ans != "y":
            return None
        print("Enter row indices (comma-separated):")
        row_input = input("→ ").strip()

    try:
        indices = [int(x.strip()) for x in row_input.split(",")]
        valid_idx = [i for i in indices if 0 <= i < len(df)]
        return valid_idx
    except ValueError:
        return None


def run_pipeline(data_path, columns, row_indices, output_path, sample_size):
    """Execute the full analysis pipeline."""
    from loader import DataLoader
    from analyzer import RelationshipAnalyzer
    from reporter import HTMLReporter

    steps = [
        ("📂 Loading data", "load"),
        ("🔍 Detecting column types", "detect"),
        ("📊 Computing statistics", "stats"),
        ("🔗 Finding relationships", "relationships"),
        ("📈 Generating charts", "charts"),
        ("📝 Building HTML report", "report"),
    ]

    if RICH_AVAILABLE:
        console.print("\n[bold green]🚀 PIPELINE RUNNING[/bold green]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Initializing...", total=len(steps))

            # ── STEP 1: Load Data ─────────────────────────────
            progress.update(task, description="[cyan]📂 Loading data...")
            loader = DataLoader(data_path, sample_size=sample_size)
            df = loader.load()
            progress.advance(task)

            # ── STEP 2: Column selection ──────────────────────
            progress.update(task, description="[cyan]🔍 Detecting column types...")
            if columns:
                valid_cols = [c for c in columns if c in df.columns]
                if valid_cols:
                    df = df[valid_cols]
            if row_indices:
                df = df.iloc[row_indices]
            progress.advance(task)

            # ── STEP 3-5: Analysis ────────────────────────────
            progress.update(task, description="[cyan]📊 Computing statistics...")
            analyzer = RelationshipAnalyzer(df, loader.metadata)
            progress.advance(task)

            progress.update(task, description="[cyan]🔗 Finding relationships...")
            results = analyzer.analyze_all()
            progress.advance(task)

            progress.update(task, description="[cyan]📈 Generating charts...")
            charts = analyzer.generate_all_charts()
            progress.advance(task)

            # ── STEP 6: Report ────────────────────────────────
            progress.update(task, description="[cyan]📝 Building HTML report...")
            reporter = HTMLReporter(df, results, charts, loader.metadata)
            reporter.generate(output_path)
            progress.advance(task)

    else:
        # Non-rich fallback
        print("\n[1/6] Loading data...")
        loader = DataLoader(data_path, sample_size=sample_size)
        df = loader.load()

        print("[2/6] Processing columns...")
        if columns:
            valid_cols = [c for c in columns if c in df.columns]
            if valid_cols:
                df = df[valid_cols]
        if row_indices:
            df = df.iloc[row_indices]

        print("[3/6] Computing statistics...")
        analyzer = RelationshipAnalyzer(df, loader.metadata)

        print("[4/6] Finding relationships...")
        results = analyzer.analyze_all()

        print("[5/6] Generating charts...")
        charts = analyzer.generate_all_charts()

        print("[6/6] Building HTML report...")
        reporter = HTMLReporter(df, results, charts, loader.metadata)
        reporter.generate(output_path)

    return df, results


def main():
    """Main entry point."""
    print_banner()
    args = parse_args()

    # ── Interactive data input ──────────────────────────────────
    data_path = args.data
    if not data_path:
        data_path = interactive_data_input()

    if not data_path:
        if RICH_AVAILABLE:
            console.print("[red]✗ No data source provided. Exiting.[/red]")
        else:
            print("No data source provided.")
        sys.exit(1)

    # ── Load data first for interactive column selection ────────
    from loader import DataLoader

    if RICH_AVAILABLE:
        console.print(f"\n[dim]Loading: [cyan]{data_path}[/cyan][/dim]")

    try:
        loader = DataLoader(data_path, sample_size=args.sample)
        df_preview = loader.load()
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[bold red]✗ Failed to load data:[/bold red] {e}")
        else:
            print(f"Error loading data: {e}")
        sys.exit(1)

    if RICH_AVAILABLE:
        console.print(
            f"[green]✓ Data loaded:[/green] "
            f"[bold]{len(df_preview):,} rows × {len(df_preview.columns)} columns[/bold]"
        )

    # ── Column selection ────────────────────────────────────────
    columns = None
    if args.columns:
        columns = [c.strip() for c in args.columns.split(",")]
    elif not args.no_interactive:
        columns = interactive_column_selection(df_preview)
        if columns == list(df_preview.columns):
            columns = None  # All columns = no filter

    # ── Row selection ───────────────────────────────────────────
    row_indices = None
    if args.rows:
        try:
            row_indices = [int(x.strip()) for x in args.rows.split(",")]
        except ValueError:
            if RICH_AVAILABLE:
                console.print("[yellow]⚠ Invalid row indices, using all rows[/yellow]")
    elif not args.no_interactive:
        row_indices = interactive_row_selection(df_preview)

    # ── Run Pipeline ────────────────────────────────────────────
    try:
        df, results = run_pipeline(
            data_path=data_path,
            columns=columns,
            row_indices=row_indices,
            output_path=args.output,
            sample_size=args.sample
        )

        # ── Final Summary ───────────────────────────────────────
        output_abs = Path(args.output).resolve()
        if RICH_AVAILABLE:
            console.print("\n")
            console.print(Panel(
                f"[bold green]✅ ANALYSIS COMPLETE![/bold green]\n\n"
                f"[white]📊 Rows analyzed:[/white] [cyan]{len(df):,}[/cyan]\n"
                f"[white]📋 Columns analyzed:[/white] [cyan]{len(df.columns)}[/cyan]\n"
                f"[white]🔗 Relationships found:[/white] "
                f"[cyan]{results.get('relationship_count', 0)}[/cyan]\n\n"
                f"[white]📄 Report saved to:[/white]\n"
                f"[bold cyan]{output_abs}[/bold cyan]",
                title="[bold]Pipeline Complete[/bold]",
                border_style="green",
                padding=(1, 3)
            ))
        else:
            print(f"\n✅ Done! Report: {output_abs}")

    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"\n[bold red]✗ Pipeline Error:[/bold red] {e}")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        else:
            import traceback
            print(f"Pipeline Error: {e}")
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
