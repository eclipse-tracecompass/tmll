from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.box import ROUNDED
from typing import Dict, List, Any, Optional


class DocumentGenerator:
    """
    A utility class for creating formatted text outputs. 
    This class uses the Rich library to create visually appealing text outputs.
    """
    console = Console(force_jupyter=True)
    WIDTH = 80
    PADDING = (1, 1)

    @staticmethod
    def section(title: str) -> None:
        """
        Display a section title with a separator. For example, "Model Evaluation".

        :param title: The title of the section.
        :type title: str
        """
        DocumentGenerator.console.print(Panel(
            Text(title, style="bold", justify="center"),
            box=ROUNDED,
            padding=DocumentGenerator.PADDING,
            width=DocumentGenerator.WIDTH
        ))

    @staticmethod
    def metric(label: str, value: Any) -> None:
        """
        Display a single metric. For example, accuracy: 0.95.

        :param label: The label of the metric.
        :type label: str
        :param value: The value of the metric.
        :type value: Any
        """
        DocumentGenerator.console.print(f"{label}: [bold]{value}[/bold]")

    @staticmethod
    def metrics_group(title: str, metrics: Dict[str, Any]) -> None:
        """Display a group of related metrics."""
        max_label_width = max(len(k) for k in metrics.keys())
        content = "\n".join(
            f"{k.ljust(max_label_width)}: [bold]{v}[/bold]"
            for k, v in metrics.items()
        )

        panel = Panel(
            content,
            title=f"[bold]{title}[/bold]",
            title_align="left",
            box=ROUNDED,
            padding=DocumentGenerator.PADDING,
            width=DocumentGenerator.WIDTH
        )
        DocumentGenerator.console.print(panel)

    @staticmethod
    def table(headers: List[str], rows: List[List[Any]], title: Optional[str] = None) -> None:
        """
        Display a table with optional title.

        :param headers: The headers of the table.
        :type headers: List[str]
        :param rows: The rows of the table.
        :type rows: List[List[Any]]
        """
        table = Table(
            box=ROUNDED,
            title=f"[bold]{title}[/bold]" if title else None,
            show_header=True,
            header_style="bold",
            padding=DocumentGenerator.PADDING,
            collapse_padding=True,
            width=DocumentGenerator.WIDTH
        )

        for header in headers:
            table.add_column(header, justify="left")

        for row in rows:
            table.add_row(*[str(cell) for cell in row])

        DocumentGenerator.console.print(table)

    @staticmethod
    def list(items: List[str], title: Optional[str] = None) -> None:
        """
        Display a list of items.

        :param items: The items to display.
        :type items: List[str]
        :param title: The title of the list.
        :type title: Optional[str]
        """
        if title:
            DocumentGenerator.console.print(f"\n[bold]{title}[/bold]")

        for item in items:
            DocumentGenerator.console.print(f"• {item}")
