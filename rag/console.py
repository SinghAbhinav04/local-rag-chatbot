"""
Rich console singleton with the project's custom theme.
Import `console` from here wherever you need styled output.
"""

from rich.console import Console
from rich.theme import Theme


custom_theme = Theme({
    "user":   "bold cyan",
    "ai":     "bold green",
    "system": "bold yellow",
    "cmd":    "bold magenta",
    "error":  "bold red",
    "info":   "dim white",
    "source": "bold blue",
    "chunk":  "italic dim white",
})

console = Console(theme=custom_theme)
