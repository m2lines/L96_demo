import inspect

from IPython.display import HTML, display
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer


def display_source(non_builtin_object):
    """Displays the source code of the given object with syntax highlighting."""
    code = inspect.getsource(non_builtin_object)
    html = highlight(code, PythonLexer(), HtmlFormatter(style="colorful"))
    stylesheet = f"<style>{HtmlFormatter().get_style_defs('.highlight')}</style>"
    display(HTML(f"{stylesheet}{html}"))
