"""Interface for ``python -m xrpd_toolbox``."""

import click

from ._version import __version__

__all__ = ["main"]


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, message="%(version)s")
@click.pass_context
def main(ctx: click.Context) -> None:
    """xrpd_toolbox command line interface."""
    pass


@main.command(name="bad_pixel_gui")
@click.pass_context
def bad_pixel_gui(ctx: click.Context) -> None:
    """Launch the bad pixel GUI."""

    from xrpd_toolbox.gui.bad_pixel_gui import run_bad_pixel_gui

    run_bad_pixel_gui()


if __name__ == "__main__":
    main()
