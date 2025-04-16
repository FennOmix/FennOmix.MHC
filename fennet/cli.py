import click
import fennet

from fennet.mhc.cli import mhc

@click.group(
    context_settings=dict(help_option_names=["-h", "--help"]),
    invoke_without_command=True,
)
@click.pass_context
@click.version_option(fennet.__version__, "-v", "--version")
def run(ctx, **kwargs):
    click.echo(
        rf"""
                   _____                     _
                  |  ___|__ _ __  _ __   ___| |_
                  | |_ / _ \ '_ \| '_ \ / _ \ __|
                  |  _|  __/ | | | | | |  __/ |_
                  |_|  \___|_| |_|_| |_|\___|\__|
        ...................................................
        .{fennet.__version__.center(50)}.
        .{fennet.__github__.center(50)}.
        .{fennet.__license__.center(50)}.
        ...................................................
        """
    )
    if ctx.invoked_subcommand is None:
        click.echo(run.get_help(ctx))


run.add_command(mhc)


if __name__ == "__main__":
    run()
