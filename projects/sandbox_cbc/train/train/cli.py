from lightning.pytorch.cli import LightningCLI
from train.data.base import BaseDataset
from train.model import PEModel


class PECLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # link shape arguments required by the model
        # architecture from data module so we don't
        # have to specify them twice.
        parser.link_arguments(
            "data.num_ifos",
            "model.init_args.arch.init_args.embedding_net.init_args.num_ifos",  # noqa
            apply_on="instantiate",
        )

        parser.link_arguments(
            "model.init_args.arch.init_args.context_dim",
            "model.init_args.arch.init_args.embedding_net.init_args.context_dim",  # noqa
            apply_on="parse",
        )

        parser.link_arguments(
            "data.num_params",
            "model.init_args.arch.init_args.num_params",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "data.strain_dim",
            "model.init_args.arch.init_args.embedding_net.init_args.strain_dim",  # noqa
            apply_on="instantiate",
        )


def main(args=None):
    # any subclasses of PEModel and BaseDataset
    # will automatically be registered with the CLI
    # and their arguments will be available at
    # the command line
    cli = PECLI(
        PEModel,
        BaseDataset,
        subclass_mode_model=True,
        subclass_mode_data=True,
        run=False,
        save_config_kwargs={"overwrite": True},
        seed_everything_default=101588,
        args=args,
    )
    cli.trainer.fit(cli.model, cli.datamodule)


if __name__ == "__main__":
    main()
