import torch


class Embedding(torch.nn.Module):
    """
    Dummy base class for embedding networks.

    All embeddings should accept as arguments
    `num_ifos`, `context_dim` and `strain_dim` as
    their first through third arguments, since the
    CLI links to this argument from the datamodule
    at initialization time.

    This class obvioulsy isn't necessary, but leaving this
    as a reminder that we may wan't to enforce
    the above behavior more explicitly in the future
    """
