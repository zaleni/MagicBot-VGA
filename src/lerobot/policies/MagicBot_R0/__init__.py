from .configuration_fastwam import MagicBotR0Config, MagicBotR0DatasetConfig

__all__ = ["MagicBotR0Config", "MagicBotR0DatasetConfig", "MagicBotR0Policy"]


def __getattr__(name: str):
    if name == "MagicBotR0Policy":
        from .modeling_fastwam import MagicBotR0Policy

        return MagicBotR0Policy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
