"""LoRA Prober for Qwen-Image 2.1, Krea 2 and MiniMax H3: find which blocks and layers carry a
character LoRA's identity, and compare checkpoints and training runs side by side (videos for MiniMax H3)."""

from .compare_node import NODE_CLASS_MAPPINGS as _COMPARE_NODES
from .compare_node import NODE_DISPLAY_NAME_MAPPINGS as _COMPARE_NAMES
from .probe_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **_COMPARE_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **_COMPARE_NAMES}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
