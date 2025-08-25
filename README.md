# Janky Memory Patcher for ComfyUI

A custom node that provides manual control over ComfyUI's automatic memory management to help run workflows that would otherwise encounter out-of-memory (OOM) errors.

## Will This Node Help You?

**Check your ComfyUI terminal output when you get OOM errors:**
- **If you see `loaded partially` or `loaded completely` with a number > 1000** - This node can help! You have room to reduce memory allocation
- **If you see `loaded partially"` with a small number like 128** - This node won't help. You're already at minimum and need to use GGUF models, reduce batch size/resolution, or get more VRAM

## What It Does

This node gives you manual control over ComfyUI's built-in partial loading system, allowing you to fine-tune how much memory is allocated when models are partially loaded into VRAM. It's like adding manual "knobs" to ComfyUI's automatic system for when the defaults don't quite work.

## Installation

1. Copy `janky_memory_patch.py` to your `ComfyUI/custom_nodes/` folder
2. Restart ComfyUI
3. Find the node under "model_patches" category as "Janky Memory Patcher"

## Basic Usage

1. Add the "Janky Memory Patcher" node to your workflow
2. Connect any model through it (passes through unchanged)
3. Start with default settings (optionally change the `model_threshold_gb` to ~75% of your VRAM)
4. If you still get OOM errors, increase `buffer_gb` to 1.0 or 2.0
5. Run your workflow

### Extra steps:
6. If you *still* get OOM errors, keep increasing `buffer_gb`, change `force_partial_load` to `True`
7. Still having issues? Try setting `buffer_gb` to 0, then set `manual_partial_gb` to 0.5.
8. if you **still** get OOM errors, go buy more VRAM

**Important:** 
- Settings apply when the node RUNS, not when added
- Settings persist until changed or ComfyUI restarts
- The node is marked as an "output node" so you can connect it to your first model line without using the passthrough - this ensures it runs every time your workflow executes

## Parameters

- **model**: Pass any model through (not modified, just passthrough)
- **min_weight_memory_ratio** (0.0-1.0, default: 0.1): Minimum fraction of VRAM for model weights
- **model_threshold_gb** (default: 10.0): Only apply to models larger than this size
- **buffer_gb** (default: 0.5): Memory to "hold back" during partial loading - increase if getting OOM
- **manual_partial_gb** (default: 0.0): When buffer_gb=0, manually set partial load size
- **enable** (default: True): Toggle patch on/off

- **force_partial_load** (default: False): Force partial loading even when not needed
