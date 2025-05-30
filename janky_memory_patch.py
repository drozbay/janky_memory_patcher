# janky memory patcher
# drozbay 2025-02-07

import comfy.model_management
import comfy.model_patcher
import logging

class ModelMemoryPatcher:
    original_ratio = None
    original_partially_load = None

    # initial values
    MEMORY_THRESHOLD = 1024 * 1024 * 1024 * 10
    MEMORY_REDUCTION = 1024 * 1024 * 1024 * 0.0
    MEMORY_PARTIAL = 1024 * 1024 * 1024 * 0.0

    @classmethod
    def init_patch(cls) -> None:
        cls._init_ratio_patch()
        cls._init_partial_load_patch()

    @classmethod
    def _init_ratio_patch(cls) -> None:
        if cls.original_ratio is None:
            cls.original_ratio = comfy.model_management.MIN_WEIGHT_MEMORY_RATIO
            cls.set_ratio(cls.original_ratio)
            logging.info(f"Initialized janky MIN_WEIGHT_MEMORY_RATIO patch: {comfy.model_management.MIN_WEIGHT_MEMORY_RATIO}")

    @classmethod
    def _init_partial_load_patch(cls) -> None:
        if cls.original_partially_load is None:
            logging.info("Initializing janky partial load memory patch")
            cls.original_partially_load = comfy.model_patcher.ModelPatcher.partially_load
            
            def new_partially_load(self, device_to, extra_memory=0, force_patch_weights=False) -> int:
                with self.use_ejected(skip_and_inject_on_exit_only=True):
                    unpatch_weights = self.model.current_weight_patches_uuid is not None and (
                        self.model.current_weight_patches_uuid != self.patches_uuid or force_patch_weights)
                    
                    used = self.model.model_loaded_weight_memory
                    self.unpatch_model(self.offload_device, unpatch_weights=unpatch_weights)
                    if unpatch_weights:
                        extra_memory += (used - self.model.model_loaded_weight_memory)

                    self.patch_model(load_weights=False)
                    full_load = False
                    if self.model.model_lowvram == False and self.model.model_loaded_weight_memory > 0:
                        self.apply_hooks(self.forced_hooks, force_apply=True)
                        return 0
                    if self.model.model_loaded_weight_memory + extra_memory > self.model_size():
                        full_load = True
                    
                    current_used = self.model.model_loaded_weight_memory
                    adjusted_current_used = current_used
                    adjusted_extra_memory = extra_memory
                    # custom code
                    comfy.model_management.soft_empty_cache()
                    comfy.model_management.unload_all_models()
                    import gc
                    gc.collect()

                    if (full_load is False and self.model_size() > cls.MEMORY_THRESHOLD):
                        if (cls.MEMORY_REDUCTION <= 1e-6 and cls.MEMORY_PARTIAL >= 1e-6):
                            adjusted_extra_memory = 0
                            adjusted_current_used = max(current_used, cls.MEMORY_PARTIAL)
                            logging.info(f"Manual partial load set to {int(adjusted_current_used + adjusted_extra_memory):,} from {int(current_used + extra_memory):,}")
                        elif current_used > cls.MEMORY_REDUCTION:
                            adjusted_current_used = current_used - cls.MEMORY_REDUCTION
                            adjusted_current_used = max(0, adjusted_current_used)
                            logging.info(f"Adjusting current_used memory from {int(current_used):,} to {int(adjusted_current_used):,}")
                        else:
                            adjusted_extra_memory = extra_memory - cls.MEMORY_REDUCTION
                            adjusted_extra_memory = max(0, adjusted_extra_memory)
                            logging.info(f"Adjusting extra_memory from {int(extra_memory):,} to {int(adjusted_extra_memory):,}")

                        adjusted_total = adjusted_current_used + adjusted_extra_memory
                        try:
                            self.load(device_to, lowvram_model_memory=adjusted_total, 
                                    force_patch_weights=force_patch_weights, full_load=full_load)
                        except Exception as e:
                            self.detach()
                            raise e
                        
                        return self.model.model_loaded_weight_memory - current_used
                    else:
                        logging.info(f"Skipping memory patcher. full_load: {full_load}, model size: {int(self.model_size()):,}")
                    # end custom code

                    try:
                        self.load(device_to, lowvram_model_memory=current_used + extra_memory, 
                                force_patch_weights=force_patch_weights, full_load=full_load)
                    except Exception as e:
                        self.detach()
                        raise e

                    return self.model.model_loaded_weight_memory - current_used

            # Apply patch
            comfy.model_patcher.ModelPatcher.partially_load = new_partially_load

    @classmethod
    def restore(cls) -> None:
        cls.restore_ratio()
        cls.restore_partial_load()

    @classmethod
    def restore_ratio(cls) -> None:
        if cls.original_ratio is not None:
            comfy.model_management.MIN_WEIGHT_MEMORY_RATIO = cls.original_ratio
            logging.info(f"Restored MIN_WEIGHT_MEMORY_RATIO to: {cls.original_ratio}")
            cls.original_ratio = None

    @classmethod
    def restore_partial_load(cls) -> None:
        if cls.original_partially_load is not None:
            comfy.model_patcher.ModelPatcher.partially_load = cls.original_partially_load
            cls.original_partially_load = None
            logging.info("Restored original partially_load function")

    @classmethod
    def set_ratio(cls, ratio) -> None:
        if cls.original_ratio is None:
            cls.original_ratio = comfy.model_management.MIN_WEIGHT_MEMORY_RATIO
        comfy.model_management.MIN_WEIGHT_MEMORY_RATIO = ratio
        logging.info(f"MIN_WEIGHT_MEMORY_RATIO patched to: {ratio}")

    @classmethod
    def set_threshold(cls, threshold) -> None:
        cls.MEMORY_THRESHOLD = threshold
        logging.info(f"Model size threshold set to: {int(threshold):,}")

    @classmethod
    def set_reduction(cls, reduction) -> None:
        cls.MEMORY_REDUCTION = reduction
        logging.info(f"Memory reduction set to: {int(reduction):,}")

    @classmethod
    def set_manual_partial(cls, manual_partial) -> None:
        cls.MEMORY_PARTIAL = manual_partial
        logging.info(f"Manual partial load set to: {int(manual_partial):,}")


class MemoryPatcherNode:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required":
            {
                "model": ("MODEL",),
            },
            "optional":
            {
                "min_weight_memory_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The minimum partially loaded size as a fraction of VRAM (default is 0.1 for NVIDIA GPUs)."}),
                "model_threshold_gb": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Only attempt to apply the patch when the model size exceeds this threshold."}),
                "buffer_gb": ("FLOAT", {"default": 0.5, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "The amount of memory to reduce when the model is partially loaded."}),
                "manual_partial_gb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Only used if buffer_gb is 0. Manually set partial load size."}),
                "enable": ("BOOLEAN", {"default": True, "tooltip": "Enable or disable the memory patching. (You have to run this node again to disable the patching.)"}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches"
    EXPERIMENTAL = True
    DESCRIPTION = """
This is a janky experiment to patch the memory management
of the model when a model is partially loaded. Sort of like
tricking ComfyUI to doing extra block-swapping. It probably
won't work as expected but may help some situations.
Use at your own risk.
(model is passthrough, patch is applied to the backend)
"""

    def patch(self, model, min_weight_memory_ratio, model_threshold_gb, buffer_gb, manual_partial_gb, enable) -> tuple:
        if enable:
            ModelMemoryPatcher.init_patch()
            ModelMemoryPatcher.set_ratio(min_weight_memory_ratio)
            ModelMemoryPatcher.set_threshold(model_threshold_gb * 1024 * 1024 * 1024)
            ModelMemoryPatcher.set_reduction(buffer_gb * 1024 * 1024 * 1024)
            ModelMemoryPatcher.set_manual_partial(manual_partial_gb * 1024 * 1024 * 1024)
        else:
            ModelMemoryPatcher.restore()

        return (model,)
    

NODE_CLASS_MAPPINGS = {
    "Janky Memory Patcher": MemoryPatcherNode
}