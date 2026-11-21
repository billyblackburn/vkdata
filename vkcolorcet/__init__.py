"""vkcolorcet: Perceptually uniform & categorical colormaps (colorcet-compatible) for Vulkan GPU."""

from .palettes import (
    get_lut_gpu,
    get_lut_u32,
    get_rgb_lut,
    palette,
    swatches,
)

# Expose top colormaps as direct module attributes (matching `import colorcet as cc; cc.fire, cc.bmy, ...`)
fire = palette["fire"]
fire_r = palette["fire_r"]
bmy = palette["bmy"]
bgy = palette["bgy"]
bgyw = palette["bgyw"]
kbc = palette["kbc"]
kb = palette["kb"]
kg = palette["kg"]
kr = palette["kr"]
bmw = palette["bmw"]
viridis = palette["viridis"]
inferno = palette["inferno"]
plasma = palette["plasma"]
magma = palette["magma"]
turbo = palette["turbo"]
coolwarm = palette["coolwarm"]
bwr = palette["bwr"]
bkr = palette["bkr"]
rainbow = palette["rainbow"]
gray = palette["gray"]
blues = palette["blues"]
greens = palette["greens"]
reds = palette["reds"]
hsv = palette["hsv"]
hsv_r = palette["hsv_r"]
glasbey = palette["glasbey"]
glasbey_dark = palette["glasbey_dark"]
category10 = palette["category10"]

__all__ = [
    "bgy", "bgyw", "bkr", "blues", "bmw", "bmy", "bwr", "category10", "coolwarm",
    "fire", "fire_r", "get_lut_gpu", "get_lut_u32", "get_rgb_lut", "glasbey", "glasbey_dark",
    "gray", "greens", "hsv", "hsv_r", "inferno", "kb", "kbc", "kg", "kr", "magma",
    "palette", "plasma", "rainbow", "reds", "swatches", "turbo", "viridis",
]
