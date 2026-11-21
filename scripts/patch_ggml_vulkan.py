#!/usr/bin/env python3
"""Idempotently append the 4 zero-copy Vulkan interop C accessors to ggml-vulkan.cpp.

Run automatically by `pixi run setup` and `pixi run update-deps` so fresh clones
and upstream `llama.cpp` updates always expose `VkDevice`, compute queue family,
raw `VkBuffer`, and queue submission to `vknumpy` / `vkshader`.
"""

from __future__ import annotations

from pathlib import Path
import sys

TARGET = Path(__file__).resolve().parent.parent / "third_party/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp"
MARKER = "ggml_backend_vk_get_raw_device"

SNIPPET = """
extern "C" {

GGML_BACKEND_API void * ggml_backend_vk_get_raw_device(ggml_backend_t backend) {
    if (!backend || !ggml_backend_is_vk(backend) || !backend->context) {
        return nullptr;
    }
    auto * ctx = static_cast<ggml_backend_vk_context *>(backend->context);
    if (!ctx->device) {
        return nullptr;
    }
    return static_cast<void *>(static_cast<VkDevice>(ctx->device->device));
}

GGML_BACKEND_API uint32_t ggml_backend_vk_get_compute_queue_family(ggml_backend_t backend) {
    if (!backend || !ggml_backend_is_vk(backend) || !backend->context) {
        return UINT32_MAX;
    }
    auto * ctx = static_cast<ggml_backend_vk_context *>(backend->context);
    if (!ctx->device || !ctx->device->compute_queue) {
        return UINT32_MAX;
    }
    return ctx->device->compute_queue->queue_family_index;
}

GGML_BACKEND_API void * ggml_backend_vk_get_raw_buffer(ggml_backend_buffer_t buffer) {
    if (!buffer || !buffer->context) {
        return nullptr;
    }
    auto * bctx = static_cast<ggml_backend_vk_buffer_context *>(buffer->context);
    if (!bctx->dev_buffer) {
        return nullptr;
    }
    return static_cast<void *>(static_cast<VkBuffer>(bctx->dev_buffer->buffer));
}

GGML_BACKEND_API int ggml_backend_vk_submit_compute_cmd(ggml_backend_t backend, void * cmd_buf, void * fence) {
    if (!backend || !ggml_backend_is_vk(backend) || !backend->context || !cmd_buf || !fence) {
        return -1;
    }
    auto * ctx = static_cast<ggml_backend_vk_context *>(backend->context);
    if (!ctx->device || !ctx->device->compute_queue || !ctx->device->compute_queue->handle) {
        return -1;
    }
    vk::CommandBuffer cb(static_cast<VkCommandBuffer>(cmd_buf));
    vk::Fence fn(static_cast<VkFence>(fence));
    vk::SubmitInfo si(0, nullptr, nullptr, 1, &cb, 0, nullptr);
    ctx->device->compute_queue->handle->submit(si, fn);
    return 0;
}

} // extern "C"
"""


def main() -> int:
    if not TARGET.is_file():
        print(f"Error: {TARGET} not found. Run `pixi run setup` first.", file=sys.stderr)
        return 1

    text = TARGET.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"ggml-vulkan.cpp already patched ({MARKER} present).")
        return 0

    TARGET.write_text(text.rstrip() + "\n" + SNIPPET, encoding="utf-8")
    print(f"Patched {TARGET} with vkdata zero-copy Vulkan interop accessors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
