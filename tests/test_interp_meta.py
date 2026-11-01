"""Provenance helpers: the quantization label and the fingerprint of weights an engine
quantizes at load."""

import json

from simtest.interp.meta import load_fingerprint, quantization, weights_fingerprint


def model_dir(tmp_path, cfg):
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "model.safetensors").write_bytes(b"\0" * 64)
    return tmp_path


def test_fp8_block_checkpoint_is_described(tmp_path):
    """The official Qwen3.8 FP8 config has no bits or group size: say what it is."""
    d = model_dir(tmp_path, {"quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                                                     "weight_block_size": [128, 128],
                                                     "modules_to_not_convert": ["lm_head", "model.embed_tokens"]}})
    assert quantization(d) == "fp8 e4m3, weight blocks 128x128, dynamic activations"


def test_other_quantizations_keep_their_label(tmp_path):
    d = model_dir(tmp_path, {"quantization": {"mode": "affine", "bits": 8, "group_size": 64}})
    assert quantization(d) == "affine 8-bit, group 64"
    model_dir(d, {"text_config": {"dtype": "bfloat16"}})
    assert quantization(d) == "none (bfloat16)"


def test_quantize_on_load_changes_the_fingerprint(tmp_path):
    """bf16 files quantized by the engine at load are other weights than the files read as
    they are, so vectors built on one must not steer the other."""
    d = model_dir(tmp_path, {"text_config": {"dtype": "bfloat16"}})
    files = weights_fingerprint(d)
    assert load_fingerprint(d, None) == files
    fp8 = load_fingerprint(d, "vllm:fp8")
    assert fp8 == files + "+vllm:fp8" and fp8 != load_fingerprint(d, "hf:fp8-block128")
