from vipe.priors.depth.unidepth.models.encoder import _make_dinov2_model


def dinov2_vits14(config, pretrained: bool = True, **kwargs):
    vit = _make_dinov2_model(
        arch_name="vit_small",
        pretrained=config["pretrained"],
        output_idx=config.get("output_idx", [3, 6, 9, 12]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        num_register_tokens=config.get("num_register_tokens", 0),
        use_norm=config.get("use_norm", False),
        interpolate_offset=config.get("interpolate_offset", 0.0),
        frozen_stages=config.get("frozen_stages", 0),
        # freeze_norm=config.get("freeze_norm", False),
        **kwargs,
    )
    return vit


def dinov2_vitb14(config, pretrained: bool = True, **kwargs):
    vit = _make_dinov2_model(
        arch_name="vit_base",
        pretrained=config["pretrained"],
        output_idx=config.get("output_idx", [3, 6, 9, 12]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        num_register_tokens=config.get("num_register_tokens", 0),
        use_norm=config.get("use_norm", False),
        interpolate_offset=config.get("interpolate_offset", 0.0),
        frozen_stages=config.get("frozen_stages", 0),
        # freeze_norm=config.get("freeze_norm", False),
        **kwargs,
    )
    return vit


def dinov2_vitl14(config, pretrained: str = "", **kwargs):
    vit = _make_dinov2_model(
        arch_name="vit_large",
        pretrained=config["pretrained"],
        output_idx=config.get("output_idx", [5, 12, 18, 24]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        num_register_tokens=config.get("num_register_tokens", 0),
        use_norm=config.get("use_norm", False),
        interpolate_offset=config.get("interpolate_offset", 0.0),
        frozen_stages=config.get("frozen_stages", 0),
        # freeze_norm=config.get("freeze_norm", False),
        **kwargs,
    )
    return vit


def dinov2_vitg14(config, pretrained: str = "", **kwargs):
    vit = _make_dinov2_model(
        arch_name="vit_giant2",
        ffn_layer="swiglufused",
        pretrained=config["pretrained"],
        output_idx=config.get("output_idx", [10, 20, 30, 40]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        num_register_tokens=config.get("num_register_tokens", 0),
        use_norm=config.get("use_norm", False),
        interpolate_offset=config.get("interpolate_offset", 0.0),
        **kwargs,
    )
    return vit
