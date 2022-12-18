_base_ = ['./ppyoloe_s_300e_g8b256_base.py']


max_epochs = 80

model = dict(
    backbone=dict(
        use_alpha=True,
        ),
    train_cfg=dict(
        initial_epoch=30,
        ),
    init_cfg=dict(type='Pretrained',
                  checkpoint='pkgs/_ppyoloe_crn_s_obj365_pretrained.pth'
                  )
    )


optimizer = dict(
    lr=0.001 / 64 * 64,
    )

checkpoint_config = dict(interval=10)