

_base_ = ['faster-rcnn_r50_fpn_1x_coco.py']  # referencia nominal

custom_imports = dict(
    imports=['src.freqfusion.freqfusion_module'],  # wrapper propio
    allow_failed_imports=False
)

model = dict(
    type='FasterRCNN',
    backbone=dict(type='ResNet', depth=50),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        
        fusion_cfg=dict(type='FreqFusion', low_pass=True, high_pass=True, offset=True)
    ),
    rpn_head=dict(type='RPNHead'),
    roi_head=dict(type='StandardRoIHead')
)

train_cfg = dict(max_epochs=1)  # avance: 1 época