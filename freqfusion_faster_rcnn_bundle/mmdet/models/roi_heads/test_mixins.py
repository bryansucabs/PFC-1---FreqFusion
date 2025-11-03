"""Mixins de test simplificados para escenarios sin máscara."""
import sys

import torch

from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_bboxes,
                        multiclass_nms)

if sys.version_info >= (3, 7):
    from mmdet.utils.contextmanagers import completed


class BBoxTestMixin:

    if sys.version_info >= (3, 7):

        async def async_test_bboxes(self,
                                    x,
                                    img_metas,
                                    proposals,
                                    rcnn_test_cfg,
                                    rescale=False,
                                    **kwargs):
            """Asynchronized test for box head without augmentation."""
            rois = bbox2roi(proposals)
            roi_feats = self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                roi_feats = self.shared_head(roi_feats)
            sleep_interval = rcnn_test_cfg.get('async_sleep_interval', 0.017)

            async with completed(
                    __name__, 'bbox_head_forward',
                    sleep_interval=sleep_interval):
                cls_score, bbox_pred = self.bbox_head(roi_feats)

            img_shape = img_metas[0]['img_shape']
            scale_factor = img_metas[0]['scale_factor']
            det_bboxes, det_labels = self.bbox_head.get_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=rcnn_test_cfg)
            return det_bboxes, det_labels

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros(0, dtype=torch.long)
            return [det_bbox.clone() for _ in range(batch_size)], [
                det_label.clone() for _ in range(batch_size)
            ]

        bbox_results = self._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        num_imgs = len(img_metas)
        rois = rois.reshape(num_imgs, -1, rois.size(-1))
        cls_score = cls_score.reshape(num_imgs, -1, cls_score.size(-1))
        bbox_pred = bbox_pred.reshape(num_imgs, -1, bbox_pred.size(-1))

        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_metas[i]['img_shape'],
                img_metas[i]['scale_factor'],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels

    def aug_test_bboxes(self, feats, img_metas, proposals, rcnn_test_cfg):
        """Test bboxes with test time augmentation."""
        aug_bboxes = []
        aug_scores = []
        for x, img_meta, proposal in zip(feats, img_metas, proposals):
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']

            proposals = bbox_mapping(proposal[0], img_shape, scale_factor,
                                     flip, flip_direction)

            rois = bbox2roi([proposals])
            bbox_results = self._bbox_forward(x, rois)
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']

            bboxes, scores = self.bbox_head.get_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=False,
                cfg=rcnn_test_cfg)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(
            merged_bboxes,
            merged_scores,
            rcnn_test_cfg.score_thr,
            rcnn_test_cfg.nms,
            rcnn_test_cfg.max_per_img)

        return [det_bboxes], [det_labels]


class MaskTestMixin:
    """Stub para cuando no se incluye una cabeza de máscaras."""

    def simple_test_mask(self, *args, **kwargs):  # pragma: no cover - stub
        raise NotImplementedError('La cabeza de máscara no está disponible en este bundle.')

    def aug_test_mask(self, *args, **kwargs):  # pragma: no cover - stub
        raise NotImplementedError('La cabeza de máscara no está disponible en este bundle.')

    async def async_test_mask(self, *args, **kwargs):  # pragma: no cover - stub
        raise NotImplementedError('La cabeza de máscara no está disponible en este bundle.')
