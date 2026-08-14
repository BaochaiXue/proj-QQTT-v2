#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import numpy as np
import torch

from demo_v7.runtime.models.matching import Matching
from demo_v7.runtime.models.utils import (
    make_matching_plot,
    AverageTimer,
    read_image,
)

torch.set_grad_enabled(False)

_MATCHING_MODEL_CACHE = {}


def get_matching_model(
    *,
    nms_radius=4,
    keypoint_threshold=0.005,
    max_keypoints=1024,
    superglue="indoor",
    sinkhorn_iterations=20,
    match_threshold=0.2,
    device=None,
):
    """Return a cached SuperPoint+SuperGlue model for the given config.

    Weights and eval mode are identical to constructing the model inline; the
    cache only lets the prewarm path load checkpoints before the first request.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (
        nms_radius,
        keypoint_threshold,
        max_keypoints,
        superglue,
        sinkhorn_iterations,
        match_threshold,
        str(device),
    )
    model = _MATCHING_MODEL_CACHE.get(key)
    if model is None:
        config = {
            "superpoint": {
                "nms_radius": nms_radius,
                "keypoint_threshold": keypoint_threshold,
                "max_keypoints": max_keypoints,
            },
            "superglue": {
                "weights": superglue,
                "sinkhorn_iterations": sinkhorn_iterations,
                "match_threshold": match_threshold,
            },
        }
        model = Matching(config).eval().to(device)
        _MATCHING_MODEL_CACHE[key] = model
    return model


def extract_superpoint_features(matching, image_tensor):
    """Run SuperPoint once on one prepared image tensor.

    Returns the raw SuperPoint dict (values are per-item tensor lists) exactly
    as Matching.forward produces internally; feeding it back through Matching
    skips SuperPoint for that side with identical numerics.
    """
    return matching.superpoint({"image": image_tensor})


def prepare_candidate_features(matching, image, device, resize=[-1], resize_float=False):
    """read_image + SuperPoint for one candidate image, matching-loop style.

    Uses the same read_image arguments as the image_pair_matching loop so a
    feature set computed ahead of time (align prerender) is byte-identical to
    one computed inside the loop.
    """
    image0, inp0, _scales = read_image(image, device, resize, 0, resize_float)
    if image0 is None:
        raise ValueError("Problem reading candidate image")
    return {
        "image": image0,
        "input_tensor": inp0,
        "features": extract_superpoint_features(matching, inp0),
    }


def image_pair_matching(
    input_images,
    ref_image,
    output_dir,
    resize=[-1],
    resize_float=False,
    superglue="indoor",
    max_keypoints=1024,
    keypoint_threshold=0.005,
    nms_radius=4,
    sinkhorn_iterations=20,
    match_threshold=0.2,
    viz=False,
    fast_viz=False,
    cache=True,
    show_keypoints=False,
    viz_extension="png",
    save=False,
    viz_best=True,
    candidate_features=None,
):

    """Return the image pair matching.

    candidate_features optionally supplies prepare_candidate_features output
    for every input image (align prerender); SuperPoint then runs zero times
    for candidates and once for the reference instead of once per pair.

    When no viz/cache/save side channel is requested (the formal align call),
    the loop runs GPU-resident: per-candidate matches stay on the device, the
    match counts synchronize once after the loop, and only the winning pair
    is copied to the CPU. The forwards themselves are unchanged — same model,
    same batch size, same order — so the returned arrays are byte-identical
    to the legacy per-candidate-sync loop; only D2H copy timing moves.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Running inference on device "{}"'.format(device))
    matching = get_matching_model(
        nms_radius=nms_radius,
        keypoint_threshold=keypoint_threshold,
        max_keypoints=max_keypoints,
        superglue=superglue,
        sinkhorn_iterations=sinkhorn_iterations,
        match_threshold=match_threshold,
        device=device,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory "{}"'.format(output_dir))
    if viz:
        print('`Will writ`e visualization images to directory "{}"'.format(output_dir))

    timer = AverageTimer(newline=True)
    match_nums = []
    match_result = []

    best_match = {}
    best_match_num = -1

    gpu_resident = not (viz or viz_best or save or cache)
    # (candidate keypoints, matches0, matching_scores0) tensors per candidate
    # plus the running count tensors — synchronized once after the loop.
    gpu_outputs = []
    gpu_match_counts = []

    # Reference-side work is identical for every candidate pair: read the
    # reference image and run its SuperPoint once, then hand the features to
    # Matching (which skips SuperPoint whenever keypoints are supplied). Same
    # inputs -> same features -> same matches as the per-pair version.
    rot0, rot1 = 0, 0
    image1, inp1, scales1 = read_image(ref_image, device, resize, rot1, resize_float)
    if image1 is None:
        print("Problem reading ref image")
        exit(1)
    ref_features = extract_superpoint_features(matching, inp1)
    ref_kpts = ref_features["keypoints"][0].cpu().numpy()

    for i, image in enumerate(input_images):
        matches_path = output_dir / "matches_{}.npz".format(i)
        viz_path = output_dir / "matches_{}.{}".format(i, viz_extension)

        do_match = True
        do_viz = viz
        if cache:
            if matches_path.exists():
                try:
                    results = np.load(matches_path)
                except:
                    raise IOError("Cannot load matches .npz file: %s" % matches_path)

                kpts0, kpts1 = results["keypoints0"], results["keypoints1"]
                matches, conf = results["matches"], results["match_confidence"]
                do_match = False
            if viz and viz_path.exists():
                do_viz = False
            timer.update("load_cache")

        if candidate_features is not None:
            candidate = candidate_features[i]
            image0, inp0 = candidate["image"], candidate["input_tensor"]
            cand_features = candidate["features"]
        else:
            image0, inp0, scales0 = read_image(
                image, device, resize, rot0, resize_float
            )
            if image0 is None:
                print("Problem reading image pair: {} and ref".format(i))
                exit(1)
            cand_features = None
        timer.update("load_image")

        if do_match:
            if cand_features is None:
                cand_features = extract_superpoint_features(matching, inp0)
            data = {"image0": inp0, "image1": inp1}
            data.update({k + "0": v for k, v in cand_features.items()})
            data.update({k + "1": v for k, v in ref_features.items()})
            pred = matching(data)
            if gpu_resident:
                matches0 = pred["matches0"][0]
                gpu_outputs.append(
                    (
                        cand_features["keypoints"][0],
                        matches0,
                        pred["matching_scores0"][0],
                    )
                )
                gpu_match_counts.append((matches0 > -1).sum())
                timer.update("matcher")
                continue
            kpts0 = cand_features["keypoints"][0].cpu().numpy()
            kpts1 = ref_kpts
            matches = pred["matches0"][0].cpu().numpy()
            conf = pred["matching_scores0"][0].cpu().numpy()
            timer.update("matcher")

            out_matches = {
                "keypoints0": kpts0,
                "keypoints1": kpts1,
                "matches": matches,
                "match_confidence": conf,
            }
            match_result.append(out_matches)
            if save:
                np.savez(str(matches_path), **out_matches)
        else:
            match_result.append(results)

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        match_nums.append(len(mkpts0))

        if len(mkpts0) > best_match_num:
            best_match_num = len(mkpts0)
            best_match["image0"] = image0
            best_match["image1"] = image1
            best_match["kpts0"] = kpts0
            best_match["kpts1"] = kpts1
            best_match["mkpts0"] = mkpts0
            best_match["mkpts1"] = mkpts1
            best_match["mconf"] = mconf

        if do_viz:
            import matplotlib.cm as cm

            color = cm.jet(mconf)
            text = [
                "SuperGlue",
                "Keypoints: {}:{}".format(len(kpts0), len(kpts1)),
                "Matches: {}".format(len(mkpts0)),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append("Rotation: {}:{}".format(rot0, rot1))

            k_thresh = matching.superpoint.config["keypoint_threshold"]
            m_thresh = matching.superglue.config["match_threshold"]
            small_text = [
                "Keypoint Threshold: {:.4f}".format(k_thresh),
                "Match Threshold: {:.2f}".format(m_thresh),
                "Image Pair: {} : ref".format(i),
            ]

            make_matching_plot(
                image0,
                image1,
                kpts0,
                kpts1,
                mkpts0,
                mkpts1,
                color,
                text,
                viz_path,
                show_keypoints,
                fast_viz,
                small_text,
            )

            timer.update("viz_match")

    if gpu_resident:
        # One synchronization for every candidate's count; the winner picks
        # with the same first-max tie-break as the legacy list.index below.
        counts = torch.stack(gpu_match_counts).cpu().numpy()
        match_nums = [int(count) for count in counts]
        best_pose = match_nums.index(max(match_nums))
        best_kpts0, best_matches0, best_scores0 = gpu_outputs[best_pose]
        return best_pose, {
            "keypoints0": best_kpts0.cpu().numpy(),
            "keypoints1": ref_kpts,
            "matches": best_matches0.cpu().numpy(),
            "match_confidence": best_scores0.cpu().numpy(),
        }

    best_pose = match_nums.index(max(match_nums))

    if viz_best:
        import matplotlib.cm as cm

        viz_path = f"{output_dir}/best_match.{viz_extension}"
        color = cm.jet(best_match["mconf"])
        text = [
            "SuperGlue",
            "Keypoints: {}:{}".format(
                len(best_match["kpts0"]), len(best_match["kpts1"])
            ),
            "Matches: {}".format(len(best_match["mkpts0"])),
        ]

        make_matching_plot(
            best_match["image0"],
            best_match["image1"],
            best_match["kpts0"],
            best_match["kpts1"],
            best_match["mkpts0"],
            best_match["mkpts1"],
            color,
            text,
            viz_path,
            show_keypoints,
            fast_viz,
        )

        timer.update("viz_match")
    return best_pose, match_result[best_pose]
