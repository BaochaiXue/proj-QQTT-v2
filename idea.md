1. Real-time PCD demo

* 做一个 real-time transforming into PCD 的 demo，用来展示最终的 FPS。
* 重新检查当前的 visualization issue。
* 和 Max 讨论 visualization 的问题及解决方案。

2. FFS

* 目标：下周把 FFS 做成 ready-to-use 的部分。
* 结合 confidence 重新梳理 FFS output，让输出逻辑更清晰、更稳健。
* 检查并测试 FFS 的 batch inference。

3. Newton robot demo

* 完成 Newton robot demo。
* 检查并分析当前的 simulation speed problem，定位瓶颈。
把这三点自己重新梳理下，看看你的plate里还有什么没完成的，列的详细一点，现在每次展示逻辑性比较缺失，就容易让听众觉得不靠谱，所以结合每一点自己罗列清楚，下周对应每一点给出结论，不要只给出observation，要每一个都解释清楚
Hanxiao我剛剛check你PhysTwin的code
我們只對occ_mask做dilation而已並沒有對obj_mask做特殊操作
@莫怨東風當自嗟 關於disparity confidence的部分，你可以利用輸出的logits，加上threshold去做filtering
https://github.com/NVlabs/Fast-FoundationStereo/blob/f8442a5f406d3058e060c48acbd019963e54f490/core/foundation_stereo.py#L222
你可以試試不同方法
1. Max Probability:

confidence, pred_disp = prob.max(dim=1)  # both: (B, H, W)

2. Entropy-Based Confidence

entropy = -(prob * (prob + 1e-8).log()).sum(dim=1)  # (B, H, W)
max_entropy = torch.log(torch.tensor(prob.shape[1], dtype=torch.float))
confidence = 1.0 - entropy / max_entropy

3. Soft Argmax + Variance

d_vals = torch.arange(D, device=prob.device).float()  # (D,)
d_vals = d_vals.view(1, D, 1, 1)

pred_disp = (prob * d_vals).sum(dim=1)           # E[d], (B, H, W)
pred_disp2 = (prob * d_vals**2).sum(dim=1)       # E[d^2]
variance = pred_disp2 - pred_disp**2             # Var[d]
confidence = 1.0 / (variance + 1e-4)            # inverse variance
prediction
^ 可以看這邊 (https://github.com/NVlabs/Fast-FoundationStereo/blob/f8442a5f406d3058e060c48acbd019963e54f490/core/foundation_stereo.py#L247)
@啸 關於Real-time PCD demo的部分，最終我們想要實時操作的同時，輸出RGB + depth stream並重建3d pcd for every frames (?)
嗯嗯理论上应该是完整3D pcd
后面还要给heqian这边来做inverse physics, 所以我们这边可以视作独立的部分
quality和速度越多越好完整pcd是指要利用3d generative prior (?)
完整我指多个camera merge
3 views合再一起
你们这边目前先不用care generative model部分
got itgot it嗯嗯@莫怨東風當自嗟 我覺得你目前可以這麼規劃

(1) Test the FPS of input data stream
   - option 1: RGB + native depth
   - option 2: RGB + IR images
   - check what is the maximum FPS we could achieve in these two settings
(2) Test FastFoundationStereo's performance:
    - 大部分參數延用，沒有特殊情況不要去動(e.g., scale=1.0, valid_iters=4)，in case the performance might drop
    - 嘗試利用confidence去對部分pixel做filtering (見上面那串訊息)
    - 使用ONNX/TRT版本的FFS，測試這個版本在3 camera views下的FPS
        - 可以嘗試compile batched version，看是否能再加速
    - (Potential problem) If confidence filtering work out, how to enable this feature within ONNT/TRT compiled version?
(3) Map depth to 3D pcd:
    - Try to follow what Hanxiao had done previously
        - pcd outlier filtering via open3d
        - only care about object (i.e., unproject from object mask)
        - depth clipping within some range
這些測試完後，就可以結合成兩種demo case:
(1) RGB + depth --> 3D pcd
(2) RGB + IR --> FFS --> 3D pcd
現在實作時盡量keep your code clean，後續整合或是我們幫忙debug時會更容易些
浮点不要有，空洞在没有浮点基础上越少越好