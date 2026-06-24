import torch
from qqtt.utils import logger, cfg
import warp as wp

wp.init()
wp.set_device("cuda:0")
if not cfg.use_graph:
    wp.config.mode = "debug"
    wp.config.verbose = True
    wp.config.verify_autograd_array_access = True


class State:
    def __init__(self, wp_init_vertices, num_control_points):
        self.wp_x = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v_before_collision = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v_before_ground = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v = wp.zeros_like(self.wp_x, requires_grad=True)
        self.wp_vertice_forces = wp.zeros_like(self.wp_x, requires_grad=True)
        # No need to compute the gradient for the control points
        self.wp_control_x = wp.zeros(
            (num_control_points), dtype=wp.vec3, requires_grad=False
        )
        self.wp_control_v = wp.zeros_like(self.wp_control_x, requires_grad=False)

    def clear_forces(self):
        self.wp_vertice_forces.zero_()

    # This takes more time but not necessary, will be overwritten directly
    # def clear_control(self):
    #     self.wp_control_x.zero_()
    #     self.wp_control_v.zero_()

    # def clear_states(self):
    #     self.wp_x.zero_()
    #     self.wp_v_before_ground.zero_()
    #     self.wp_v.zero_()

    @property
    def requires_grad(self):
        """Indicates whether the state arrays have gradient computation enabled."""
        return self.wp_x.requires_grad


@wp.kernel(enable_backward=False)
def copy_vec3(data: wp.array(dtype=wp.vec3), origin: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def copy_int(data: wp.array(dtype=wp.int32), origin: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def copy_float(data: wp.array(dtype=wp.float32), origin: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel
def set_control_points(
    num_substeps: int,
    original_control_point: wp.array(dtype=wp.vec3),
    target_control_point: wp.array(dtype=wp.vec3),
    step: int,
    control_x: wp.array(dtype=wp.vec3),
):
    # Set the control points in each substep
    tid = wp.tid()

    t = float(step + 1) / float(num_substeps)
    control_x[tid] = (
        original_control_point[tid]
        + (target_control_point[tid] - original_control_point[tid]) * t
    )


@wp.kernel
def eval_springs(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    control_x: wp.array(dtype=wp.vec3),
    control_v: wp.array(dtype=wp.vec3),
    num_object_points: int,
    springs: wp.array(dtype=wp.vec2i),
    rest_lengths: wp.array(dtype=float),
    spring_Y: wp.array(dtype=float),
    dashpot_damping: float,
    spring_Y_min: float,
    spring_Y_max: float,
    n_springs_single: int,
    use_batched_rest_lengths: int,
    f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    sid = tid % n_springs_single

    if wp.exp(spring_Y[sid]) > spring_Y_min:

        idx1 = springs[tid][0]
        idx2 = springs[tid][1]

        if idx1 >= num_object_points:
            x1 = control_x[idx1 - num_object_points]
            v1 = control_v[idx1 - num_object_points]
        else:
            x1 = x[idx1]
            v1 = v[idx1]
        if idx2 >= num_object_points:
            x2 = control_x[idx2 - num_object_points]
            v2 = control_v[idx2 - num_object_points]
        else:
            x2 = x[idx2]
            v2 = v[idx2]

        if use_batched_rest_lengths != 0:
            rest = rest_lengths[tid]
        else:
            rest = rest_lengths[sid]

        dis = x2 - x1
        dis_len = wp.length(dis)

        d = dis / wp.max(dis_len, 1e-6)

        spring_force = (
            wp.clamp(wp.exp(spring_Y[sid]), low=spring_Y_min, high=spring_Y_max)
            * (dis_len / rest - 1.0)
            * d
        )

        v_rel = wp.dot(v2 - v1, d)
        dashpot_forces = dashpot_damping * v_rel * d

        overall_force = spring_force + dashpot_forces

        if idx1 < num_object_points:
            wp.atomic_add(f, idx1, overall_force)
        if idx2 < num_object_points:
            wp.atomic_sub(f, idx2, overall_force)


@wp.kernel
def update_vel_from_force(
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    dt: float,
    drag_damping: float,
    reverse_factor: float,
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    v0 = v[tid]
    f0 = f[tid]
    m0 = masses[tid]

    drag_damping_factor = wp.exp(-dt * drag_damping)
    all_force = f0 + m0 * wp.vec3(0.0, 0.0, -9.8) * reverse_factor
    a = all_force / m0
    v1 = v0 + a * dt
    v2 = v1 * drag_damping_factor

    v_new[tid] = v2


@wp.func
def loop(
    i: int,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    masks: wp.array(dtype=wp.int32),
    num_object_points_single: int,
    collision_dist: float,
    clamp_collide_object_elas: float,
    clamp_collide_object_fric: float,
):
    x1 = x[i]
    v1 = v[i]
    m1 = masses[i]
    mask1 = masks[i]

    valid_count = float(0.0)
    J_sum = wp.vec3(0.0, 0.0, 0.0)
    inst_i = i // num_object_points_single
    for k in range(collision_number[i]):
        index = collision_indices[i][k]
        inst_j = index // num_object_points_single
        if inst_j != inst_i:
            continue
        x2 = x[index]
        v2 = v[index]
        m2 = masses[index]
        mask2 = masks[index]

        dis = x2 - x1
        dis_len = wp.length(dis)
        relative_v = v2 - v1
        # If the distance is less than the collision distance and the two points are moving towards each other
        if (
            mask1 != mask2
            and dis_len < collision_dist
            and wp.dot(dis, relative_v) < -1e-4
        ):
            valid_count += 1.0

            collision_normal = dis / wp.max(dis_len, 1e-6)
            v_rel_n = wp.dot(relative_v, collision_normal) * collision_normal
            impulse_n = (-(1.0 + clamp_collide_object_elas) * v_rel_n) / (
                1.0 / m1 + 1.0 / m2
            )
            v_rel_n_length = wp.length(v_rel_n)

            v_rel_t = relative_v - v_rel_n
            v_rel_t_length = wp.max(wp.length(v_rel_t), 1e-6)
            a = wp.max(
                0.0,
                1.0
                - clamp_collide_object_fric
                * (1.0 + clamp_collide_object_elas)
                * v_rel_n_length
                / v_rel_t_length,
            )
            impulse_t = (a - 1.0) * v_rel_t / (1.0 / m1 + 1.0 / m2)

            J = impulse_n + impulse_t

            J_sum += J

    return valid_count, J_sum


@wp.kernel(enable_backward=False)
def update_potential_collision(
    x: wp.array(dtype=wp.vec3),
    masks: wp.array(dtype=wp.int32),
    num_object_points_single: int,
    collision_dist: float,
    grid: wp.uint64,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    x1 = x[i]
    mask1 = masks[i]
    inst_i = i // num_object_points_single

    neighbors = wp.hash_grid_query(grid, x1, collision_dist * 5.0)
    for index in neighbors:
        if index != i:
            inst_j = index // num_object_points_single
            if inst_j != inst_i:
                continue
            x2 = x[index]
            mask2 = masks[index]

            dis = x2 - x1
            dis_len = wp.length(dis)
            # If the distance is less than the collision distance and the two points are moving towards each other
            if mask1 != mask2 and dis_len < collision_dist:
                collision_indices[i][collision_number[i]] = index
                collision_number[i] += 1


@wp.kernel
def object_collision(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    masks: wp.array(dtype=wp.int32),
    collide_object_elas: wp.array(dtype=float),
    collide_object_fric: wp.array(dtype=float),
    collision_dist: float,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
    num_object_points_single: int,
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    v1 = v[tid]
    m1 = masses[tid]

    clamp_collide_object_elas = wp.clamp(collide_object_elas[0], low=0.0, high=1.0)
    clamp_collide_object_fric = wp.clamp(collide_object_fric[0], low=0.0, high=2.0)

    valid_count, J_sum = loop(
        tid,
        collision_indices,
        collision_number,
        x,
        v,
        masses,
        masks,
        num_object_points_single,
        collision_dist,
        clamp_collide_object_elas,
        clamp_collide_object_fric,
    )

    if valid_count > 0:
        J_average = J_sum / valid_count
        v_new[tid] = v1 - J_average / m1
    else:
        v_new[tid] = v1


@wp.kernel
def integrate_ground_collision(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    collide_elas: wp.array(dtype=float),
    collide_fric: wp.array(dtype=float),
    dt: float,
    reverse_factor: float,
    x_new: wp.array(dtype=wp.vec3),
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    x0 = x[tid]
    v0 = v[tid]

    normal = wp.vec3(0.0, 0.0, 1.0) * reverse_factor

    x_z = x0[2]
    v_z = v0[2]
    next_x_z = (x_z + v_z * dt) * reverse_factor

    if next_x_z < 0.0 and v_z * reverse_factor < -1e-4:
        # Ground Collision
        v_normal = wp.dot(v0, normal) * normal
        v_tao = v0 - v_normal
        v_normal_length = wp.length(v_normal)
        v_tao_length = wp.max(wp.length(v_tao), 1e-6)
        clamp_collide_elas = wp.clamp(collide_elas[0], low=0.0, high=1.0)
        clamp_collide_fric = wp.clamp(collide_fric[0], low=0.0, high=2.0)

        v_normal_new = -clamp_collide_elas * v_normal
        a = wp.max(
            0.0,
            1.0
            - clamp_collide_fric
            * (1.0 + clamp_collide_elas)
            * v_normal_length
            / v_tao_length,
        )
        v_tao_new = a * v_tao

        v1 = v_normal_new + v_tao_new
        toi = -x_z / v_z
    else:
        v1 = v0
        toi = 0.0

    x_new[tid] = x0 + v0 * toi + v1 * (dt - toi)
    v_new[tid] = v1


@wp.kernel(enable_backward=False)
def compute_distances(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    gt_mask: wp.array(dtype=wp.int32),
    num_object_points_single: int,
    num_original_points_single: int,
    distances: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    if gt_mask[i] == 1:
        batch_idx = i // num_original_points_single
        pred_idx = batch_idx * num_object_points_single + j
        dist = wp.length(gt[i] - pred[pred_idx])
        distances[i, j] = dist
    else:
        distances[i, j] = 1e6


@wp.kernel(enable_backward=False)
def compute_neigh_indices(
    distances: wp.array2d(dtype=float),
    neigh_indices: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    min_dist = float(1e6)
    min_index = int(-1)
    for j in range(distances.shape[1]):
        if distances[i, j] < min_dist:
            min_dist = distances[i, j]
            min_index = j
    neigh_indices[i] = min_index


@wp.kernel
def compute_chamfer_loss(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    gt_mask: wp.array(dtype=wp.int32),
    num_valid_per_batch: wp.array(dtype=wp.int32),
    batch_size: int,
    neigh_indices: wp.array(dtype=wp.int32),
    num_object_points_single: int,
    num_original_points_single: int,
    loss_weight: float,
    window_loss_weights: wp.array(dtype=wp.float32),
    chamfer_loss: wp.array(dtype=float),
    chamfer_loss_per_batch: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    if gt_mask[i] == 1:
        batch_idx = i // num_original_points_single
        min_pred = pred[batch_idx * num_object_points_single + neigh_indices[i]]
        min_dist = wp.length(min_pred - gt[i])
        denom = wp.max(num_valid_per_batch[batch_idx], 1)
        final_min_dist_per_batch = loss_weight * min_dist * min_dist / float(denom)
        wp.atomic_add(chamfer_loss_per_batch, batch_idx, final_min_dist_per_batch)
        wp.atomic_add(
            chamfer_loss,
            0,
            window_loss_weights[batch_idx] * final_min_dist_per_batch / float(batch_size),
        )


@wp.kernel
def compute_track_loss(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    gt_mask: wp.array(dtype=wp.int32),
    num_valid_per_batch: wp.array(dtype=wp.int32),
    batch_size: int,
    num_object_points_single: int,
    num_original_points_single: int,
    loss_weight: float,
    window_loss_weights: wp.array(dtype=wp.float32),
    track_loss: wp.array(dtype=float),
    track_loss_per_batch: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    if gt_mask[i] == 1:
        batch_idx = i // num_original_points_single
        local_idx = i - batch_idx * num_original_points_single
        pred_idx = batch_idx * num_object_points_single + local_idx
        # Calculate the smooth l1 loss modifed from fvcore.nn.smooth_l1_loss
        pred_x = pred[pred_idx][0]
        pred_y = pred[pred_idx][1]
        pred_z = pred[pred_idx][2]
        gt_x = gt[i][0]
        gt_y = gt[i][1]
        gt_z = gt[i][2]

        dist_x = wp.abs(pred_x - gt_x)
        dist_y = wp.abs(pred_y - gt_y)
        dist_z = wp.abs(pred_z - gt_z)

        if dist_x < 1.0:
            temp_track_loss_x = 0.5 * (dist_x**2.0)
        else:
            temp_track_loss_x = dist_x - 0.5

        if dist_y < 1.0:
            temp_track_loss_y = 0.5 * (dist_y**2.0)
        else:
            temp_track_loss_y = dist_y - 0.5

        if dist_z < 1.0:
            temp_track_loss_z = 0.5 * (dist_z**2.0)
        else:
            temp_track_loss_z = dist_z - 0.5

        temp_track_loss = temp_track_loss_x + temp_track_loss_y + temp_track_loss_z

        denom = wp.max(num_valid_per_batch[batch_idx], 1)
        average_factor = float(denom) * 3.0

        final_track_loss_per_batch = loss_weight * temp_track_loss / average_factor
        wp.atomic_add(track_loss_per_batch, batch_idx, final_track_loss_per_batch)
        wp.atomic_add(
            track_loss,
            0,
            window_loss_weights[batch_idx] * final_track_loss_per_batch / float(batch_size),
        )


@wp.kernel(enable_backward=False)
def set_int(input: int, output: wp.array(dtype=wp.int32)):
    output[0] = input


@wp.kernel(enable_backward=False)
def update_acc(
    v1: wp.array(dtype=wp.vec3),
    v2: wp.array(dtype=wp.vec3),
    prev_acc: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    prev_acc[tid] = v2[tid] - v1[tid]


@wp.kernel
def compute_acc_loss(
    v1: wp.array(dtype=wp.vec3),
    v2: wp.array(dtype=wp.vec3),
    prev_acc: wp.array(dtype=wp.vec3),
    num_object_points_single: int,
    batch_size: int,
    acc_count: wp.array(dtype=wp.int32),
    acc_weight: float,
    window_loss_weights: wp.array(dtype=wp.float32),
    acc_loss: wp.array(dtype=wp.float32),
    acc_loss_per_batch: wp.array(dtype=wp.float32),
):
    if acc_count[0] == 1:
        # Calculate the smooth l1 loss modifed from fvcore.nn.smooth_l1_loss
        tid = wp.tid()
        cur_acc = v2[tid] - v1[tid]
        cur_x = cur_acc[0]
        cur_y = cur_acc[1]
        cur_z = cur_acc[2]

        prev_x = prev_acc[tid][0]
        prev_y = prev_acc[tid][1]
        prev_z = prev_acc[tid][2]

        dist_x = wp.abs(cur_x - prev_x)
        dist_y = wp.abs(cur_y - prev_y)
        dist_z = wp.abs(cur_z - prev_z)

        if dist_x < 1.0:
            temp_acc_loss_x = 0.5 * (dist_x**2.0)
        else:
            temp_acc_loss_x = dist_x - 0.5

        if dist_y < 1.0:
            temp_acc_loss_y = 0.5 * (dist_y**2.0)
        else:
            temp_acc_loss_y = dist_y - 0.5

        if dist_z < 1.0:
            temp_acc_loss_z = 0.5 * (dist_z**2.0)
        else:
            temp_acc_loss_z = dist_z - 0.5

        temp_acc_loss = temp_acc_loss_x + temp_acc_loss_y + temp_acc_loss_z

        batch_idx = tid // num_object_points_single
        average_factor = float(num_object_points_single) * 3.0
        final_acc_loss_per_batch = acc_weight * temp_acc_loss / average_factor
        wp.atomic_add(acc_loss_per_batch, batch_idx, final_acc_loss_per_batch)
        wp.atomic_add(
            acc_loss,
            0,
            window_loss_weights[batch_idx] * final_acc_loss_per_batch / float(batch_size),
        )


@wp.kernel
def compute_final_loss(
    chamfer_loss: wp.array(dtype=wp.float32),
    track_loss: wp.array(dtype=wp.float32),
    acc_loss: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
):
    loss[0] = chamfer_loss[0] + track_loss[0] + acc_loss[0]


@wp.kernel(enable_backward=False)
def compute_final_loss_per_batch(
    chamfer_loss_per_batch: wp.array(dtype=wp.float32),
    track_loss_per_batch: wp.array(dtype=wp.float32),
    acc_loss_per_batch: wp.array(dtype=wp.float32),
    loss_per_batch: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    loss_per_batch[tid] = (
        chamfer_loss_per_batch[tid]
        + track_loss_per_batch[tid]
        + acc_loss_per_batch[tid]
    )


@wp.kernel
def compute_simple_loss(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    num_object_points_single: int,
    batch_size: int,
    window_loss_weights: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
    loss_per_batch: wp.array(dtype=wp.float32),
):
    # Calculate the smooth l1 loss modifed from fvcore.nn.smooth_l1_loss
    tid = wp.tid()
    pred_x = pred[tid][0]
    pred_y = pred[tid][1]
    pred_z = pred[tid][2]

    gt_x = gt[tid][0]
    gt_y = gt[tid][1]
    gt_z = gt[tid][2]

    dist_x = wp.abs(pred_x - gt_x)
    dist_y = wp.abs(pred_y - gt_y)
    dist_z = wp.abs(pred_z - gt_z)

    if dist_x < 1.0:
        temp_simple_loss_x = 0.5 * (dist_x**2.0)
    else:
        temp_simple_loss_x = dist_x - 0.5

    if dist_y < 1.0:
        temp_simple_loss_y = 0.5 * (dist_y**2.0)
    else:
        temp_simple_loss_y = dist_y - 0.5

    if dist_z < 1.0:
        temp_simple_loss_z = 0.5 * (dist_z**2.0)
    else:
        temp_simple_loss_z = dist_z - 0.5

    temp_simple_loss = temp_simple_loss_x + temp_simple_loss_y + temp_simple_loss_z

    batch_idx = tid // num_object_points_single
    average_factor = float(num_object_points_single) * 3.0
    final_simple_loss_per_batch = temp_simple_loss / average_factor
    wp.atomic_add(loss_per_batch, batch_idx, final_simple_loss_per_batch)
    wp.atomic_add(
        loss,
        0,
        window_loss_weights[batch_idx] * final_simple_loss_per_batch / float(batch_size),
    )


class SpringMassSystemWarp:
    def _build_batched_springs(self, init_springs):
        """Tile single-instance springs to batched global indices."""
        if self.batch_size == 1:
            return init_springs.contiguous()

        springs = init_springs.to(dtype=torch.int64)
        batched = []
        for b in range(self.batch_size):
            s = springs.clone()

            # Endpoint 0
            e0_obj = s[:, 0] < self.num_object_points_single
            s[e0_obj, 0] += b * self.num_object_points_single
            if self.num_control_points_single > 0:
                s[~e0_obj, 0] = (
                    self.num_object_points_total
                    + b * self.num_control_points_single
                    + (s[~e0_obj, 0] - self.num_object_points_single)
                )

            # Endpoint 1
            e1_obj = s[:, 1] < self.num_object_points_single
            s[e1_obj, 1] += b * self.num_object_points_single
            if self.num_control_points_single > 0:
                s[~e1_obj, 1] = (
                    self.num_object_points_total
                    + b * self.num_control_points_single
                    + (s[~e1_obj, 1] - self.num_object_points_single)
                )

            batched.append(s)

        return torch.cat(batched, dim=0).to(dtype=torch.int32, device=init_springs.device)

    def _compute_num_valid_per_batch(self, mask):
        """Compute per-batch valid counts from a flattened [B*N_single] mask."""
        mask_i32 = mask.int().reshape(self.batch_size, self.num_original_points_single)
        return mask_i32.sum(dim=1).to(dtype=torch.int32, device=self.device)

    def __init__(
        self,
        init_vertices,
        init_springs,
        init_rest_lengths,
        init_masses,
        dt,
        num_substeps,
        spring_Y,
        collide_elas,
        collide_fric,
        dashpot_damping,
        drag_damping,
        collide_object_elas=0.7,
        collide_object_fric=0.3,
        init_masks=None,
        collision_dist=0.02,
        init_velocities=None,
        batch_size=1,
        num_object_points_single=None,
        num_control_points_single=None,
        num_original_points_single=None,
        num_surface_points_single=None,
        num_object_points=None,
        num_surface_points=None,
        num_original_points=None,
        controller_points=None,
        reverse_z=False,
        spring_Y_min=1e3,
        spring_Y_max=1e5,
        gt_object_points=None,
        gt_object_visibilities=None,
        gt_object_motions_valid=None,
        loss_weights=None,
        self_collision=False,
        disable_backward=False,
    ):
        logger.info(f"[SIMULATION]: Initialize the Spring-Mass System")
        self.device = cfg.device
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        # Keep both single-instance counts and total (batched) counts.
        if num_object_points is None and num_object_points_single is None:
            raise ValueError(
                "Either num_object_points or num_object_points_single must be provided"
            )
        if num_object_points_single is None:
            if num_object_points % self.batch_size != 0:
                raise ValueError(
                    f"num_object_points ({num_object_points}) is not divisible by "
                    f"batch_size ({self.batch_size})"
                )
            num_object_points_single = num_object_points // self.batch_size
        if num_object_points is None:
            num_object_points = num_object_points_single * self.batch_size
        if num_object_points != num_object_points_single * self.batch_size:
            raise ValueError(
                "num_object_points must equal num_object_points_single * batch_size"
            )

        num_control_points_total = (
            controller_points.shape[1] if controller_points is not None else 0
        )
        if num_control_points_single is None:
            if num_control_points_total == 0:
                num_control_points_single = 0
            else:
                if num_control_points_total % self.batch_size != 0:
                    raise ValueError(
                        f"controller_points.shape[1] ({num_control_points_total}) is not "
                        f"divisible by batch_size ({self.batch_size})"
                    )
                num_control_points_single = (
                    num_control_points_total // self.batch_size
                )
        if num_control_points_total and (
            num_control_points_total
            != num_control_points_single * self.batch_size
        ):
            raise ValueError(
                "controller_points.shape[1] must equal "
                "num_control_points_single * batch_size"
            )

        if num_original_points_single is None and num_original_points is not None:
            if num_original_points % self.batch_size != 0:
                raise ValueError(
                    f"num_original_points ({num_original_points}) is not divisible by "
                    f"batch_size ({self.batch_size})"
                )
            num_original_points_single = num_original_points // self.batch_size
        if num_surface_points_single is None and num_surface_points is not None:
            if num_surface_points % self.batch_size != 0:
                raise ValueError(
                    f"num_surface_points ({num_surface_points}) is not divisible by "
                    f"batch_size ({self.batch_size})"
                )
            num_surface_points_single = num_surface_points // self.batch_size

        # Record the parameters
        self.wp_init_vertices = wp.from_torch(
            init_vertices[:num_object_points].contiguous(),
            dtype=wp.vec3,
            requires_grad=False,
        )
        if init_velocities is None:
            self.wp_init_velocities = wp.zeros_like(
                self.wp_init_vertices, requires_grad=False
            )
        else:
            self.wp_init_velocities = wp.from_torch(
                init_velocities[:num_object_points].contiguous(),
                dtype=wp.vec3,
                requires_grad=False,
            )

        self.n_vertices = init_vertices.shape[0]
        self.n_springs_single = init_springs.shape[0]
        self.n_springs_total = self.n_springs_single * self.batch_size
        # Keep compatibility with existing trainer/checkpoint logic.
        self.n_springs = self.n_springs_single

        self.dt = dt
        self.num_substeps = num_substeps
        self.dashpot_damping = dashpot_damping
        self.drag_damping = drag_damping
        self.reverse_factor = 1.0 if not reverse_z else -1.0
        self.spring_Y_min = spring_Y_min
        self.spring_Y_max = spring_Y_max

        if controller_points is None:
            assert num_object_points == self.n_vertices
        else:
            assert (controller_points.shape[1] + num_object_points) == self.n_vertices
        self.num_object_points_single = int(num_object_points_single)
        self.num_object_points_total = int(num_object_points)
        self.num_object_points = self.num_object_points_total

        self.num_control_points_single = int(num_control_points_single)
        self.num_control_points_total = int(num_control_points_total)
        self.num_control_points = self.num_control_points_total
        self.controller_points = controller_points

        # Deal with the any collision detection
        self.object_collision_flag = 0
        if init_masks is not None:
            if torch.unique(init_masks).shape[0] > 1:
                self.object_collision_flag = 1

        if self_collision:
            assert init_masks is None
            self.object_collision_flag = 1
            # Make each object point collide with all other object points
            # within the same instance.
            init_masks = torch.arange(
                self.num_object_points_single, dtype=torch.int32, device=self.device
            )

        if self.object_collision_flag:
            if init_masks.shape[0] == self.num_object_points_single and self.batch_size > 1:
                init_masks = init_masks.repeat(self.batch_size)
            if init_masks.shape[0] < self.num_object_points_total:
                raise ValueError(
                    f"init_masks length ({init_masks.shape[0]}) is smaller than "
                    f"num_object_points_total ({self.num_object_points_total})"
                )
            self.wp_masks = wp.from_torch(
                init_masks[: self.num_object_points_total].int(),
                dtype=wp.int32,
                requires_grad=False,
            )

            self.collision_grid = wp.HashGrid(128, 128, 128)
            self.collision_dist = collision_dist

            self.wp_collision_indices = wp.zeros(
                (self.wp_init_vertices.shape[0], 500),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_collision_number = wp.zeros(
                (self.wp_init_vertices.shape[0]), dtype=wp.int32, requires_grad=False
            )

        # Initialize the GT for calculating losses
        self.gt_object_points = gt_object_points
        if cfg.data_type == "real":
            self.gt_object_visibilities = gt_object_visibilities.int()
            self.gt_object_motions_valid = gt_object_motions_valid.int()

        self.num_surface_points_single = num_surface_points_single
        self.num_original_points_single = num_original_points_single
        self.num_surface_points = num_surface_points
        self.num_original_points = num_original_points
        if num_original_points is None:
            self.num_original_points = self.num_object_points
        if self.num_original_points_single is None:
            if self.num_original_points % self.batch_size != 0:
                raise ValueError(
                    f"num_original_points ({self.num_original_points}) is not divisible by "
                    f"batch_size ({self.batch_size})"
                )
            self.num_original_points_single = (
                self.num_original_points // self.batch_size
            )
        if self.num_surface_points_single is None and self.num_surface_points is not None:
            if self.num_surface_points % self.batch_size != 0:
                raise ValueError(
                    f"num_surface_points ({self.num_surface_points}) is not divisible by "
                    f"batch_size ({self.batch_size})"
                )
            self.num_surface_points_single = (
                self.num_surface_points // self.batch_size
            )
        self.num_original_points_total = int(self.num_original_points)
        self.num_surface_points_total = (
            int(self.num_surface_points) if self.num_surface_points is not None else None
        )
        self.num_original_points = self.num_original_points_total
        self.num_surface_points = self.num_surface_points_total

        # # Do some initialization to initialize the warp cuda graph
        batched_springs = self._build_batched_springs(init_springs)
        self.wp_springs = wp.from_torch(
            batched_springs, dtype=wp.vec2i, requires_grad=False
        )
        if init_rest_lengths.shape[0] == self.n_springs_total:
            self.use_batched_rest_lengths = 1
            rest_lengths_to_use = init_rest_lengths
            self.n_rest_lengths = self.n_springs_total
        elif init_rest_lengths.shape[0] == self.n_springs_single:
            self.use_batched_rest_lengths = 0
            rest_lengths_to_use = init_rest_lengths[: self.n_springs_single]
            self.n_rest_lengths = self.n_springs_single
        else:
            raise ValueError(
                "init_rest_lengths must have length "
                f"{self.n_springs_single} or {self.n_springs_total}, "
                f"got {init_rest_lengths.shape[0]}"
            )
        self.wp_rest_lengths = wp.from_torch(
            rest_lengths_to_use.contiguous(),
            dtype=wp.float32,
            requires_grad=False,
        )
        if init_masses.shape[0] == self.num_object_points_single and self.batch_size > 1:
            init_masses = init_masses.repeat(self.batch_size)
        if init_masses.shape[0] < self.num_object_points_total:
            raise ValueError(
                f"init_masses length ({init_masses.shape[0]}) is smaller than "
                f"num_object_points_total ({self.num_object_points_total})"
            )
        self.wp_masses = wp.from_torch(
            init_masses[: self.num_object_points_total],
            dtype=wp.float32,
            requires_grad=False,
        )
        if cfg.data_type == "real":
            self.prev_acc = wp.zeros_like(self.wp_init_vertices, requires_grad=False)
            self.acc_count = wp.zeros(1, dtype=wp.int32, requires_grad=False)

        self.wp_current_object_points = wp.from_torch(
            self.gt_object_points[1].clone(), dtype=wp.vec3, requires_grad=False
        )
        if cfg.data_type == "real":
            self.wp_current_object_visibilities = wp.from_torch(
                self.gt_object_visibilities[1].clone(),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_current_object_motions_valid = wp.from_torch(
                self.gt_object_motions_valid[0].clone(),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.num_valid_visibilities_per_batch = self._compute_num_valid_per_batch(
                self.gt_object_visibilities[1]
            )
            self.num_valid_motions_per_batch = self._compute_num_valid_per_batch(
                self.gt_object_motions_valid[0]
            )
            self.wp_num_valid_visibilities = wp.from_torch(
                self.num_valid_visibilities_per_batch.clone(),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_num_valid_motions = wp.from_torch(
                self.num_valid_motions_per_batch.clone(),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.num_valid_visibilities = int(self.gt_object_visibilities[1].sum())
            self.num_valid_motions = int(self.gt_object_motions_valid[0].sum())

            self.wp_original_control_point = wp.from_torch(
                self.controller_points[0].clone(), dtype=wp.vec3, requires_grad=False
            )
            self.wp_target_control_point = wp.from_torch(
                self.controller_points[1].clone(), dtype=wp.vec3, requires_grad=False
            )

            self.chamfer_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.track_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.acc_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.chamfer_loss_per_batch = wp.zeros(
                self.batch_size, dtype=wp.float32, requires_grad=False
            )
            self.track_loss_per_batch = wp.zeros(
                self.batch_size, dtype=wp.float32, requires_grad=False
            )
            self.acc_loss_per_batch = wp.zeros(
                self.batch_size, dtype=wp.float32, requires_grad=False
            )
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.loss_per_batch = wp.zeros(
            self.batch_size, dtype=wp.float32, requires_grad=False
        )
        if loss_weights is None:
            loss_weights = torch.ones(
                self.batch_size, dtype=torch.float32, device=self.device
            )
        else:
            loss_weights = loss_weights.to(
                device=self.device, dtype=torch.float32
            ).contiguous()
            if loss_weights.shape[0] != self.batch_size:
                raise ValueError(
                    f"loss_weights length ({loss_weights.shape[0]}) must match "
                    f"batch_size ({self.batch_size})"
                )
        self.wp_loss_weights = wp.from_torch(
            loss_weights.contiguous(), dtype=wp.float32, requires_grad=False
        )

        # Initialize the warp parameters
        self.wp_states = []
        for i in range(self.num_substeps + 1):
            state = State(self.wp_init_velocities, self.num_control_points)
            self.wp_states.append(state)
        if cfg.data_type == "real":
            if self.num_surface_points_single is None:
                raise ValueError(
                    "num_surface_points_single must be provided for real-data batched loss"
                )
            self.distance_matrix = wp.zeros(
                (self.num_original_points_total, self.num_surface_points_single),
                requires_grad=False,
            )
            self.neigh_indices = wp.zeros(
                (self.num_original_points_total), dtype=wp.int32, requires_grad=False
            )

        # Parameter to be optimized
        self.wp_spring_Y = wp.from_torch(
            torch.log(torch.tensor(spring_Y, dtype=torch.float32, device=self.device))
            * torch.ones(self.n_springs, dtype=torch.float32, device=self.device),
            requires_grad=True,
        )
        self.wp_collide_elas = wp.from_torch(
            torch.tensor([collide_elas], dtype=torch.float32, device=self.device),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_fric = wp.from_torch(
            torch.tensor([collide_fric], dtype=torch.float32, device=self.device),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_object_elas = wp.from_torch(
            torch.tensor(
                [collide_object_elas], dtype=torch.float32, device=self.device
            ),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_object_fric = wp.from_torch(
            torch.tensor(
                [collide_object_fric], dtype=torch.float32, device=self.device
            ),
            requires_grad=cfg.collision_learn,
        )

        # Create the CUDA graph to acclerate
        if cfg.use_graph:
            if cfg.data_type == "real":
                if not disable_backward:
                    with wp.ScopedCapture() as capture:
                        self.tape = wp.Tape()
                        with self.tape:
                            self.step()
                            self.calculate_loss()
                        self.tape.backward(self.loss)
                else:
                    with wp.ScopedCapture() as capture:
                        self.step()
                        self.calculate_loss()
                self.graph = capture.graph
            elif cfg.data_type == "synthetic":
                if not disable_backward:
                    # For synthetic data, we compute simple loss
                    with wp.ScopedCapture() as capture:
                        self.tape = wp.Tape()
                        with self.tape:
                            self.step()
                            self.calculate_simple_loss()
                        self.tape.backward(self.loss)
                else:
                    with wp.ScopedCapture() as capture:
                        self.step()
                        self.calculate_simple_loss()
                self.graph = capture.graph
            else:
                raise NotImplementedError

            with wp.ScopedCapture() as forward_capture:
                self.step()
            self.forward_graph = forward_capture.graph
        else:
            self.tape = wp.Tape()

    def set_controller_target(self, frame_idx, pure_inference=False):
        if self.controller_points is not None:
            # Set the controller points
            wp.launch(
                copy_vec3,
                dim=self.num_control_points_total,
                inputs=[self.controller_points[frame_idx - 1]],
                outputs=[self.wp_original_control_point],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_control_points_total,
                inputs=[self.controller_points[frame_idx]],
                outputs=[self.wp_target_control_point],
            )

        if not pure_inference:
            # Set the target points
            wp.launch(
                copy_vec3,
                dim=self.num_original_points_total,
                inputs=[self.gt_object_points[frame_idx]],
                outputs=[self.wp_current_object_points],
            )

            if cfg.data_type == "real":
                wp.launch(
                    copy_int,
                    dim=self.num_original_points_total,
                    inputs=[self.gt_object_visibilities[frame_idx]],
                    outputs=[self.wp_current_object_visibilities],
                )
                wp.launch(
                    copy_int,
                    dim=self.num_original_points_total,
                    inputs=[self.gt_object_motions_valid[frame_idx - 1]],
                    outputs=[self.wp_current_object_motions_valid],
                )

                self.num_valid_visibilities_per_batch = self._compute_num_valid_per_batch(
                    self.gt_object_visibilities[frame_idx]
                )
                self.num_valid_motions_per_batch = self._compute_num_valid_per_batch(
                    self.gt_object_motions_valid[frame_idx - 1]
                )
                wp.launch(
                    copy_int,
                    dim=self.batch_size,
                    inputs=[self.num_valid_visibilities_per_batch],
                    outputs=[self.wp_num_valid_visibilities],
                )
                wp.launch(
                    copy_int,
                    dim=self.batch_size,
                    inputs=[self.num_valid_motions_per_batch],
                    outputs=[self.wp_num_valid_motions],
                )
                self.num_valid_visibilities = int(
                    self.gt_object_visibilities[frame_idx].sum()
                )
                self.num_valid_motions = int(
                    self.gt_object_motions_valid[frame_idx - 1].sum()
                )

    def set_controller_interactive(
        self, last_controller_interactive, controller_interactive
    ):
        # Set the controller points
        wp.launch(
            copy_vec3,
            dim=self.num_control_points_total,
            inputs=[last_controller_interactive],
            outputs=[self.wp_original_control_point],
        )
        wp.launch(
            copy_vec3,
            dim=self.num_control_points_total,
            inputs=[controller_interactive],
            outputs=[self.wp_target_control_point],
        )

    def set_init_state(self, wp_x, wp_v, pure_inference=False):
        # Detach and clone and set requires_grad=True
        assert (
            self.num_object_points_total == wp_x.shape[0]
            and self.num_object_points_total == self.wp_states[0].wp_x.shape[0]
        )

        if not pure_inference:
            wp.launch(
                copy_vec3,
                dim=self.num_object_points_total,
                inputs=[wp.clone(wp_x, requires_grad=False)],
                outputs=[self.wp_states[0].wp_x],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_object_points_total,
                inputs=[wp.clone(wp_v, requires_grad=False)],
                outputs=[self.wp_states[0].wp_v],
            )
        else:
            wp.launch(
                copy_vec3,
                dim=self.num_object_points_total,
                inputs=[wp_x],
                outputs=[self.wp_states[0].wp_x],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_object_points_total,
                inputs=[wp_v],
                outputs=[self.wp_states[0].wp_v],
            )

    def set_acc_count(self, acc_count):
        if acc_count:
            input = 1
        else:
            input = 0
        wp.launch(
            set_int,
            dim=1,
            inputs=[input],
            outputs=[self.acc_count],
        )

    def update_acc(self):
        wp.launch(
            update_acc,
            dim=self.num_object_points_total,
            inputs=[
                wp.clone(self.wp_states[0].wp_v, requires_grad=False),
                wp.clone(self.wp_states[-1].wp_v, requires_grad=False),
            ],
            outputs=[self.prev_acc],
        )

    def update_collision_graph(self):
        assert self.object_collision_flag
        self.collision_grid.build(self.wp_states[0].wp_x, self.collision_dist * 5.0)
        self.wp_collision_number.zero_()
        wp.launch(
            update_potential_collision,
            dim=self.num_object_points_total,
            inputs=[
                self.wp_states[0].wp_x,
                self.wp_masks,
                self.num_object_points_single,
                self.collision_dist,
                self.collision_grid.id,
            ],
            outputs=[self.wp_collision_indices, self.wp_collision_number],
        )

    def step(self):
        for i in range(self.num_substeps):
            self.wp_states[i].clear_forces()
            if not self.controller_points is None:
                # Set the control point
                wp.launch(
                    set_control_points,
                    dim=self.num_control_points_total,
                    inputs=[
                        self.num_substeps,
                        self.wp_original_control_point,
                        self.wp_target_control_point,
                        i,
                    ],
                    outputs=[self.wp_states[i].wp_control_x],
                )

            # Calculate the spring forces
            wp.launch(
                kernel=eval_springs,
                dim=self.n_springs_total,
                inputs=[
                    self.wp_states[i].wp_x,
                    self.wp_states[i].wp_v,
                    self.wp_states[i].wp_control_x,
                    self.wp_states[i].wp_control_v,
                    self.num_object_points_total,
                    self.wp_springs,
                    self.wp_rest_lengths,
                    self.wp_spring_Y,
                    self.dashpot_damping,
                    self.spring_Y_min,
                    self.spring_Y_max,
                    self.n_springs_single,
                    self.use_batched_rest_lengths,
                ],
                outputs=[self.wp_states[i].wp_vertice_forces],
            )

            if self.object_collision_flag:
                output_v = self.wp_states[i].wp_v_before_collision
            else:
                output_v = self.wp_states[i].wp_v_before_ground

            # Update the output_v using the vertive_forces
            wp.launch(
                kernel=update_vel_from_force,
                dim=self.num_object_points_total,
                inputs=[
                    self.wp_states[i].wp_v,
                    self.wp_states[i].wp_vertice_forces,
                    self.wp_masses,
                    self.dt,
                    self.drag_damping,
                    self.reverse_factor,
                ],
                outputs=[output_v],
            )

            if self.object_collision_flag:
                # Update the wp_v_before_ground based on the collision handling
                wp.launch(
                    kernel=object_collision,
                    dim=self.num_object_points_total,
                    inputs=[
                        self.wp_states[i].wp_x,
                        self.wp_states[i].wp_v_before_collision,
                        self.wp_masses,
                        self.wp_masks,
                        self.wp_collide_object_elas,
                        self.wp_collide_object_fric,
                        self.collision_dist,
                        self.wp_collision_indices,
                        self.wp_collision_number,
                        self.num_object_points_single,
                    ],
                    outputs=[self.wp_states[i].wp_v_before_ground],
                )

            # Update the x and v
            wp.launch(
                kernel=integrate_ground_collision,
                dim=self.num_object_points_total,
                inputs=[
                    self.wp_states[i].wp_x,
                    self.wp_states[i].wp_v_before_ground,
                    self.wp_collide_elas,
                    self.wp_collide_fric,
                    self.dt,
                    self.reverse_factor,
                ],
                outputs=[self.wp_states[i + 1].wp_x, self.wp_states[i + 1].wp_v],
            )

    def calculate_loss(self):
        # Compute the chamfer loss
        # Precompute the distances matrix for the chamfer loss
        wp.launch(
            compute_distances,
            dim=(self.num_original_points_total, self.num_surface_points_single),
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.wp_current_object_visibilities,
                self.num_object_points_single,
                self.num_original_points_single,
            ],
            outputs=[self.distance_matrix],
        )

        wp.launch(
            compute_neigh_indices,
            dim=self.num_original_points_total,
            inputs=[self.distance_matrix],
            outputs=[self.neigh_indices],
        )

        wp.launch(
            compute_chamfer_loss,
            dim=self.num_original_points_total,
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.wp_current_object_visibilities,
                self.wp_num_valid_visibilities,
                self.batch_size,
                self.neigh_indices,
                self.num_object_points_single,
                self.num_original_points_single,
                cfg.chamfer_weight,
                self.wp_loss_weights,
            ],
            outputs=[self.chamfer_loss, self.chamfer_loss_per_batch],
        )

        # Compute the tracking loss
        wp.launch(
            compute_track_loss,
            dim=self.num_original_points_total,
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.wp_current_object_motions_valid,
                self.wp_num_valid_motions,
                self.batch_size,
                self.num_object_points_single,
                self.num_original_points_single,
                cfg.track_weight,
                self.wp_loss_weights,
            ],
            outputs=[self.track_loss, self.track_loss_per_batch],
        )

        wp.launch(
            compute_acc_loss,
            dim=self.num_object_points_total,
            inputs=[
                self.wp_states[0].wp_v,
                self.wp_states[-1].wp_v,
                self.prev_acc,
                self.num_object_points_single,
                self.batch_size,
                self.acc_count,
                cfg.acc_weight,
                self.wp_loss_weights,
            ],
            outputs=[self.acc_loss, self.acc_loss_per_batch],
        )

        wp.launch(
            compute_final_loss_per_batch,
            dim=self.batch_size,
            inputs=[
                self.chamfer_loss_per_batch,
                self.track_loss_per_batch,
                self.acc_loss_per_batch,
            ],
            outputs=[self.loss_per_batch],
        )

        wp.launch(
            compute_final_loss,
            dim=1,
            inputs=[self.chamfer_loss, self.track_loss, self.acc_loss],
            outputs=[self.loss],
        )

    def calculate_simple_loss(self):
        wp.launch(
            compute_simple_loss,
            dim=self.num_object_points_total,
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.num_object_points_single,
                self.batch_size,
                self.wp_loss_weights,
            ],
            outputs=[self.loss, self.loss_per_batch],
        )

    def clear_loss(self):
        if cfg.data_type == "real":
            self.distance_matrix.zero_()
            self.neigh_indices.zero_()
            self.chamfer_loss.zero_()
            self.track_loss.zero_()
            self.acc_loss.zero_()
            self.chamfer_loss_per_batch.zero_()
            self.track_loss_per_batch.zero_()
            self.acc_loss_per_batch.zero_()
        self.loss.zero_()
        self.loss_per_batch.zero_()

    # Functions used to load the parmeters
    def set_spring_Y(self, spring_Y):
        # assert spring_Y.shape[0] == self.n_springs
        wp.launch(
            copy_float,
            dim=self.n_springs,
            inputs=[spring_Y],
            outputs=[self.wp_spring_Y],
        )

    def set_rest_lengths(self, rest_lengths):
        wp.launch(
            copy_float,
            dim=self.n_rest_lengths,
            inputs=[rest_lengths],
            outputs=[self.wp_rest_lengths],
        )

    def set_loss_weights(self, loss_weights):
        wp.launch(
            copy_float,
            dim=self.batch_size,
            inputs=[loss_weights.contiguous()],
            outputs=[self.wp_loss_weights],
        )

    def set_collide(self, collide_elas, collide_fric):
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_elas],
            outputs=[self.wp_collide_elas],
        )
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_fric],
            outputs=[self.wp_collide_fric],
        )

    def set_collide_object(self, collide_object_elas, collide_object_fric):
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_object_elas],
            outputs=[self.wp_collide_object_elas],
        )
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_object_fric],
            outputs=[self.wp_collide_object_fric],
        )
