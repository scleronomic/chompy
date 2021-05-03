import numpy as np
from wzk import element_at_depth


def next_frame_idx2influence_frames_frames(nfi):
    n_frames = len(nfi)
    assert nfi[-1] == -1
    iff = np.eye(n_frames, dtype=bool)  # influence frame-> frame
    for frame in range(n_frames - 2, -1, -1):
        next_frame = nfi[frame]
        if isinstance(next_frame, (list, tuple)):
            for nf in next_frame:
                iff[frame] += iff[nf]

        elif next_frame != -1:
            iff[frame] += iff[next_frame]

    return iff


def influence_frames_frames2joints_frames(*, jfi, iff=None, nfi=None):
    n_dof = len(jfi)
    if iff is None:
        assert nfi is not None
        iff = next_frame_idx2influence_frames_frames(nfi)

    n_frames = iff.shape[0]

    ijf = np.zeros((n_dof, n_frames), dtype=bool)
    for joint in range(n_dof):
        frame = jfi[joint]
        if isinstance(frame, (list, tuple)):
            for f in frame:
                ijf[joint] += iff[f]
        else:
            ijf[joint] += iff[frame]

    return ijf


def __get_joint_frame_indices_first_last(jfi):
    jfi_first = np.array([i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in jfi])
    jfi_last = np.array([i[-1] if isinstance(i, (list, tuple, np.ndarray)) else i for i in jfi])
    return jfi, jfi_first, jfi_last


def prev2next_frame_idx(pfi):
    nfi = np.zeros_like(pfi, dtype=object)
    for i in range(len(pfi)):
        where_i = (np.nonzero(pfi == i)[0])
        if len(where_i) == 0:
            nfi[i] = -1
        elif len(where_i) == 1:
            nfi[i] = where_i[0]
        else:  # len(where_i) > 1
            nfi[i] = where_i.tolist()

    return nfi


def next2prev_frame_idx(nfi):
    nfi = nfi.tolist()
    nfi_level1 = element_at_depth(nfi, depth=1, with_index=True)
    pfi = np.zeros(len(nfi), dtype=int)
    pfi[0] = -1

    for i in range(1, len(pfi)):
        try:
            pfi[i] = nfi.index(i)
        except ValueError:
            for j, nfi_l1 in nfi_level1:
                if i in nfi_l1:
                    pfi[i] = j[0]
                    continue

    return pfi


def in_kinematic_chain(jf_influence, frame_idx):
    return jf_influence[:, frame_idx].sum(axis=-1) != 0


def not_in_kinematic_chain(influence_jf, frame_idx):
    return np.logical_not(in_kinematic_chain(jf_influence=influence_jf, frame_idx=frame_idx))


def num_in_kinematic_chain(influence_jf, frame_idx):
    return sum(in_kinematic_chain(jf_influence=influence_jf, frame_idx=frame_idx))


def replace_outside_kinematic_chain(*, q, q2=None, in_kc):
    n_dof = len(in_kc)
    n_samples_q, n_wp, n_dof_q = q.shape

    if q2 is None:
        if n_dof_q < n_dof:
            q_new = np.zeros((n_samples_q, n_wp, n_dof))
            q_new[:, :, in_kc] = q
        else:
            q_new = q
    else:
        q_new = q2.repeat(n_samples_q // q2.shape[0], axis=0)
        if n_dof_q < n_dof:
            q_new[:, :, in_kc] = q
        else:
            q_new[:, :, in_kc] = q[:, :, in_kc]

    return q_new
