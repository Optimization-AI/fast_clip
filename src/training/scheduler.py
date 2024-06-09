import numpy as np


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def const_lr(param_groups, base_lr, warmup_length, steps):
    if isinstance(base_lr, list):
        assert len(param_groups) == len(base_lr)
        base_lr_list = base_lr
    else:
        base_lr_list = [base_lr for _ in range(len(param_groups))]
    def _lr_adjuster(step):
        for i, param_group in enumerate(param_groups):
            base_lr = base_lr_list[i]
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                lr = base_lr
            param_group["lr"] = lr
    return _lr_adjuster


def const_lr_cooldown(param_groups, base_lr, warmup_length, steps, cooldown_steps, cooldown_power=1.0, cooldown_end_lr=0.):
    if isinstance(base_lr, list):
        assert len(param_groups) == len(base_lr)
        base_lr_list = base_lr
    else:
        base_lr_list = [base_lr for _ in range(len(param_groups))]
    def _lr_adjuster(step):
        for i, param_group in enumerate(param_groups):
            base_lr = base_lr_list[i]
            start_cooldown_step = steps - cooldown_steps
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                if step < start_cooldown_step:
                    lr = base_lr
                else:
                    e = step - start_cooldown_step
                    es = steps - start_cooldown_step
                    # linear decay if power == 1; polynomial decay otherwise;
                    decay = (1 - (e/es)) ** cooldown_power
                    lr = decay * (base_lr - cooldown_end_lr) + cooldown_end_lr
            param_group["lr"] = lr
    return _lr_adjuster


def cosine_lr(param_groups, base_lr, warmup_length, steps, lr_min=0.0):
    if isinstance(base_lr, list):
        assert len(param_groups) == len(base_lr)
        base_lr_list = base_lr
    else:
        base_lr_list = [base_lr for _ in range(len(param_groups))]
    def _lr_adjuster(step):
        for i, param_group in enumerate(param_groups):
            base_lr = base_lr_list[i]
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * (base_lr - lr_min) + lr_min
            param_group["lr"] = lr
    return _lr_adjuster


def step_lr_thresh(param_groups, base_lr, warmup_length, thresh_list, ratio_list, model):
    if isinstance(base_lr, list):
        assert len(param_groups) == len(base_lr)
        base_lr_list = base_lr
    else:
        base_lr_list = [base_lr for _ in range(len(param_groups))]
    def _lr_adjuster(step):
        for i, param_group in enumerate(param_groups):
            base_lr = base_lr_list[i]
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                lr = base_lr
                for thresh, ratio in zip(thresh_list, ratio_list):
                    if 1.0 / model.logit_scale.exp() <= thresh:
                        lr = base_lr * ratio
                    else:
                        break
            param_group["lr"] = lr
    return _lr_adjuster
