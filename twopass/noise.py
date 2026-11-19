"""Discrete solver-query noise levels used for training.

Training samples the noise levels that the UniPC sampler visits at 4 or 6
function evaluations: ten (level, timestep) slots in total, drawn uniformly, so
the 4-call schedule is chosen with probability 0.4 and the 6-call schedule with
probability 0.6.
"""
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

NFE_SCHEDULES = (4, 6)


def build_scheduler(nfe, device, shift):
    """UniPC flow-matching scheduler configured for `nfe` solver queries."""
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)
    scheduler.set_timesteps(nfe, device=device, shift=shift)
    return scheduler


class QueryGrid:
    """Noise levels and model timesteps queried by the 4- and 6-call samplers."""

    def __init__(self, device, shift):
        self.sigmas = {}
        self.timesteps = {}
        for nfe in NFE_SCHEDULES:
            scheduler = build_scheduler(nfe, device, shift)
            self.sigmas[nfe] = scheduler.sigmas[:-1].to(device)
            self.timesteps[nfe] = scheduler.timesteps.to(device)
            assert len(self.sigmas[nfe]) == nfe and float(self.sigmas[nfe][0]) > .99
            assert bool((self.sigmas[nfe] > 0).all())

    def select(self, uniform_draw):
        """Map a uniform draw in [0, 1) to (nfe, index) over the ten slots."""
        slot = int((uniform_draw * 10).floor().clamp(max=9))
        return (4, slot) if slot < 4 else (6, slot - 4)
