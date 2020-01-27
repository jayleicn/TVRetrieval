# MIT License
#
# Copyright (c) 2018 Victor Escorcia Castillo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
"""
Group multiple methods to generate salient temporal windows in a video"""
import itertools
import numpy as np

PROPOSAL_SCHEMES = ['DidemoICCV17SS', 'SlidingWindowMSRSS']


class TemporalProposalsBase:
    """Base class (signature) to generate temporal candidate in video"""
    def __call__(self, video_id, metadata=None, feature_collection=None):
        raise NotImplementedError('Implement with the signature above')


class DidemoICCV17SS(TemporalProposalsBase):
    """Original search space of moments proposed in ICCV-2017

    Attributes:
        clip_length_min (float) : minimum length, in seconds, of a video clip.
        proposals (numpy array) : of shape [21, 2] representing all the
            possible temporal segments of valid annotations of DiDeMo dataset.
            It represents the search space of a temporal localization
            algorithm.

    Reference: Hendricks et al. Localizing Moments in Video with Natural
        Language. ICCV 2017.
    """
    clip_length_min = 5.0

    def __init__(self, *args, dtype=np.float32, **kwargs):
        clips_indices = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
        for i in itertools.combinations(range(len(clips_indices)), 2):
            clips_indices.append(i)
        self.proposals = np.array(clips_indices, dtype=dtype)
        self.proposals *= self.clip_length_min
        self.proposals[:, 1] += self.clip_length_min

    def __call__(self, *args, **kwargs):
        return self.proposals


class SlidingWindowMSRSS(TemporalProposalsBase):
    """Multi-scale sliding window with relative stride within the same scale

    Attributes:
        length (float) : length of smallest window.
        scales (sequence of int) : duration of moments relative to
            `length`.
        stride (float) : relative stride between two windows with the same
            duration. We used different strides for each scale rounding it
            towards a multiple of `length`. Note that the minimum stride is
            `length` for any window will be the `length` itself.
        dtype (numpy.dtype) :
    """

    def __init__(self, length, scales, stride=0.5, round_base=0.5, dtype=np.float32):
        self.length = length
        self.scales = scales
        self.round_base = round_base
        self.relative_stride = stride
        # pick strides per scale that are multiples of length
        self.strides = [max(round(s * stride / round_base) * round_base, round_base)
                        * length for s in scales]
        self.dtype = dtype
        assert len(scales) > 0

    def sliding_windows(self, t_end, t_start=0):
        """sliding canonical windows over a given time interval"""
        windows_ = []
        for i, stride in enumerate(self.strides):
            num_i = np.ceil((t_end - t_start) / stride)
            windows_i = np.empty((int(num_i), 2), dtype=np.float32)
            windows_i[:, 0] = np.arange(t_start, t_end, stride)
            windows_i[:, 1] = windows_i[:, 0] + self.length * self.scales[i]
            windows_i[windows_i[:, 1] > t_end, 1] = t_end
            windows_.append(windows_i)
            # print("--------------------------------{}".format(i))
            # print(windows_i)
        # import sys
        # sys.exit(1)
        windows = np.concatenate(windows_, axis=0)
        # Hacky way to make windows fit inside video
        # It implies windows at the end may not belong to the set spanned by
        # length and scales.
        return np.unique(windows, axis=0)

    def __call__(self, video_id, metadata=None, feature_collection=None):
        """return: (N_window, 2), each row contains (start, end)"""
        duration = metadata.get('duration')
        assert duration is not None
        return self.sliding_windows(duration)


ProposalConfigs = {
    "didemo": {
        "proposal_interface": "DidemoICCV17SS",
        "clip_length": 2.5,
    },
    "tvr": {
        "length": 3,  # min proposal length
        "scales": [1, 2, 4, 8],
        "stride": 0.3,
        "round_base": 1,
        "min_proposal_length": 3,  # length * min(scales)
        "clip_length": 1.5,  # length should be divisible by clip_length
        "proposal_interface": "SlidingWindowMSRSS",
    },
    "anet_cap": {
        "length": 5,
        "scales": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26],
        "stride": 0.3,
        "round_base": 1,
        "min_proposal_length": 10,  # length * min(scales)
        "clip_length": 5,  # length * min(scales) / 2
        "proposal_interface": "SlidingWindowMSRSS",
    },
    "charades_sta": {
        "length": 3,
        "scales": [2, 3, 4, 5, 6, 7, 8],
        "stride": 0.3,
        "round_base": 1,
        "min_proposal_length": 6,  # length * min(scales)
        "clip_length": 3,  # length * min(scales) / 2
        "proposal_interface": "SlidingWindowMSRSS",
    },
    "profiling": {
        "length": 5,
        "scales": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "stride": 0.3,
        "round_base": 1,
        "clip_length": 5,  # length * min(scales) / 2
        "proposal_interface": "SlidingWindowMSRSS",
    },
}
"""
'min_clip_length' is used to uniformly segment the video into smaller clips, it is a half of
the 'min_proposal_length'. Thus we can enforce each moment has at least 2 clips.
"""


def get_proposal_interface(dset_name):
    """ dset_name (str): one of ["tvr"] """
    assert dset_name in ProposalConfigs
    if dset_name == "didemo":
        return DidemoICCV17SS()
    else:
        arg_names = ["length", "scales", "stride", "round_base"]
        func_args = {k: ProposalConfigs[dset_name][k] for k in arg_names}
        return SlidingWindowMSRSS(**func_args)


if __name__ == '__main__':
    test_fns_args = [(DidemoICCV17SS, (),),
                     (SlidingWindowMSRSS, (1.5, [2, 4, 6, 12]))]
    for fn_i, args_i in test_fns_args:
        proposal_fn = fn_i(*args_i)
        x = proposal_fn('hola', {'duration': 15})
        if fn_i == DidemoICCV17SS:
            assert len(x) == 21
