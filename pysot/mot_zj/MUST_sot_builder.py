# Copyright (c) SenseTime. All Rights Reserved.
# modified by Yinqiang Zhang 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.core.config import cfg
from pysot.tracker.siamrpn_tracker import SiamRPNTracker
from pysot.tracker.siammask_tracker import SiamMaskTracker
from pysot.tracker.siamrpnlt_tracker import SiamRPNLTTracker
from mot_zj.MUST_sot_tracker import MUSTTracker

TRACKS = {
          'SiamRPNTracker': SiamRPNTracker,
          'SiamMaskTracker': SiamMaskTracker,
          'SiamRPNLTTracker': SiamRPNLTTracker,
          'MUSTTracker': MUSTTracker
         }


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)
