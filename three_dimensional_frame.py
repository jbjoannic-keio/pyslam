"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.



* Class by Jean-Baptiste JOANNIC <jean-baptiste.joannic@keio.jp>
"""

import numpy as np

import cv2


class ThreeDimensionalFrame(object):

    def __init__(self, imsize=(4, 4)):
        self.cur_pose = None
        self.reference_pose = None
        self.imsize = imsize
        self.three_dimensional_image = None
        self.initilaizated = False

    def compute_img(self, twc, imsize):
        print("\n\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", twc)
        cv2.imshow("compute twc to repere", twc)
        return twc

    def draw_twc(self, slam):
        map = slam.map

        if map.num_frames() > 0:
            self.cur_pose = map.get_frame(-1).Twc.copy()
            self.initilaizated = True

        if slam.tracking.kf_ref is not None:
            self.reference_pose = slam.tracking.kf_ref.Twc.copy()

        if self.initilaizated:
            self.three_dimensional_image = self.compute_img(
                self.cur_pose, self.imsize)

        return self.three_dimensional_image
