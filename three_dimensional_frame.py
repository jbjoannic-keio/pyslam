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
import math
import cv2


def draw_text(img, text, pos=(0, 0),
              font=cv2.FONT_HERSHEY_PLAIN,
              font_scale=3,
              text_color=(0, 255, 0),
              font_thickness=2,
              line=2,
              text_color_bg=(0, 0, 0)
              ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1),
                font, font_scale, text_color, font_thickness, line)

    return text_size


class ThreeDimensionalFrame(object):

    def __init__(self, cam, imsize, imframesize=(300, 300, 3)):
        self.cam = cam
        self.cur_pose = None
        self.reference_pose = np.zeros((4, 4))
        self.imframesize = imframesize
        self.imsize = imsize
        self.three_dimensional_frame_image = np.array([])
        self.three_dimensional_grid = np.array([])
        self.initilaizated = False
        self.view = None

        self.original_frame_point = np.array(
            [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1]])
        self.rotation_frame_point = None
        self.pixel_frame_point = None

        self.original_grid_point_wo_translation, self.original_grid_line_direction = self.create_grid(
            -2, 2, 1, 1, -30, 30)
        self.rotation_grid_point = None
        self.pixel_grid_point = None

    def reset_position(self):
        self.reference_pose = self.cur_pose.copy() - np.eye(4)
        return

    def compute_view(self, twc):
        twc = twc - self.reference_pose
        R = twc[0:3, 0:3]
        T = (twc[0:3, 3]).reshape((3, 1))
        Rt = np.transpose(R)
        T_inv = np.dot(Rt, T)
        self.view = np.vstack((np.hstack((Rt, T_inv)), [twc[3, 0:4]]))
        return

    def create_grid(self, x1, x2, y1, y2, z1, z2):
        original = []
        line_direction = []
        for x in range(x1, x2+1):
            for y in range(y1, y2+1):
                if z1 != z2:
                    original.append([x, y, z1, 1])
                    original.append([x, y, z2, 1])
                    line_direction.append(2)
                    line_direction.append(2)
        for y in range(y1, y2+1):
            for z in range(z1, z2+1):
                if x1 != x2:
                    original.append([x1, y, z, 1])
                    original.append([x2, y, z, 1])
                    line_direction.append(0)
                    line_direction.append(0)
        for x in range(x1, x2+1):
            for z in range(z1, z2+1):
                if y1 != y2:
                    original.append([x, y1, z, 1])
                    original.append([x, y2, z, 1])
                    line_direction.append(1)
                    line_direction.append(1)
        return original, line_direction

    def compute_frame_rotation(self, original):
        view_wo_translation = self.view.copy()
        # view_wo_rotation = self.cur_pose.copy()
        view_wo_translation[0:3, 3] = np.zeros((1, 3))
        rotation = np.dot(view_wo_translation, original)

        return rotation

    def compute_grid_rotation(self, original):

        view_wo_translation = self.view.copy()
        # view_wo_rotation = self.cur_pose.copy()
        view_wo_translation[0:3, 3] = np.zeros((1, 3))
        rotation = np.dot(view_wo_translation, original)
        return rotation

    def correct_grid_rotation(self, rotation, line_direction):
        corrected_rotation = rotation.copy().astype(np.float64)
        for i in range(0, corrected_rotation.shape[1], 2):
            if corrected_rotation[2, i] < 0.1 and corrected_rotation[2, i+1] > 0.1:
                # remove points behind camera (z<0) and replace it by point in the same line at z = .1
                ratio = (
                    0.1-corrected_rotation[2, i+1])/(corrected_rotation[2, i]-corrected_rotation[2, i+1])
                new_point = ratio * \
                    (corrected_rotation[:, i]-corrected_rotation[:,
                     i+1]) + corrected_rotation[:, i+1]
                corrected_rotation[:, i] = new_point

            if corrected_rotation[2, i+1] < 0.1 and corrected_rotation[2, i] > 0.1:
                # remove points behind camera (z<0) and replace it by point in the same line at z = .1
                ratio = (
                    0.1-corrected_rotation[2, i])/(corrected_rotation[2, i+1]-corrected_rotation[2, i])
                new_point = ratio * \
                    (corrected_rotation[:, i+1]-corrected_rotation[:, i]
                     ) + corrected_rotation[:, i]
                corrected_rotation[:, i+1] = new_point

            if corrected_rotation[2, i] < 0.1 and corrected_rotation[2, i+1] < 0.1:
                # remove lines behind camera (z<0)
                corrected_rotation[:, i] = np.array([0, 0, 0, 0])
                corrected_rotation[:, i+1] = np.array([0, 0, 0, 0])
        mask = np.where(
            corrected_rotation[3, :] != 0, True, False)
        return corrected_rotation[:, mask],  line_direction[mask]

    def compute_oblique_proj_points(self, rotation):
        proj_matrix = np.array(
            [[0, 100, 100*math.sin(math.pi/4)], [100, 0, 100*math.cos(math.pi/4)]])
        imsize_i = self.imframesize[0]
        imsize_j = self.imframesize[1]
        pixel = np.dot(proj_matrix, rotation[0:3, :])
        pixel += np.array([[imsize_j], [imsize_i]])/2
        return pixel

    def compute_cam_grid_proj(self):
        cam_pixel_grid_points = self.cam.project(
            self.rotation_grid_point[:3, :].T)[0]

        return

    def compute_frame_image(self):
        img = np.zeros(self.imframesize)
        colors = [(255, 0, 0), (0, 100, 0), (0, 0, 255)]
        img = cv2.line(
            img, (0, 0), (self.imframesize[1], self.imframesize[0]), (255, 255, 255), 1)
        img = cv2.line(
            img, (math.floor(self.pixel_frame_point[1, 0]), 0), (math.floor(self.pixel_frame_point[1, 0]), self.imframesize[0]), (255, 255, 255), 1)
        img = cv2.line(
            img, (0, math.floor(self.pixel_frame_point[0, 0])), (self.imframesize[1], math.floor(self.pixel_frame_point[0, 0])), (255, 255, 255), 1)
        for i in range(1, 4):
            img = cv2.line(img, (math.floor(self.pixel_frame_point[1, 0]),
                                 math.floor(self.pixel_frame_point[0, 0])), (math.floor(self.pixel_frame_point[1, i]),
                                                                             math.floor(self.pixel_frame_point[0, i])), colors[i-1], 2)
            img[math.floor(self.pixel_frame_point[0, i]),
                math.floor(self.pixel_frame_point[1, i])] = 255
        img = cv2.flip(img, 0)

        patch_text = np.zeros((90, self.imframesize[1], 3))
        img = np.vstack((patch_text, img))
        draw_text(img, "X = {}".format(round(self.cur_pose[0, 3], 2)),
                  (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, 2, (255, 255, 255))

        draw_text(img, "Y = {}".format(round(self.cur_pose[1, 3], 2)),
                  (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 0), 2, 2, (255, 255, 255))

        draw_text(img, "Z = {}".format(round(self.cur_pose[2, 3], 2)),
                  (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 2, (255, 255, 255))

        self.three_dimensional_frame_image = img
        return

    def compute_grid_image(self):
        img = np.zeros(self.imsize).astype(np.uint8)
        n = np.shape(self.pixel_grid_point)[0]
        colors = [(100, 100, 100), (255, 255, 255), (100, 150, 0)]
        thickness = [1, 1, 2]
        for i in range(0, n, 2):
            cv2.line(img, (math.floor(self.pixel_grid_point[i, 0]), math.floor(self.pixel_grid_point[i, 1])),
                     (math.floor(self.pixel_grid_point[i+1, 0]), math.floor(self.pixel_grid_point[i+1, 1])), colors[self.corrected_grid_line_direction[i]], thickness[self.corrected_grid_line_direction[i]], cv2.LINE_AA)
        # img = cv2.flip(img, 0)

        # img = cv2.dilate(img, np.ones((5, 5)))
        self.three_dimensional_grid = img
        return

    def draw_twc(self, slam):
        map = slam.map

        if map.num_frames() > 0:
            self.cur_pose = map.get_frame(-1).Twc.copy()
            self.initilaizated = True

        if self.initilaizated:
            self.compute_view(self.cur_pose)
            self.rotation_frame_point = self.compute_frame_rotation(
                self.original_frame_point)
            self.pixel_frame_point = self.compute_oblique_proj_points(
                self.rotation_frame_point)
            self.compute_frame_image()

            # little translation to the grid as if it also moves
            self.original_grid_point = self.original_grid_point_wo_translation - \
                np.hstack((self.cur_pose[0, 3] %
                          1, 0, self.cur_pose[2, 3] % 1, 0))
            #self.original_grid_point = self.original_grid_point_wo_translation
            self.original_grid_point = np.array(self.original_grid_point).T
            self.original_grid_line_direction = np.array(
                self.original_grid_line_direction)

            self.rotation_grid_point = self.compute_grid_rotation(
                self.original_grid_point)
            print("rotation")
            self.corrected_rotation_grid_point, self.corrected_grid_line_direction = self.correct_grid_rotation(
                self.rotation_grid_point, self.original_grid_line_direction)
            print("corrected")
            self.pixel_grid_point = self.cam.project(
                self.corrected_rotation_grid_point[:3, :].T)[0]

            self.compute_grid_image()
        return self.three_dimensional_frame_image, self.three_dimensional_grid
