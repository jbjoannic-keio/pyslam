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
"""

#import pygame
from pygame.locals import DOUBLEBUF
import cv2
import pygame
import numpy as np
import math
print("importing pygame")
print("initialising pygame")
pygame.init()


class Display2D(object):
    def __init__(self, W, H, is_BGR=True):
        pygame.init()
        pygame.display.set_caption('Camera')
        self.screen = pygame.display.set_mode((W, H), DOUBLEBUF)
        self.surface = pygame.Surface(self.screen.get_size()).convert()
        self.is_BGR = is_BGR
        self.first_redim = True
        self.result = cv2.VideoWriter(
            "./videos/result.mp4", cv2.VideoWriter_fourcc(*'MJPG'), 10, (W, H))
        self.result2 = cv2.VideoWriter(
            "./videos/result2.mp4", cv2.VideoWriter_fourcc(*'MJPG'), 10, (W, H))

    def quit(self):
        self.result.release()
        self.result2.release()
        pygame.display.quit()
        pygame.quit()

    def draw(self, img, imraw, three_dimensional_img, three_dimensional_grid):
        if three_dimensional_img.shape[0] > 0:
            img_partial_fused = cv2.add(imraw, three_dimensional_grid)
            cv2.imshow("partial_fused", img_partial_fused)
            self.result.write(img_partial_fused)
            img = cv2.add(img, three_dimensional_grid)
            n = np.shape(img)[0]
            height, length = np.shape(three_dimensional_img)[:2]
            ratio = length/height
            self.result2.write(img)
            three_dimensional_img = cv2.resize(
                three_dimensional_img, (math.floor(ratio*n), n))
            img = np.hstack((img, three_dimensional_img))
            if self.first_redim:
                H, W = np.shape(img)[:2]
                self.screen = pygame.display.set_mode((W, H), DOUBLEBUF)
                self.surface = pygame.Surface(self.screen.get_size()).convert()
                self.first_redim = False
        # junk
        for event in pygame.event.get():
            pass

        if self.is_BGR:
            # draw BGR
            pygame.surfarray.blit_array(
                self.surface, img.swapaxes(0, 1)[:, :, [2, 1, 0]])
        else:
            # draw RGB, not BGR
            pygame.surfarray.blit_array(
                self.surface, img.swapaxes(0, 1)[:, :, [0, 1, 2]])

        self.screen.blit(self.surface, (0, 0))

        # blit
        pygame.display.flip()
