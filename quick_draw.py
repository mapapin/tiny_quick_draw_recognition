import os
import threading

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from argparse import ArgumentParser  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

import numpy as np  # noqa: E402
import pygame  # noqa: E402
from PIL import Image  # noqa: E402
from tensorflow import argmax, reduce_max  # noqa: E402

from common import get_config, generate_model  # noqa: E402
from schemas import Config  # noqa: E402


class ModelManager:
    def __init__(self, model_path: Path, config: Config):
        self.model_path = model_path
        self.config = config
        self._load_model()

    def _load_model(self):
        self.model = generate_model(self.config, self.model_path)

    def _get_drawing_zone(self, img: np.ndarray) -> np.ndarray:
        non_white_indices = np.argwhere(img != 255)

        xmin, ymin = np.min(non_white_indices, axis=0)
        xmax, ymax = np.max(non_white_indices, axis=0)

        side_length = max(xmax - xmin, ymax - ymin)

        xcenter = (xmin + xmax) / 2
        ycenter = (ymin + ymax) / 2

        xmin = max(0, int(xcenter - side_length / 2))
        ymin = max(0, int(ycenter - side_length / 2))

        xmax = xmin + side_length
        ymax = ymin + side_length

        padding = 20
        xmin, ymin, xmax, ymax = self._ensure_within_bounds(img.shape, xmin, ymin, xmax, ymax, padding, side_length)

        cropped_square = img[xmin:xmax, ymin:ymax]
        return cropped_square

    def _ensure_within_bounds(
        self, img_shape: tuple, xmin: int, ymin: int, xmax: int, ymax: int, padding: int, side_length: int
    ) -> tuple:
        if xmax >= img_shape[0]:
            xmax = img_shape[0] - 1
            xmin = max(0, xmax - side_length - 2 * padding)
        if ymax >= img_shape[1]:
            ymax = img_shape[1] - 1
            ymin = max(0, ymax - side_length - 2 * padding)

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        return xmin, ymin, xmax, ymax

    def predict(self, input: np.ndarray) -> tuple[str, float]:
        input_cropped = self._get_drawing_zone(input.T)
        img = Image.fromarray(input_cropped).resize((28, 28))
        arr = np.array(img)
        arr = np.expand_dims(arr, 0)

        arr = arr // 255
        threshold = 0.9
        arr = np.where(arr <= threshold, 0, 1)

        output = self.model.predict(arr, verbose=0)

        output_idx = argmax(output, axis=1)
        output_idx = reduce_max(output_idx)
        prob = output[:, output_idx][0]

        return config.data.classes[output_idx], prob * 100


class DrawingPredictor:
    def __init__(self, model_path: Path, config: Config):
        self.win_h = 640
        self.win_w = 640
        self.win_size = (self.win_w, self.win_h)
        self.background = (255, 255, 255)
        self.brush_size = 3
        self.pixels = np.ones((self.win_h, self.win_w), dtype=np.int16) * 255
        self.last: Optional[np.ndarray] = None
        self.drawing = False
        self.model_path = model_path
        self.predicted_class = ""
        self.probability = 0.0
        self.show_help = False
        self.config = config

        pygame.init()
        pygame.display.set_caption("Tiny Quick Draw")
        self.screen = pygame.display.set_mode(self.win_size)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

    def start(self):
        self.loading = True
        threading.Thread(target=self._load_model).start()
        self._run_loading_screen()

    def _load_model(self) -> None:
        self.model_manager = ModelManager(self.model_path, self.config)
        self.loading = False

    def _run_loading_screen(self) -> None:
        font = pygame.font.Font(None, 36)
        text = font.render("Loading...", True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.win_w / 2, self.win_h / 2))

        while self.loading:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            self.screen.fill((0, 0, 0))
            self.screen.blit(text, text_rect)
            pygame.display.flip()

        self.run()

    def _draw_line(self, current: np.ndarray) -> np.ndarray:
        if self.last is not None:
            y1, x1 = self.last
            y2, x2 = current

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1

            err = dx - dy

            while x1 != x2 or y1 != y2:
                x_low, x_high = max(x1 - self.brush_size, 0), min(x1 + self.brush_size, self.win_w)
                y_low, y_high = max(y1 - self.brush_size, 0), min(y1 + self.brush_size, self.win_h)
                self.pixels[y_low:y_high, x_low:x_high] = 0

                e2 = err * 2

                if e2 > -dy:
                    err -= dy
                    x1 += sx
                if e2 < dx:
                    err += dx
                    y1 += sy

        return current

    def _print_probs(self, percentage: float) -> str:
        red = 255 - int(255 * percentage / 100)
        green = int(255 * percentage / 100)
        color = f"\033[38;2;{red};{green};0m"
        reset = "\033[0m"
        return color + f"{percentage:.2f}%" + reset

    def _print_prediction_in_box(self, class_name: str, percentage: float) -> None:
        box_line = "━" * 28
        print("┏" + box_line + "┓")
        print(f"┃ Prediction: {class_name:<15}┃")
        percentage_str = self._print_probs(percentage)
        print(f"┃ Confidence: {percentage_str}" + " " * (14 - len(f"{percentage:.2f}")) + "┃")
        print("┗" + box_line + "┛")

    def _draw(self) -> None:
        x, y = pygame.mouse.get_pos()
        if 0 <= x < self.win_w and 0 <= y < self.win_h:
            mouse_pos = np.array((x, y), dtype=np.int16)
            if self.last is not None:
                self.last = self._draw_line(mouse_pos)
            else:
                x_low, x_high = max(x - self.brush_size, 0), min(x + self.brush_size, self.win_w)
                y_low, y_high = max(y - self.brush_size, 0), min(y + self.brush_size, self.win_h)
                self.pixels[x_low:x_high, y_low:y_high] = 0
                self.last = mouse_pos

    def _check_key_events(self, event: pygame.event.EventType) -> None:
        if event.key == pygame.K_ESCAPE:
            pygame.quit()
            exit()

        elif event.key == pygame.K_u:
            self.brush_size += 1

        elif event.key == pygame.K_d:
            self.brush_size -= 1

        elif event.key == pygame.K_r:
            self.pixels = np.ones((self.win_w, self.win_h), dtype=np.int16) * 255
            self.predicted_class = ""
            self.probability = 0.0

    def _is_help_button_collide(self, position: tuple[int, int]) -> bool:
        button_rect = pygame.Rect(self.win_w - 60, 0, 60, 40)
        return button_rect.collidepoint(position)

    def _check_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEMOTION:
                if self.drawing:
                    self._draw()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self._is_help_button_collide(event.pos):
                    self.show_help = not self.show_help
                else:
                    self.drawing = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if not self._is_help_button_collide(event.pos):
                    self.drawing = False
                    self.last = None
                    if self.pixels[self.pixels != 255].shape[0] > 0:
                        (
                            self.predicted_class,
                            self.probability,
                        ) = self.model_manager.predict(self.pixels)

            elif event.type == pygame.QUIT:
                pygame.quit()
                exit()

            elif event.type == pygame.KEYDOWN:
                self._check_key_events(event)

    def _display_prediction(self) -> None:
        font = pygame.font.Font(None, 36)
        msg = "Draw !" if not self.predicted_class else f"{self.predicted_class} {self.probability:.2f}%"
        text = font.render(msg, True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (self.win_w // 2, 20)
        self.screen.blit(text, text_rect)

    def _display_help(self) -> None:
        overlay = pygame.Surface((self.win_w, self.win_h))
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        texts = [
            "[R] to clear",
            "[U] to increase brush size",
            "[D] to decrease brush size",
        ]

        for i, text in enumerate(texts):
            rendered_text = self.font.render(text, True, (255, 255, 255))
            text_rect = rendered_text.get_rect()
            text_rect.topleft = (
                2 * self.win_w // 10,
                self.win_h // 2 - (40 * len(texts) // 2) + i * 40,
            )
            self.screen.blit(rendered_text, text_rect)

        self._draw_help_button("X", (255, 255, 255))

    def _draw_help_button(self, char: str, color: tuple[int, int, int]) -> None:
        x_text = self.font.render(char, True, color)
        x_rect = x_text.get_rect(center=(self.win_w - 30, 20))
        self.screen.blit(x_text, x_rect)

    def run(self) -> None:
        while True:
            self._check_events()

            self.screen.fill(self.background)

            if self.show_help:
                self._display_help()
            else:
                surface = pygame.surfarray.make_surface(self.pixels)
                self.screen.blit(surface, (0, 0))

                self._display_prediction()

                self._draw_help_button("?", (0, 0, 0))

            self.clock.tick(60)
            pygame.display.flip()


if __name__ == "__main__":
    parser = ArgumentParser(prog="Tiny Quick Draw")
    parser.add_argument("model_path", nargs="?", default="best_model.keras", type=Path)
    parser.add_argument("config_path", nargs="?", default="config.yaml", type=Path)

    args = parser.parse_args()

    config = get_config(args.config_path)

    predictor = DrawingPredictor(args.model_path, config)
    predictor.start()
