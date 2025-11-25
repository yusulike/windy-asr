"""
UI Widgets Module
Contains reusable UI components
"""

import logging
import numpy as np
import customtkinter as ctk

logger = logging.getLogger('windy_asr.ui.widgets')


class AudioLevelDisplay:
    """오디오 레벨(좌/우 채널) 시각화 위젯"""

    def __init__(self, parent, width=400, height=60):
        self.width = width
        self.height = height

        self.frame = ctk.CTkFrame(parent, width=width, height=height, fg_color="#1a1a1a")

        ctk.CTkLabel(self.frame, text="L:", text_color="#888888").pack(side="left", padx=5, pady=5)
        self.progress_l = ctk.CTkProgressBar(self.frame, width=150, progress_color="#00ff00")
        self.progress_l.pack(side="left", padx=3, pady=5)
        self.label_l = ctk.CTkLabel(self.frame, text="0%", text_color="#888888", width=40)
        self.label_l.pack(side="left", padx=3, pady=5)

        ctk.CTkLabel(self.frame, text="R:", text_color="#888888").pack(side="left", padx=5, pady=5)
        self.progress_r = ctk.CTkProgressBar(self.frame, width=150, progress_color="#00ff00")
        self.progress_r.pack(side="left", padx=3, pady=5)
        self.label_r = ctk.CTkLabel(self.frame, text="0%", text_color="#888888", width=40)
        self.label_r.pack(side="left", padx=3, pady=5)

    def get_widget(self):
        return self.frame

    def update_level(self, audio_data):
        try:
            if audio_data.ndim > 1:
                level_l = np.sqrt(np.mean(audio_data[:, 0] ** 2))
                if audio_data.shape[1] > 1:
                    level_r = np.sqrt(np.mean(audio_data[:, 1] ** 2))
                else:
                    level_r = level_l
            else:
                level_l = level_r = np.sqrt(np.mean(audio_data ** 2))

            level_l = np.log10(max(level_l, 0.001)) + 2
            level_r = np.log10(max(level_r, 0.001)) + 2

            lvl_l_display = max(0, min(level_l / 4, 1.0))
            lvl_r_display = max(0, min(level_r / 4, 1.0))

            self.progress_l.set(lvl_l_display)
            self.progress_r.set(lvl_r_display)
            self.label_l.configure(text=f"{int(lvl_l_display * 100)}%")
            self.label_r.configure(text=f"{int(lvl_r_display * 100)}%")
        except Exception as exc:
            logger.debug("Audio level update skipped: %s", exc)

    def reset(self):
        self.progress_l.set(0)
        self.progress_r.set(0)
        self.label_l.configure(text="0%")
        self.label_r.configure(text="0%")
