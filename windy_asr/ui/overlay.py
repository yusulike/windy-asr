"""
Subtitle Overlay Module
Canvas-based subtitle overlay window with shadow effects and scroll animations
"""

import logging
import time
import tkinter as tk
from collections import deque
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageTk

logger = logging.getLogger('windy_asr.ui.overlay')


class SubtitleOverlay(tk.Toplevel):
    """자막 오버레이 창 - Canvas 기반 그림자 효과 + 스크롤 애니메이션"""
    
    def __init__(self, master=None, compact=False):
        super().__init__(master)
        
        # Compact 모드 플래그
        self.compact_mode = compact
        
        # 창 설정
        self.title("Subtitle Overlay")
        self.attributes('-topmost', True)
        self.attributes('-alpha', 0.5)
        self.overrideredirect(True)

        # 윈도우 배경을 특정 색으로 통일한 뒤 해당 색상을 투명 처리하여 둥근 모서리가 제대로 보이도록 함
        self.transparent_color = "#010101"
        self.configure(bg=self.transparent_color)
        try:
            self.wm_attributes('-transparentcolor', self.transparent_color)
        except tk.TclError:
            logger.warning("Transparent color attribute not supported on this platform")
        
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        # Compact 모드에서는 더 작은 크기
        if self.compact_mode:
            self.window_width = 1600
            self.window_height = 120  # 1줄만 표시
        else:
            self.window_width = 1900
            self.window_height = 300  # 높이 증가 (3줄 자막에 맞춤)
        x = (screen_width - self.window_width) // 2
        y = screen_height - self.window_height - 100
        
        self.geometry(f"{self.window_width}x{self.window_height}+{x}+{y}")
        
        # 메인 Canvas
        self.canvas = tk.Canvas(
            self,
            width=self.window_width,
            height=self.window_height,
            bg=self.transparent_color,
            highlightthickness=0
        )
        self.canvas.pack(fill='both', expand=True)
        
        # 둥근 배경 그리기
        self.bg_radius = 60
        self.bg_rect = self._create_rounded_rectangle(
            0, 0, self.window_width, self.window_height,
            radius=self.bg_radius,
            fill='black'
        )
        
        # 드래그 및 메뉴 바인딩
        self.canvas.bind('<Button-1>', self.start_move)
        self.canvas.bind('<B1-Motion>', self.on_move)
        self.canvas.bind('<Button-3>', self.show_menu)
        
        # 자막 데이터
        if self.compact_mode:
            self.subtitles = deque(maxlen=1)  # Compact 모드: 최근 1개만
        else:
            self.subtitles = deque(maxlen=2)  # 완성된 자막 2개
        
        self.current_text = ""  # 진행중 텍스트
        self.animation_counter = 0  # 진행 중 인디케이터 애니메이션용 카운터
        
        # Canvas 아이템 ID들
        self.canvas_items = []
        
        # 애니메이션 관련
        self.animation_running = False
        self.animation_y_offset = 0
        self.target_y_offset = 0
        
        # 폰트 캐싱
        self.fonts = {}
        self._init_fonts()
        
        # 우클릭 메뉴
        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(label="투명도 높이기", command=lambda: self.change_alpha(0.1))
        self.menu.add_command(label="투명도 낮추기", command=lambda: self.change_alpha(-0.1))
        self.menu.add_separator()
        mode_label = "📋 일반 모드로" if self.compact_mode else "📋 컴팩트 모드로"
        self.menu.add_command(label=mode_label, command=self.toggle_compact_mode)
        self.menu.add_separator()
        self.menu.add_command(label="자막 지우기", command=self.clear_subtitles)
        self.menu.add_separator()
        self.menu.add_command(label="닫기", command=self.withdraw)
        
        # 드래그용 변수
        self._drag_start_x = 0
        self._drag_start_y = 0
        
        # 초기 렌더링
        self.render_subtitles()
    
    def _init_fonts(self):
        """폰트 초기화 (PIL)"""
        try:
            # 로컬 NanumSquareB.ttf 사용 (현재 디렉토리 기준)
            self.fonts['large'] = ImageFont.truetype("NanumSquareB.ttf", 45)
            self.fonts['medium'] = ImageFont.truetype("NanumSquareB.ttf", 40)
            self.fonts['small'] = ImageFont.truetype("malgunbd.ttf", 32)

        except:
            try:
                # 절대 경로로 시도 (파일이 다른 위치에 있을 경우)
                self.fonts['large'] = ImageFont.truetype("e:/myWork/WhisperLive/streaming-sensevoice/NanumSquareB.ttf", 42)
                self.fonts['medium'] = ImageFont.truetype("e:/myWork/WhisperLive/streaming-sensevoice/NanumSquareB.ttf", 38 )
            except:
                try:
                    # 맑은 고딕 볼드 폴백
                    self.fonts['large'] = ImageFont.truetype("malgunbd.ttf", 38)
                    self.fonts['medium'] = ImageFont.truetype("malgunbd.ttf", 34)
                except:
                    try:
                        # 일반 맑은 고딕
                        self.fonts['large'] = ImageFont.truetype("malgun.ttf", 42)
                        self.fonts['medium'] = ImageFont.truetype("malgun.ttf", 38)
                    except:
                        try:
                            # Arial Bold
                            self.fonts['large'] = ImageFont.truetype("arialbd.ttf", 38)
                            self.fonts['medium'] = ImageFont.truetype("arialbd.ttf", 34)
                        except:
                            # 기본 폰트
                            self.fonts['large'] = ImageFont.load_default()
                            self.fonts['medium'] = ImageFont.load_default()
    
    def _create_rounded_rectangle(self, x1, y1, x2, y2, radius=25, **kwargs):
        """둥근 사각형 그리기"""
        points = [
            x1 + radius, y1,
            x1 + radius, y1,
            x2 - radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1 + radius,
            x1, y1
        ]
        return self.canvas.create_polygon(points, smooth=True, **kwargs)
    
    def _create_text_image(self, text, font, text_color, shadow_color, shadow_offset=3, shadow_blur=2):
        """그림자 효과가 있는 텍스트 이미지 생성 (PIL)"""
        if not text:
            return None
        
        # 텍스트 크기 측정
        dummy_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        dummy_draw = ImageDraw.Draw(dummy_img)
        
        # 여러 줄로 나누기 (wraplength 고려)
        max_width = self.window_width - 100
        lines = self._wrap_text(text, font, max_width)
        
        # 각 줄의 bbox 계산
        line_heights = []
        line_widths = []
        for line in lines:
            bbox = dummy_draw.textbbox((0, 0), line, font=font)
            line_widths.append(bbox[2] - bbox[0])
            line_heights.append(bbox[3] - bbox[1])
        
        total_width = max(line_widths) if line_widths else 0
        total_height = sum(line_heights) + (len(lines) - 1) * 5  # 줄 간격 5px
        
        # 이미지 생성 (여유 공간 포함)
        img_width = total_width + shadow_offset * 2 + 40
        img_height = total_height + shadow_offset * 2 + 40
        
        # 투명 배경
        img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        
        # 그림자 레이어
        shadow_img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_img)
        
        # 그림자 텍스트 그리기
        y_offset = 20 + shadow_offset
        for line in lines:
            shadow_draw.text(
                (20 + shadow_offset, y_offset),
                line,
                font=font,
                fill=shadow_color
            )
            y_offset += line_heights[lines.index(line)] + 5
        
        # 그림자 블러
        shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
        
        # 그림자 합성
        img = Image.alpha_composite(img, shadow_img)
        
        # 메인 텍스트 그리기
        main_draw = ImageDraw.Draw(img)
        y_offset = 20
        for line in lines:
            main_draw.text(
                (20, y_offset),
                line,
                font=font,
                fill=text_color
            )
            y_offset += line_heights[lines.index(line)] + 5
        
        return img, img_width, img_height
    
    def _wrap_text(self, text, font, max_width):
        """텍스트를 최대 너비에 맞게 줄바꿈"""
        dummy_img = Image.new('RGBA', (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            bbox = dummy_draw.textbbox((0, 0), test_line, font=font)
            width = bbox[2] - bbox[0]
            
            if width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines if lines else [text]
    
    def render_subtitles(self):
        """자막 렌더링 (모든 Canvas 아이템 다시 그리기)"""
        # 기존 자막 아이템 제거 (배경 제외)
        for item in self.canvas_items:
            self.canvas.delete(item)
        self.canvas_items.clear()
        
        subtitles_list = list(self.subtitles)
        
        if self.compact_mode:
            # Compact 모드: 최근 1개만 표시 (가운데)
            if subtitles_list:
                translated, original = subtitles_list[-1]  # 가장 최근 항목
                display_text = translated if translated else original
                
                result = self._create_text_image(
                    display_text,
                    self.fonts['large'],
                    "white",
                    "#000000",
                    shadow_offset=0,
                    shadow_blur=0
                )
                
                if result:
                    pil_img, img_w, img_h = result
                    photo = ImageTk.PhotoImage(pil_img)
                    
                    # 중앙에 배치
                    item_id = self.canvas.create_image(
                        30, 10,
                        image=photo,
                        anchor='nw'
                    )
                    
                    self.canvas.itemconfig(item_id, tags="subtitle_latest")
                    self.canvas.photo_refs = getattr(self.canvas, 'photo_refs', [])
                    self.canvas.photo_refs.append(photo)
                    self.canvas_items.append(item_id)
            
            # Compact 모드에서 진행중 텍스트가 있으면 인디케이터 표시
            if self.current_text:
                # 점 애니메이션 인디케이터 (진행 중임을 표시)
                dots_count = (self.animation_counter % 3) + 1
                indicator_text = "●" * dots_count + "○" * (3 - dots_count)
                
                result = self._create_text_image(
                    indicator_text,
                    self.fonts['small'],
                    "#666666",
                    "#333333",
                    shadow_offset=0,
                    shadow_blur=0
                )
                
                if result:
                    pil_img, img_w, img_h = result
                    photo = ImageTk.PhotoImage(pil_img)
                    
                    # 우측 하단에 배치
                    item_id = self.canvas.create_image(
                        self.window_width - img_w - 20, self.window_height - img_h - 5,
                        image=photo,
                        anchor='nw'
                    )
                    
                    self.canvas.itemconfig(item_id, tags="indicator")
                    self.canvas.photo_refs = getattr(self.canvas, 'photo_refs', [])
                    self.canvas.photo_refs.append(photo)
                    self.canvas_items.append(item_id)
        else:
            # 일반 모드: 원래대로 동작
            # Y 위치 계산
            y_positions = [
                10 + self.animation_y_offset,   # 가장 오래된 자막 (위)
                100 + self.animation_y_offset,  # 최근 자막 (중간)
                200  # 진행중 자막 (하단, 고정)
            ]
            
            # 완성된 자막 렌더링 (오래된 것부터)
            for i, (translated, original) in enumerate(subtitles_list):
                display_text = translated if translated else original
                
                # 색상 및 폰트 설정
                if i == 0:  # 가장 오래된 자막
                    text_color = "#999999"
                    shadow_color = "#333333"
                    font = self.fonts['medium']
                else:  # 최근 자막
                    text_color = "white"
                    shadow_color = "#000000"
                    font = self.fonts['large']
                
                # 텍스트 이미지 생성
                result = self._create_text_image(
                    display_text,
                    font,
                    text_color,
                    shadow_color,
                    shadow_offset=0,
                    shadow_blur=0
                )
                
                if result:
                    pil_img, img_w, img_h = result
                    photo = ImageTk.PhotoImage(pil_img)
                    
                    # Canvas에 이미지 추가
                    item_id = self.canvas.create_image(
                        30, y_positions[i],
                        image=photo,
                        anchor='nw'
                    )
                    
                    # 참조 유지 (가비지 컬렉션 방지)
                    self.canvas.itemconfig(item_id, tags=f"subtitle_{i}")
                    self.canvas.photo_refs = getattr(self.canvas, 'photo_refs', [])
                    self.canvas.photo_refs.append(photo)
                    
                    self.canvas_items.append(item_id)
            
            # 진행중 자막 렌더링 (주황색, 항상 하단)
            if self.current_text:
                result = self._create_text_image(
                    f"> {self.current_text}",
                    self.fonts['small'],
                    "orange",
                    "#663300",
                    shadow_offset=3,
                    shadow_blur=2
                )
                
                if result:
                    pil_img, img_w, img_h = result
                    photo = ImageTk.PhotoImage(pil_img)
                    
                    item_id = self.canvas.create_image(
                        30, y_positions[2],
                        image=photo,
                        anchor='nw'
                    )
                    
                    self.canvas.itemconfig(item_id, tags="subtitle_current")
                    self.canvas.photo_refs = getattr(self.canvas, 'photo_refs', [])
                    self.canvas.photo_refs.append(photo)
                    
                    self.canvas_items.append(item_id)
    
    def toggle_compact_mode(self):
        """Compact 모드 토글"""
        self.compact_mode = not self.compact_mode
        
        # 창 크기 조정
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        if self.compact_mode:
            self.window_width = 1600
            self.window_height = 100
            self.subtitles = deque(self.subtitles, maxlen=1)  # 최근 1개만 유지
            logger.info("Switched to Compact Mode (latest entry only)")
        else:
            self.window_width = 1900
            self.window_height = 300
            self.subtitles = deque(self.subtitles, maxlen=2)  # 2개 유지
            logger.info("Switched to Normal Mode (latest two entries)")
        
        x = (screen_width - self.window_width) // 2
        y = screen_height - self.window_height - 100
        
        self.geometry(f"{self.window_width}x{self.window_height}+{x}+{y}")
        
        # Canvas 크기 업데이트
        self.canvas.configure(width=self.window_width, height=self.window_height)
        self.canvas.delete(self.bg_rect)
        self.bg_rect = self._create_rounded_rectangle(
            0, 0, self.window_width, self.window_height,
            radius=self.bg_radius,
            fill='black'
        )
        
        # 메뉴 텍스트 업데이트
        mode_label = "📋 일반 모드로" if self.compact_mode else "📋 컴팩트 모드로"
        self.menu.entryconfig(3, label=mode_label)  # 메뉴 항목 인덱스
        
        self.render_subtitles()
    
    def animate_scroll_up(self, duration=300):
        """위로 스크롤 애니메이션 (부드러운 이동)"""
        if self.animation_running:
            return
        
        self.animation_running = True
        self.target_y_offset = -90
        start_offset = self.animation_y_offset
        start_time = time.time()
        
        def animate_step():
            if not self.animation_running:
                return
            
            elapsed = time.time() - start_time
            progress = min(elapsed / (duration / 1000.0), 1.0)
            
            # Ease-out 효과
            progress = 1 - (1 - progress) ** 3
            
            self.animation_y_offset = start_offset + (self.target_y_offset - start_offset) * progress
            self.render_subtitles()
            
            if progress < 1.0:
                self.after(16, animate_step)  # ~60 FPS
            else:
                self.animation_running = False
                self.animation_y_offset = 0  # 리셋
                self.render_subtitles()
        
        animate_step()
    
    def add_subtitle(self, translated_text, original_text):
        """자막 추가 (스크롤 애니메이션 포함)"""
        if not translated_text or not translated_text.strip():
            return
        
        # 중복 체크
        if self.subtitles and self.subtitles[-1][0] == translated_text:
            return
        
        # 스크롤 애니메이션 시작
        if len(self.subtitles) == 2:  # 이미 2개가 있으면 스크롤
            self.animate_scroll_up()
            # 애니메이션 후 추가
            self.after(350, lambda: self._add_subtitle_internal(translated_text, original_text))
        else:
            self._add_subtitle_internal(translated_text, original_text)
    
    def _add_subtitle_internal(self, translated_text, original_text):
        """내부: 자막 추가 및 렌더링"""
        self.subtitles.append((translated_text, original_text))
        self.current_text = ""  # 진행중 텍스트 클리어
        self.render_subtitles()
    
    def update_current(self, text):
        """진행중 자막 업데이트"""
        self.current_text = text if text else ""
        # 컴팩트 모드에서 인디케이터 애니메이션을 위해 카운터 증가
        if self.compact_mode and self.current_text:
            self.animation_counter += 1
        elif not self.current_text:
            self.animation_counter = 0
        self.render_subtitles()
    
    def clear_subtitles(self):
        """자막 지우기"""
        self.subtitles.clear()
        self.current_text = ""
        self.animation_counter = 0  # 인디케이터 리셋
        self.canvas.photo_refs = []
        self.render_subtitles()
    
    def start_move(self, event):
        """드래그 시작"""
        self._drag_start_x = event.x
        self._drag_start_y = event.y
    
    def on_move(self, event):
        """드래그 중"""
        x = self.winfo_x() + event.x - self._drag_start_x
        y = self.winfo_y() + event.y - self._drag_start_y
        self.geometry(f"+{x}+{y}")
    
    def show_menu(self, event):
        """우클릭 메뉴"""
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()
    
    def change_alpha(self, delta):
        """투명도 변경"""
        current = self.attributes('-alpha')
        new_alpha = max(0.1, min(1.0, current + delta))
        self.attributes('-alpha', new_alpha)
    
    def show_overlay(self):
        """오버레이 표시"""
        self.deiconify()
        self.lift()
    
    def hide_overlay(self):
        """오버레이 숨기기"""
        self.withdraw()
