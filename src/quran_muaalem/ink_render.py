"""
Ink Render Module for Real-Time Harakat Training

This module generates the "ink UI" for displaying Quran text during
real-time recitation training. It provides:
- Inkless rendering for inactive ayat (faint outline style)
- Inked rendering for active ayah (full readable with highlights)
- Smooth transitions between states
- Error highlighting only on active ayah

The "ink" metaphor represents text that is either:
- Inkless: Visible but faint (like pencil outline, not filled)
- Inked: Full, bold, readable text
"""

from typing import Literal

from .harakat_mode import (
    FATIHA_AYAT,
    parse_ayah_to_slots,
    render_ayah_with_highlights,
)


# =============================================================================
# CSS STYLES
# =============================================================================

INK_UI_STYLES = """
<style>
/* Container */
.ink-container {
    font-family: 'Amiri', 'Traditional Arabic', 'Scheherazade New', serif;
    direction: rtl;
    text-align: right;
    background: #0a0a0a;
    padding: 20px;
    border-radius: 16px;
    width: 100%;
}

/* Status Pill */
.status-pill {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 14px;
    margin-bottom: 16px;
    font-family: sans-serif;
}

.status-detecting {
    background: rgba(255, 204, 0, 0.2);
    color: #ffcc00;
    border: 1px solid rgba(255, 204, 0, 0.4);
}

.status-detected {
    background: rgba(48, 209, 88, 0.2);
    color: #30d158;
    border: 1px solid rgba(48, 209, 88, 0.4);
}

.status-no-speech {
    background: rgba(142, 142, 147, 0.2);
    color: #8e8e93;
    border: 1px solid rgba(142, 142, 147, 0.4);
}

/* Ayah Row */
.ayah-row {
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 12px;
    transition: all 200ms ease;
    position: relative;
}

.ayah-row.active {
    background: rgba(255, 255, 255, 0.05);
    transform: scale(1.01);
}

.ayah-row.inactive {
    background: transparent;
}

/* Verse Number Badge */
.verse-badge {
    display: inline-block;
    font-size: 14px;
    color: rgba(255, 255, 255, 0.3);
    margin-left: 12px;
    font-family: sans-serif;
}

.ayah-row.active .verse-badge {
    color: rgba(255, 255, 255, 0.6);
}

/* Ayah Text - Inkless (Faint Outline) */
.ayah-inkless {
    font-size: 32px;
    line-height: 2.0;
    color: rgba(255, 255, 255, 0.08);
    -webkit-text-stroke: 0.5px rgba(255, 255, 255, 0.15);
    text-shadow: none;
    transition: all 200ms ease;
}

/* Ayah Text - Inked (Full Readable) */
.ayah-inked {
    font-size: 38px;
    line-height: 2.2;
    color: rgba(255, 255, 255, 0.95);
    -webkit-text-stroke: 0;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    transition: all 200ms ease;
}

/* Error Highlighting */
.harakat-wrong {
    border-bottom: 4px solid #ff3b30;
    background: rgba(255, 59, 48, 0.18);
    border-radius: 6px;
    padding: 0 4px;
    cursor: help;
}

.harakat-uncertain {
    border-bottom: 4px solid #ffcc00;
    background: rgba(255, 204, 0, 0.18);
    border-radius: 6px;
    padding: 0 4px;
    cursor: help;
}

/* Legend */
.ink-legend {
    margin-top: 20px;
    padding: 12px 16px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 8px;
    font-family: sans-serif;
    font-size: 12px;
    color: rgba(255, 255, 255, 0.5);
}

.legend-item {
    display: inline-block;
    margin-left: 20px;
}

.legend-dot {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 3px;
    margin-left: 6px;
    vertical-align: middle;
}

.legend-dot.wrong {
    background: rgba(255, 59, 48, 0.5);
    border-bottom: 2px solid #ff3b30;
}

.legend-dot.uncertain {
    background: rgba(255, 204, 0, 0.5);
    border-bottom: 2px solid #ffcc00;
}

.legend-dot.correct {
    background: rgba(48, 209, 88, 0.3);
}

.legend-dot.inkless {
    background: transparent;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* Audio Level Indicator (optional) */
.audio-level {
    height: 4px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 2px;
    margin-bottom: 16px;
    overflow: hidden;
}

.audio-level-bar {
    height: 100%;
    background: linear-gradient(90deg, #30d158, #ffcc00, #ff3b30);
    border-radius: 2px;
    transition: width 100ms ease;
}

/* Instructions */
.ink-instructions {
    margin-top: 16px;
    padding: 12px 16px;
    background: rgba(10, 132, 255, 0.1);
    border: 1px solid rgba(10, 132, 255, 0.3);
    border-radius: 8px;
    font-family: sans-serif;
    font-size: 13px;
    color: rgba(255, 255, 255, 0.7);
}
</style>
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def render_status_pill(
    status: Literal["detecting", "detected", "no_speech"],
    ayah_idx: int | None = None,
    confidence: float = 0.0,
) -> str:
    """
    Render the status indicator pill.

    Args:
        status: Current status (detecting, detected, no_speech)
        ayah_idx: Active ayah index (if detected)
        confidence: Detection confidence score

    Returns:
        HTML string for status pill
    """
    if status == "detecting":
        return '<div class="status-pill status-detecting">⏳ جاري التعرف على الآية...</div>'
    elif status == "detected" and ayah_idx is not None:
        ayah_num = ayah_idx + 1  # 1-indexed for display
        conf_pct = int(confidence * 100)
        return f'<div class="status-pill status-detected">🎯 الآية {ayah_num} ({conf_pct}%)</div>'
    else:
        return '<div class="status-pill status-no-speech">🔇 في انتظار الصوت...</div>'


def render_verse_badge(ayah_num: int) -> str:
    """
    Render verse number badge.

    Args:
        ayah_num: Verse number (1-indexed)

    Returns:
        HTML string for verse badge
    """
    # Arabic-Indic numerals
    arabic_nums = "٠١٢٣٤٥٦٧٨٩"
    num_str = "".join(arabic_nums[int(d)] for d in str(ayah_num))
    return f'<span class="verse-badge">﴿{num_str}﴾</span>'


def render_inkless_ayah(ayah_text: str, ayah_num: int) -> str:
    """
    Render an ayah in inkless (faint) style.

    Args:
        ayah_text: The ayah text
        ayah_num: Verse number (1-indexed)

    Returns:
        HTML string for inkless ayah
    """
    badge = render_verse_badge(ayah_num)
    return f'''
    <div class="ayah-row inactive">
        <span class="ayah-inkless">{ayah_text}</span>
        {badge}
    </div>
    '''


def render_inked_ayah(
    ayah_text: str,
    ayah_num: int,
    wrong_slots: set = None,
    hint_map: dict = None,
    uncertain_slots: set = None,
) -> str:
    """
    Render an ayah in inked (full) style with error highlights.

    Args:
        ayah_text: The ayah text
        ayah_num: Verse number (1-indexed)
        wrong_slots: Set of slot indices with errors
        hint_map: Dict mapping slot index to tooltip text
        uncertain_slots: Set of slot indices with uncertain status

    Returns:
        HTML string for inked ayah with highlights
    """
    if wrong_slots is None:
        wrong_slots = set()
    if hint_map is None:
        hint_map = {}
    if uncertain_slots is None:
        uncertain_slots = set()

    # Parse ayah into slots
    slots = parse_ayah_to_slots(ayah_text)

    # Map letter-slot indices to actual slot indices
    # wrong_slots indices are for letter slots only
    letter_slot_indices = [i for i, s in enumerate(slots) if s.is_letter]

    # Convert letter-slot indices to actual slot indices
    actual_wrong = set()
    actual_uncertain = set()
    actual_hints = {}

    for letter_idx in wrong_slots:
        if letter_idx < len(letter_slot_indices):
            actual_idx = letter_slot_indices[letter_idx]
            actual_wrong.add(actual_idx)
            if letter_idx in hint_map:
                actual_hints[actual_idx] = hint_map[letter_idx]

    for letter_idx in uncertain_slots:
        if letter_idx < len(letter_slot_indices):
            actual_idx = letter_slot_indices[letter_idx]
            if actual_idx not in actual_wrong:
                actual_uncertain.add(actual_idx)

    # Render with highlights
    highlighted_text = render_ayah_with_highlights(
        ayah_text,
        slots,
        actual_wrong,
        actual_hints,
        actual_uncertain,
    )

    badge = render_verse_badge(ayah_num)

    return f'''
    <div class="ayah-row active">
        <span class="ayah-inked">{highlighted_text}</span>
        {badge}
    </div>
    '''


def render_legend() -> str:
    """
    Render the color legend.

    Returns:
        HTML string for legend
    """
    return '''
    <div class="ink-legend">
        <span class="legend-item">
            <span class="legend-dot wrong"></span>
            خطأ في الحركة
        </span>
        <span class="legend-item">
            <span class="legend-dot uncertain"></span>
            غير متأكد
        </span>
        <span class="legend-item">
            <span class="legend-dot correct"></span>
            صحيح
        </span>
        <span class="legend-item">
            <span class="legend-dot inkless"></span>
            آية غير نشطة
        </span>
    </div>
    '''


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_ink_view(
    ayat: list = None,
    active_ayah_idx: int | None = None,
    wrong_slots_for_active: set = None,
    hint_map_for_active: dict = None,
    uncertain_slots_for_active: set = None,
    detection_status: Literal["detecting", "detected", "no_speech"] = "detecting",
    detection_confidence: float = 0.0,
    show_legend: bool = True,
    show_instructions: bool = False,
) -> str:
    """
    Render the complete ink UI view.

    This is the main entry point for rendering the real-time Quran display.
    It shows all ayat with the active one "inked" and others "inkless".

    Args:
        ayat: List of ayah texts (defaults to FATIHA_AYAT)
        active_ayah_idx: Index of currently active ayah (None = no detection)
        wrong_slots_for_active: Set of letter-slot indices with errors
        hint_map_for_active: Dict mapping letter-slot index to tooltip text
        uncertain_slots_for_active: Set of letter-slot indices with uncertain status
        detection_status: Current detection status
        detection_confidence: Confidence score for detection
        show_legend: Whether to show color legend
        show_instructions: Whether to show instructions

    Returns:
        Complete HTML string with styles and content
    """
    if ayat is None:
        ayat = FATIHA_AYAT
    if wrong_slots_for_active is None:
        wrong_slots_for_active = set()
    if hint_map_for_active is None:
        hint_map_for_active = {}
    if uncertain_slots_for_active is None:
        uncertain_slots_for_active = set()

    # Start building HTML
    html_parts = [INK_UI_STYLES]
    html_parts.append('<div class="ink-container">')

    # Status pill
    status_html = render_status_pill(detection_status, active_ayah_idx, detection_confidence)
    html_parts.append(status_html)

    # Render each ayah
    for i, ayah_text in enumerate(ayat):
        ayah_num = i + 1  # 1-indexed

        if i == active_ayah_idx:
            # Active ayah - inked with highlights
            html_parts.append(render_inked_ayah(
                ayah_text,
                ayah_num,
                wrong_slots_for_active,
                hint_map_for_active,
                uncertain_slots_for_active,
            ))
        else:
            # Inactive ayah - inkless
            html_parts.append(render_inkless_ayah(ayah_text, ayah_num))

    # Legend
    if show_legend:
        html_parts.append(render_legend())

    # Instructions
    if show_instructions:
        html_parts.append('''
        <div class="ink-instructions">
            💡 ابدأ القراءة من أي آية — سيتعرف النظام تلقائياً على الآية التي تقرأها
        </div>
        ''')

    html_parts.append('</div>')

    return '\n'.join(html_parts)


# =============================================================================
# AUDIO LEVEL INDICATOR
# =============================================================================

def render_audio_level(level: float) -> str:
    """
    Render audio level indicator bar.

    Args:
        level: Audio level (0.0 to 1.0)

    Returns:
        HTML string for audio level indicator
    """
    width_pct = min(100, max(0, int(level * 100)))
    return f'''
    <div class="audio-level">
        <div class="audio-level-bar" style="width: {width_pct}%;"></div>
    </div>
    '''


# =============================================================================
# SIMPLIFIED VIEWS
# =============================================================================

def render_waiting_view(ayat: list = None) -> str:
    """
    Render view when waiting for user to start.

    All ayat shown as inkless.

    Args:
        ayat: List of ayah texts

    Returns:
        HTML string
    """
    return render_ink_view(
        ayat=ayat,
        active_ayah_idx=None,
        detection_status="no_speech",
        show_legend=True,
        show_instructions=True,
    )


def render_detecting_view(ayat: list = None) -> str:
    """
    Render view when detecting which ayah user is reciting.

    All ayat shown as inkless with detecting status.

    Args:
        ayat: List of ayah texts

    Returns:
        HTML string
    """
    return render_ink_view(
        ayat=ayat,
        active_ayah_idx=None,
        detection_status="detecting",
        show_legend=True,
    )


def render_active_view(
    ayat: list = None,
    active_idx: int = 0,
    wrong_slots: set = None,
    hints: dict = None,
    uncertain_slots: set = None,
    confidence: float = 0.8,
) -> str:
    """
    Render view with active ayah detected.

    Active ayah shown inked, others inkless.

    Args:
        ayat: List of ayah texts
        active_idx: Index of active ayah
        wrong_slots: Set of wrong slot indices
        hints: Hint map for wrong slots
        uncertain_slots: Set of uncertain slot indices
        confidence: Detection confidence

    Returns:
        HTML string
    """
    return render_ink_view(
        ayat=ayat,
        active_ayah_idx=active_idx,
        wrong_slots_for_active=wrong_slots,
        hint_map_for_active=hints,
        uncertain_slots_for_active=uncertain_slots,
        detection_status="detected",
        detection_confidence=confidence,
        show_legend=True,
    )
