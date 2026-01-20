from .inference import Muaalem
from .muaalem_typing import MuaalemOutput, Unit, Sifa, SingleUnit
from .explain import explain_for_terminal
from .explain_html import explain_for_gradio, explain_harakat_only
from .harakat_mode import (
    FATIHA_AYAT,
    FATIHA_FULL_TEXT,
    HarakatClass,
    LetterSlot,
    parse_ayah_to_slots,
    marks_to_class,
    extract_harakat_stream,
    compare_harakat_to_slots,
    render_ayah_with_highlights,
    get_fatiha_reference_text,
    get_fatiha_ayat_list,
    get_expected_harakat_sequence_for_ayah,
    get_expected_sequences_for_all_ayat,
    compare_harakat_sequences,
)
from .realtime_harakat import (
    RealtimeHarakatSession,
    create_session,
    infer_and_update,
    detect_ayah,
)
from .ink_render import (
    render_ink_view,
    render_waiting_view,
    render_detecting_view,
    render_active_view,
    render_audio_level,
    render_inked_ayah,
    render_inkless_ayah,
)


__all__ = [
    # Core inference
    "Muaalem",
    "MuaalemOutput",
    "Unit",
    "Sifa",
    "SingleUnit",
    # Explanation functions
    "explain_for_terminal",
    "explain_for_gradio",
    "explain_harakat_only",
    # Harakat mode
    "FATIHA_AYAT",
    "FATIHA_FULL_TEXT",
    "HarakatClass",
    "LetterSlot",
    "parse_ayah_to_slots",
    "marks_to_class",
    "extract_harakat_stream",
    "compare_harakat_to_slots",
    "render_ayah_with_highlights",
    "get_fatiha_reference_text",
    "get_fatiha_ayat_list",
    "get_expected_harakat_sequence_for_ayah",
    "get_expected_sequences_for_all_ayat",
    "compare_harakat_sequences",
    # Real-time harakat
    "RealtimeHarakatSession",
    "create_session",
    "infer_and_update",
    "detect_ayah",
    # Ink render
    "render_ink_view",
    "render_waiting_view",
    "render_detecting_view",
    "render_active_view",
    "render_audio_level",
    "render_inked_ayah",
    "render_inkless_ayah",
]
