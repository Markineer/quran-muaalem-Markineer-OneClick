from typing import Literal
import diff_match_patch as dmp

from .explain import expalin_sifat
from .modeling.vocab import SIFAT_ATTR_TO_ARABIC_WITHOUT_BRACKETS
from .harakat_mode import (
    parse_ayah_to_slots,
    render_ayah_with_highlights,
    extract_harakat_stream,
    compare_harakat_to_slots,
)


def explain_for_gradio(
    phonemes: str,
    exp_phonemes: str,
    sifat: list,
    exp_sifat: list,
    lang: Literal["arabic", "english"] = "english",
) -> str:
    # Create diff-match-patch object
    dmp_obj = dmp.diff_match_patch()

    # Calculate differences using Google's diff-match-patch (same as terminal)
    diffs = dmp_obj.diff_main(exp_phonemes, phonemes)

    # Create HTML for phoneme differences
    phoneme_html = explain_phonemes_html(dmp_obj, diffs)

    # Create HTML for sifat table using your existing function
    sifat_table = expalin_sifat(sifat, exp_sifat, diffs)
    sifat_html = explain_sifat_html(sifat_table, lang)

    # Combine both sections
    html_output = f"""
    <div style="font-family: monospace; width: 100%;">
        <h3>مقارنة الحروف</h3>
        {phoneme_html}
        <h3>مقارنة صفات الحروف</h3>
        {sifat_html}
       <div class="color-legend">
    </div>
    """

    return html_output


def explain_phonemes_html(dmp_obj, diffs):
    html_output = '<div style="background-color: #000; padding: 10px; border-radius: 5px; margin-bottom: 20px; font-size: 30px;">'

    # Process each difference (same logic as terminal version)
    for op, data in diffs:
        if op == dmp_obj.DIFF_EQUAL:
            html_output += f'<span style="color: #ffffff;">{data}</span>'
        elif op == dmp_obj.DIFF_INSERT:
            html_output += f'<span style="color: #00ff00;">{data}</span>'
        elif op == dmp_obj.DIFF_DELETE:
            html_output += f'<span style="color: #ff0000; text-decoration: line-through;">{data}</span>'

    html_output += "</div>"
    return html_output


def explain_sifat_html(table, lang):
    if not table:
        return "<p>No sifat data available</p>"

    # Create HTML table with full width
    html_output = """
    <table style="width: 100%; border-collapse: collapse; background-color: #000; color: #fff; margin-bottom: 20px;">
        <thead>
            <tr>
    """

    # Get base columns (non-exp keys without 'tag')
    base_keys = [k for k in table[0].keys() if not k.startswith("exp_") and k != "tag"]

    # Add columns
    for key in base_keys:
        html_output += f'<th style="border: 1px solid #444; padding: 8px; text-align: left;">{key.replace("_", " ").title()}</th>'

    html_output += """
            </tr>
        </thead>
        <tbody>
    """

    # Add rows
    for row in table:
        tag = row["tag"]
        html_output += "<tr>"

        for key in base_keys:
            exp_key = f"exp_{key}"
            value = str(row[key])

            # Apply Arabic translation if needed
            if key != "phonemes" and lang == "arabic":
                value = SIFAT_ATTR_TO_ARABIC_WITHOUT_BRACKETS.get(value, value)

            # Apply styling based on tag and comparison
            if tag == "exact" and row.get(exp_key) != row[key]:
                html_output += f'<td style="border: 1px solid #444; padding: 8px; color: #ff0000;">{value}</td>'
            elif tag == "insert":
                html_output += f'<td style="border: 1px solid #444; padding: 8px; color: #ffff00;">{value}</td>'
            else:
                html_output += (
                    f'<td style="border: 1px solid #444; padding: 8px;">{value}</td>'
                )

        html_output += "</tr>"

    html_output += """
        </tbody>
    </table>
    """

    return html_output


def explain_harakat_only(
    ayah_text: str,
    phonemes: str,
    exp_phonemes: str,
    lang: Literal["arabic", "english"] = "arabic",
) -> str:
    """
    Generate HTML explanation for Harakat-only training mode.

    This function displays Quran ayah text (normal Arabic) with
    wrong letters highlighted based on harakat (diacritic) errors.

    Args:
        ayah_text: The Quran text to display (Uthmani script with tashkil)
        phonemes: Predicted phoneme string from model
        exp_phonemes: Expected phoneme string (reference)
        lang: Language for labels ("arabic" or "english")

    Returns:
        HTML string with highlighted errors and debug panel
    """
    # 1) Extract simplified harakat streams
    exp_h = extract_harakat_stream(exp_phonemes)
    pred_h = extract_harakat_stream(phonemes)

    # 2) Parse ayah into slots
    slots = parse_ayah_to_slots(ayah_text)

    # 3) Compute wrong slots (compare pred_h to slots expected)
    wrong_slot_idxs, hint_map = compare_harakat_to_slots(slots, pred_h)

    # 4) Render ayah with highlights
    ayah_html = render_ayah_with_highlights(ayah_text, slots, wrong_slot_idxs, hint_map)

    # 5) Debug phoneme diff block (optional, collapsible)
    dmp_obj = dmp.diff_match_patch()
    diffs = dmp_obj.diff_main(exp_phonemes, phonemes)
    phoneme_html = explain_phonemes_html(dmp_obj, diffs)

    # Calculate stats
    total_letters = sum(1 for s in slots if s.is_letter)
    error_count = len(wrong_slot_idxs)
    correct_count = total_letters - error_count
    accuracy = (correct_count / total_letters * 100) if total_letters > 0 else 0

    # Build final HTML
    title = "تدريب الحركات" if lang == "arabic" else "Harakat Training"
    debug_label = "عرض التفاصيل الصوتية (للتشخيص)" if lang == "arabic" else "Show Phoneme Details (Debug)"
    stats_label = "الإحصائيات" if lang == "arabic" else "Statistics"
    correct_label = "صحيح" if lang == "arabic" else "Correct"
    error_label = "خطأ" if lang == "arabic" else "Errors"
    accuracy_label = "الدقة" if lang == "arabic" else "Accuracy"

    return f"""
    <div style="font-family: 'Amiri', 'Traditional Arabic', serif; width: 100%;">
        <style>
            .ayah-box {{
                background: #0b0b0b;
                padding: 20px;
                border-radius: 12px;
                direction: rtl;
                font-size: 38px;
                line-height: 2.2;
                color: #fff;
                text-align: right;
                font-family: 'Amiri', 'Traditional Arabic', 'Scheherazade New', serif;
            }}
            .harakat-wrong {{
                border-bottom: 4px solid #ff3b30;
                background: rgba(255, 59, 48, 0.18);
                border-radius: 6px;
                padding: 0 4px;
                cursor: help;
            }}
            .harakat-uncertain {{
                border-bottom: 4px solid #ffcc00;
                background: rgba(255, 204, 0, 0.18);
                border-radius: 6px;
                padding: 0 4px;
                cursor: help;
            }}
            .stats-box {{
                background: #1a1a1a;
                padding: 12px 20px;
                border-radius: 8px;
                margin-top: 16px;
                display: flex;
                gap: 24px;
                justify-content: center;
                direction: rtl;
            }}
            .stat-item {{
                text-align: center;
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                color: #fff;
            }}
            .stat-label {{
                font-size: 14px;
                color: #888;
            }}
            .stat-correct .stat-value {{ color: #30d158; }}
            .stat-error .stat-value {{ color: #ff3b30; }}
            .stat-accuracy .stat-value {{ color: #0a84ff; }}
            .legend-box {{
                background: #1a1a1a;
                padding: 12px 20px;
                border-radius: 8px;
                margin-top: 12px;
                direction: rtl;
                text-align: right;
            }}
            .legend-item {{
                display: inline-block;
                margin-left: 20px;
                font-size: 14px;
                color: #ccc;
            }}
            .legend-color {{
                display: inline-block;
                width: 16px;
                height: 16px;
                border-radius: 4px;
                margin-left: 6px;
                vertical-align: middle;
            }}
            .legend-red {{ background: rgba(255, 59, 48, 0.5); border-bottom: 3px solid #ff3b30; }}
            .legend-yellow {{ background: rgba(255, 204, 0, 0.5); border-bottom: 3px solid #ffcc00; }}
            .legend-white {{ background: #333; }}
            details {{
                margin-top: 16px;
            }}
            summary {{
                cursor: pointer;
                color: #888;
                font-size: 14px;
                padding: 8px;
                background: #1a1a1a;
                border-radius: 6px;
            }}
            summary:hover {{
                color: #fff;
                background: #2a2a2a;
            }}
            .debug-panel {{
                margin-top: 12px;
                padding: 12px;
                background: #0a0a0a;
                border-radius: 8px;
            }}
        </style>

        <h3 style="direction: rtl; color: #fff; margin-bottom: 16px;">{title}</h3>

        <div class="ayah-box">{ayah_html}</div>

        <div class="stats-box">
            <div class="stat-item stat-correct">
                <div class="stat-value">{correct_count}</div>
                <div class="stat-label">{correct_label}</div>
            </div>
            <div class="stat-item stat-error">
                <div class="stat-value">{error_count}</div>
                <div class="stat-label">{error_label}</div>
            </div>
            <div class="stat-item stat-accuracy">
                <div class="stat-value">{accuracy:.1f}%</div>
                <div class="stat-label">{accuracy_label}</div>
            </div>
        </div>

        <div class="legend-box">
            <span class="legend-item">
                <span class="legend-color legend-red"></span>
                خطأ في الحركة (Harakat Error)
            </span>
            <span class="legend-item">
                <span class="legend-color legend-yellow"></span>
                غير متأكد (Uncertain)
            </span>
            <span class="legend-item">
                <span class="legend-color legend-white"></span>
                صحيح (Correct)
            </span>
        </div>

        <details>
            <summary>{debug_label}</summary>
            <div class="debug-panel">
                <p style="color: #888; font-size: 12px; margin-bottom: 8px;">
                    Phoneme comparison (expected vs predicted):
                </p>
                {phoneme_html}
            </div>
        </details>
    </div>
    """
