import io
import os
import pytest
from streamlit.testing.v1 import AppTest
from app.theme import get_theme_css

APP_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "app", "app.py")
)


def test_theme_css_mint_button_tokens_and_styles():
    """Verify soft mint button design tokens and selectors in both dark and light themes."""
    for theme_mode in ["dark", "light"]:
        css = get_theme_css(theme_mode)

        # 1. Tokens must define soft mint gradient and dark high-contrast text
        assert "--btn-proc-bg:" in css, f"Missing --btn-proc-bg in {theme_mode} CSS"
        assert "--btn-proc-tx:" in css, f"Missing --btn-proc-tx in {theme_mode} CSS"
        assert "--btn-proc-hover:" in css, f"Missing --btn-proc-hover in {theme_mode} CSS"
        assert "--tooltip-bg:" in css, f"Missing --tooltip-bg in {theme_mode} CSS"

        # Check high-contrast dark text
        if theme_mode == "dark":
            assert "#022c22" in css or "#064e3b" in css
        else:
            assert "#064e3b" in css

        # 2. Button selectors must match Streamlit 1.30+ testids and tooltip wrappers
        assert 'button[data-testid="stBaseButton-primary"]' in css
        assert 'div[data-testid="stTooltipHoverTarget"] button' in css
        assert "border-radius: 12px" in css or "border-radius: var(--radius-md)" in css

        # 3. Tooltip must be subtle, compact, and non-intrusive
        assert 'div[data-testid="stTooltipContent"]' in css
        assert 'div[data-baseweb="tooltip"]' in css
        assert "pointer-events: none" in css
        assert "0.74rem" in css

        # 4. Immediate suppression on active click
        assert ":active) div[data-testid=\"stTooltipContent\"]" in css


def test_process_button_and_tooltip_contract():
    """Verify the process button in Streamlit UI has the exact required label, primary type, and tooltip."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # Upload audio file
    at.file_uploader[0].upload("meeting_sample.wav", b"RIFF dummy audio content").run(timeout=30)

    assert not at.exception, f"App crashed on upload: {at.exception}"

    # Find the Process Meeting Audio button
    proc_btns = [b for b in at.button if b.key == "process_audio_btn"]
    assert len(proc_btns) == 1, "Expected exactly 1 process button with key 'process_audio_btn'"

    proc_btn = proc_btns[0]
    assert proc_btn.label == "🚀 Process Meeting Audio"
    assert proc_btn.proto.type == "primary"
    assert proc_btn.help == "Run Whisper transcription, PyAnnote diarization, and Qwen intelligence extraction."

    # Verify no plain-text token is leaked in any markdown element
    for md in at.markdown:
        assert "hf_cxTF" not in str(md.value), "Found leaked HF token in markdown element!"
