import streamlit as st
import streamlit.components.v1 as components
import os
import sys
import time

# Add the parent directory to sys.path so Python can find the modular folders
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Ensure 'app' is recognized as a package and 'app.app' exists when Streamlit executes app.py
if "app" in sys.modules:
    _mod = sys.modules["app"]
    if not hasattr(_mod, "__path__"):
        _mod.__path__ = [current_dir]
    sys.modules.setdefault("app.app", _mod)

# Import all modules (Member 1 & Member 2)
from member1_speech.asr.transcriber import transcribe_audio
from member1_speech.diarization.diarizer import diarize_audio, load_diarization_pipeline
from member1_speech.alignment.aligner import align_and_merge
from member2_intelligence.pipeline import run_intelligence_pipeline, format_duration
from member2_intelligence.intelligence import (
    MeetingIntelligence,
    Claim,
    Decision,
    ActionItem,
    Topic,
    Contradiction,
    Evidence,
)
try:
    from app.theme import inject_theme
except ImportError:
    from theme import inject_theme


# ── CACHED DIARIZATION MODEL ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading PyAnnote Diarization model...")
def get_cached_diarization_pipeline(token: str):
    """
    Caches the PyAnnote speaker diarization model pipeline in memory across runs.
    Audio-specific processing state remains local to each execution.
    """
    return load_diarization_pipeline(token)


def main():
    # ── PAGE CONFIG (must be first Streamlit command in script execution) ──
    st.set_page_config(
        page_title="EvidenceMeet | AI Meeting Intelligence",
        page_icon="🎙️",
        layout="wide",
    )

    # ── REQUIREMENT 1: THEME INITIALIZATION ───────────────────────────────
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"

    inject_theme(st.session_state["theme"])

    # ── SIDEBAR ────────────────────────────────────────────────────────────
    with st.sidebar:
        # Brand
        st.markdown(
            """
            <div class="em-sidebar-brand">
              <div class="em-sidebar-brand-name">🎙️ EvidenceMeet</div>
              <div class="em-sidebar-brand-tagline">AI Meeting Intelligence Platform</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Theme toggle (Requirement 1: functional dark/light toggle preserved across reruns)
        st.markdown("**🎨 Theme**")

        def _on_theme_toggle():
            is_light_active = st.session_state.get("theme_toggle", False)
            st.session_state["theme"] = "light" if is_light_active else "dark"

        current_theme = st.session_state.get("theme", "dark")
        is_light = (current_theme == "light")
        st.toggle(
            "☀️ Light Mode" if is_light else "🌙 Dark Mode",
            value=is_light,
            key="theme_toggle",
            on_change=_on_theme_toggle,
        )

        st.markdown('<hr class="em-gradient-divider" />', unsafe_allow_html=True)

        st.markdown("**Pipeline Modules**")
        modules = [
            "Speech-to-Text (Whisper)",
            "Speaker Diarization (PyAnnote)",
            "Speaker Alignment",
            "Text Preprocessing",
            "NLP Summarization",
        ]
        chips_html = "".join(f'<span class="em-module-chip">✦ {m}</span>' for m in modules)
        st.markdown(chips_html, unsafe_allow_html=True)

        st.markdown('<hr class="em-gradient-divider" />', unsafe_allow_html=True)

        # HF Token (Requirement 2: Secure Hugging Face token — masked, no plain-text, no hardcoded secrets)
        st.markdown("**🔑 Hugging Face Token**")
        env_token = os.environ.get("HF_TOKEN", "")
        hf_token = st.text_input(
            "Hugging Face Token",
            type="password",
            value=env_token if env_token else "",
            key="hf_token_input",
            help="Required for PyAnnote speaker diarization. Keep private and masked.",
            label_visibility="collapsed",
        )

        st.markdown('<hr class="em-gradient-divider" />', unsafe_allow_html=True)

        st.markdown("**Team**")
        st.markdown(
            """
            <div class="em-team-row">🧑‍💻 <strong>Anshu Kumar</strong> &nbsp;·&nbsp; <span style="color:var(--text-muted);font-size:0.78rem">Roll 29</span></div>
            <div class="em-team-row">👩‍💻 <strong>Aman Kumar</strong> &nbsp;·&nbsp; <span style="color:var(--text-muted);font-size:0.78rem">Roll 27</span></div>
            """,
            unsafe_allow_html=True,
        )

    # ── MAIN TITLE & HEADER ───────────────────────────────────────────────
    st.title("🎙️ EvidenceMeet: Meeting Intelligence")
    st.markdown("Automated edge-based transcription, diarization, summarization, and verified intelligence.")

    # ── FILE UPLOADER ──────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload meeting audio",
        type=["mp3", "wav", "m4a", "opus"],
        key="main_audio_uploader",
        help="Upload your meeting recording here (MP3, WAV, M4A, OPUS).",
    )

    # ── SESSION STATE INITIALIZATION ──────────────────────────────────────
    if "processed_data" not in st.session_state:
        st.session_state["processed_data"] = None
    if "current_file" not in st.session_state:
        st.session_state["current_file"] = None
    if "highlighted_segment_id" not in st.session_state:
        st.session_state["highlighted_segment_id"] = None
    if "nav_warning" not in st.session_state:
        st.session_state["nav_warning"] = None
    if "main_tab_selection" not in st.session_state:
        st.session_state["main_tab_selection"] = "🧠 Verified Intelligence"
    if "nav_origin" not in st.session_state:
        st.session_state["nav_origin"] = None
    if "expanded_category" not in st.session_state:
        st.session_state["expanded_category"] = None
    if "speaker_name_map" not in st.session_state:
        st.session_state["speaker_name_map"] = {}

    if uploaded_file is not None:
        # Reset state on new file upload
        if st.session_state["current_file"] != uploaded_file.name:
            st.session_state["processed_data"] = None
            st.session_state["current_file"] = uploaded_file.name
            st.session_state["speaker_name_map"] = {}
            st.session_state["highlighted_segment_id"] = None
            st.session_state["nav_origin"] = None
            st.session_state["nav_warning"] = None

        # Audio playback card
        st.markdown(
            f"""
            <div style="background:var(--bg-card);border:1px solid var(--border-glass);
                 border-radius:var(--radius-md);padding:14px 20px;margin:12px 0 16px 0;
                 box-shadow:var(--shadow-card);">
              <div style="font-size:0.78rem;font-weight:600;color:var(--text-muted);
                   text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">
                🎵 Audio Source: {uploaded_file.name}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.audio(uploaded_file)

        # ── REQUIREMENT 3: REDESIGNED PROCESSING BUTTON ────────────────────
        process_clicked = st.button(
            "🚀 Process Meeting Audio",
            type="primary",
            key="process_audio_btn",
            use_container_width=True,
            help="Run Whisper transcription, PyAnnote diarization, and Qwen intelligence extraction.",
        )

        # Subtle helper script to ensure tooltip never sticks on mouseleave or click
        components.html(
            """
            <script>
              (function() {
                function setupTooltipGuard() {
                  try {
                    var pDoc = window.parent.document;
                    var btn = pDoc.querySelector('button[data-testid="stBaseButton-primary"]') ||
                              pDoc.querySelector('div[data-testid="stButton"] button[kind="primary"]') ||
                              pDoc.querySelector('div[data-testid="stTooltipHoverTarget"] button');
                    if (btn && !btn.dataset.ttGuard) {
                      btn.dataset.ttGuard = "true";
                      // On click: immediately drop focus and hide any open tooltip
                      btn.addEventListener("click", function() {
                        btn.blur();
                        var tts = pDoc.querySelectorAll('[data-testid="stTooltipContent"], div[data-baseweb="tooltip"], [data-baseweb="popover"]');
                        tts.forEach(function(tt) { tt.style.display = "none"; });
                      });
                      // On mouseleave: drop focus so tooltip closes immediately
                      btn.addEventListener("mouseleave", function() {
                        btn.blur();
                      });
                    }
                  } catch (e) {}
                }
                setupTooltipGuard();
                setTimeout(setupTooltipGuard, 150);
                setTimeout(setupTooltipGuard, 500);
              })();
            </script>
            """,
            height=0,
            width=0,
        )

        if process_clicked:
            # Dismiss any active tooltip immediately on processing start
            components.html(
                """
                <script>
                  try {
                    var pDoc = window.parent.document;
                    var btn = pDoc.querySelector('button[data-testid="stBaseButton-primary"]') ||
                              pDoc.querySelector('button[kind="primary"]');
                    if (btn) btn.blur();
                    var tts = pDoc.querySelectorAll('[data-testid="stTooltipContent"], div[data-baseweb="tooltip"], [data-baseweb="popover"]');
                    tts.forEach(function(tt) { tt.style.display = "none"; });
                  } catch (e) {}
                </script>
                """,
                height=0,
                width=0,
            )

            active_token = hf_token.strip() or os.environ.get("HF_TOKEN", "").strip()
            if not active_token:
                st.error("⚠️ Please enter your Hugging Face Token in the sidebar to run the pipeline!")
                st.stop()

            file_ext = uploaded_file.name.split(".")[-1]
            temp_filename = f"temp_audio.{file_ext}"
            with open(temp_filename, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.expander("🔍 See Processing Stages", expanded=True):
                with st.status("⚙️ Running EvidenceMeet Pipeline...", expanded=True) as status_box:

                    # Stage 1: Transcription
                    t0 = time.perf_counter()
                    raw_text, segments, lang_code = transcribe_audio(temp_filename)
                    t_transcription = time.perf_counter() - t0
                    st.write(f"✅ 1. Transcription — {format_duration(t_transcription)}")

                    # Stage 2: Speaker Diarization
                    t0 = time.perf_counter()
                    t0_load = time.perf_counter()
                    cached_pipeline = get_cached_diarization_pipeline(active_token)
                    t_load = time.perf_counter() - t0_load

                    diarization_intervals, diar_timings = diarize_audio(
                        temp_filename,
                        hf_token=active_token,
                        pipeline=cached_pipeline,
                        model_load_time=t_load,
                        return_timings=True,
                    )
                    t_diarization = time.perf_counter() - t0
                    unique_speakers = sorted(list(set(s["speaker"] for s in diarization_intervals)))
                    num_speakers = len(unique_speakers)
                    st.write(f"✅ 2. Speaker Diarization — {format_duration(t_diarization)}")
                    st.caption(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;⏱️ Model Init: {format_duration(diar_timings['model_load'])} &nbsp;|&nbsp; "
                        f"Audio Prep: {format_duration(diar_timings['conversion'])} &nbsp;|&nbsp; "
                        f"Inference: {format_duration(diar_timings['inference'])}"
                    )
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;👥 {num_speakers} speakers detected")

                    # Stage 3: Speaker Alignment
                    t0 = time.perf_counter()
                    aligned_transcript = align_and_merge(segments, diarization_intervals)
                    t_alignment = time.perf_counter() - t0
                    st.write(f"✅ 3. Speaker Alignment — {format_duration(t_alignment)}")

                    # Stage 4-7: Member 2 Intelligence Pipeline
                    member1_timing = {
                        "transcription": t_transcription,
                        "diarization": t_diarization,
                        "alignment": t_alignment,
                    }

                    def on_stage_complete(stage_name, duration):
                        if stage_name == "preprocessing":
                            st.write(f"✅ 4. Text Preprocessing — {format_duration(duration)}")
                        elif stage_name == "bart_summary":
                            st.write(f"✅ 5. BART Baseline Summary — {format_duration(duration)}")
                        elif stage_name == "qwen_extraction":
                            st.write(f"✅ 6. Qwen Intelligence — {format_duration(duration)}")
                        elif stage_name == "evidence":
                            st.write(f"✅ 7. Evidence Retrieval & Verification — {format_duration(duration)}")

                    intelligence_result = run_intelligence_pipeline(
                        aligned_transcript,
                        lang_code,
                        uploaded_file.name,
                        member1_timing=member1_timing,
                        stage_callback=on_stage_complete,
                    )

                    processed_transcript = intelligence_result["transcript"]
                    cleaned_text = intelligence_result["cleaned_text"]
                    summary_text = intelligence_result["summary"]
                    verified_intelligence = (
                        intelligence_result.get("verified_intelligence")
                        or intelligence_result.get("intelligence")
                    )
                    intelligence_error = intelligence_result.get("intelligence_error")
                    timing = intelligence_result["timing"]

                    total_seconds = timing.get("total", 0.0)
                    total_min = int(total_seconds // 60)
                    rem_sec = total_seconds % 60
                    total_str = f"{total_min} min {rem_sec:.2f} sec"

                    st.write("")
                    st.write("🚀 **Pipeline Complete!**")
                    st.write(f"Total processing time: {total_str}")
                    status_box.update(label="🚀 Pipeline Complete!", state="complete", expanded=False)

            # Cleanup temporary audio files
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            for suffix in [".wav", "_converted.wav"]:
                cand = temp_filename.rsplit(".", 1)[0] + suffix
                if os.path.exists(cand):
                    os.remove(cand)

            # Store in session state
            st.session_state["processed_data"] = {
                "aligned_transcript": aligned_transcript,
                "unique_speakers": unique_speakers,
                "cleaned_text": cleaned_text,
                "summary_text": summary_text,
                "verified_intelligence": verified_intelligence,
                "intelligence_error": intelligence_error,
                "timing": timing,
                "total_str": total_str,
                "diar_timings": diar_timings,
            }

        # Display processing stages summary when results exist but pipeline wasn't just triggered
        if st.session_state["processed_data"] is not None and not process_clicked:
            data = st.session_state["processed_data"]
            t = data.get("timing", {})
            dt = data.get("diar_timings")
            with st.expander("🔍 See Processing Stages", expanded=False):
                st.write(f"✅ 1. Transcription — {format_duration(t.get('transcription', 0.0))}")
                st.write(f"✅ 2. Speaker Diarization — {format_duration(t.get('diarization', 0.0))}")
                if dt:
                    st.caption(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;⏱️ Model Init: {format_duration(dt.get('model_load', 0.0))} &nbsp;|&nbsp; "
                        f"Audio Prep: {format_duration(dt.get('conversion', 0.0))} &nbsp;|&nbsp; "
                        f"Inference: {format_duration(dt.get('inference', 0.0))}"
                    )
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;👥 {len(data.get('unique_speakers', []))} speakers detected")
                st.write(f"✅ 3. Speaker Alignment — {format_duration(t.get('alignment', 0.0))}")
                st.write(f"✅ 4. Text Preprocessing — {format_duration(t.get('preprocessing', 0.0))}")
                st.write(f"✅ 5. BART Baseline Summary — {format_duration(t.get('bart_summary', 0.0))}")
                st.write(f"✅ 6. Qwen Intelligence — {format_duration(t.get('qwen_extraction', 0.0))}")
                st.write(f"✅ 7. Evidence Retrieval & Verification — {format_duration(t.get('evidence', 0.0))}")
                st.write("")
                st.write("🚀 **Pipeline Complete!**")
                st.write(f"Total processing time: {data.get('total_str', '')}")

        # ── DISPLAY RESULTS ────────────────────────────────────────────────
        if st.session_state["processed_data"] is not None:
            data = st.session_state["processed_data"]
            summary_text       = data.get("summary_text", "")
            cleaned_text       = data.get("cleaned_text", "")
            aligned_transcript = data.get("aligned_transcript", [])
            unique_speakers    = data.get("unique_speakers", [])
            intelligence       = data.get("verified_intelligence")
            intelligence_error = data.get("intelligence_error")
            timing             = data.get("timing", {})
            total_str          = data.get("total_str", "")

            # ── Synchronize speaker renaming ──────────────────────────────
            speaker_name_map = st.session_state["speaker_name_map"]

            def _on_intel_speaker_change(spk):
                val = st.session_state.get(f"intel_speaker_input_{spk}", "").strip()
                st.session_state["speaker_name_map"][spk] = val if val else spk
                st.session_state[f"speaker_input_{spk}"] = val

            def _on_transcript_speaker_change(spk):
                val = st.session_state.get(f"speaker_input_{spk}", "").strip()
                st.session_state["speaker_name_map"][spk] = val if val else spk
                st.session_state[f"intel_speaker_input_{spk}"] = val

            for spk in unique_speakers:
                ikey = f"intel_speaker_input_{spk}"
                tkey = f"speaker_input_{spk}"
                ival = st.session_state.get(ikey, "").strip()
                tval = st.session_state.get(tkey, "").strip()
                mval = speaker_name_map.get(spk, spk)

                resolved = None
                if ival:
                    resolved = ival
                elif tval:
                    resolved = tval
                elif mval and mval != spk:
                    resolved = mval

                if resolved:
                    speaker_name_map[spk] = resolved
                    st.session_state[ikey] = resolved
                    st.session_state[tkey] = resolved
                else:
                    speaker_name_map[spk] = spk
                    if ikey not in st.session_state:
                        st.session_state[ikey] = ""

            # ── REQUIREMENT 5: DIRECT EVIDENCE NAVIGATION WITH RETURN ──────
            def _navigate_to_segment(target_seg_id, aligned_blocks, origin=None):
                valid_ids = [b.get("id", i) for i, b in enumerate(aligned_blocks, start=1)]
                if target_seg_id is not None and target_seg_id in valid_ids:
                    st.session_state["highlighted_segment_id"] = target_seg_id
                    st.session_state["nav_warning"] = None
                    st.session_state["main_tab_selection"] = "💬 Smart Transcript"
                    if origin:
                        st.session_state["nav_origin"] = origin
                else:
                    st.session_state["highlighted_segment_id"] = None
                    st.session_state["nav_warning"] = (
                        f"Segment #{target_seg_id} could not be found in the current meeting transcript."
                        if target_seg_id is not None
                        else "Evidence segment reference is missing or invalid."
                    )

            # ── Quick stats strip ──────────────────────────────────────────
            intel = intelligence
            n_claims      = len(getattr(intel, "claims", []) or [])
            n_decisions   = len(getattr(intel, "decisions", []) or [])
            n_actions     = len(getattr(intel, "action_items", []) or [])
            n_topics      = len(getattr(intel, "topics", []) or [])
            n_contradicts = len(getattr(intel, "contradictions", []) or [])
            n_segments    = len(aligned_transcript)

            st.markdown(
                f"""
                <div style="margin:16px 0 14px 0;display:flex;flex-wrap:wrap;gap:8px;">
                  <span class="em-stat-pill">👥 <span>{len(unique_speakers)}</span> Speakers</span>
                  <span class="em-stat-pill">💬 <span>{n_segments}</span> Segments</span>
                  <span class="em-stat-pill">✅ <span>{n_actions}</span> Actions</span>
                  <span class="em-stat-pill">🎯 <span>{n_decisions}</span> Decisions</span>
                  <span class="em-stat-pill">📌 <span>{n_claims}</span> Claims</span>
                  <span class="em-stat-pill">🏷️ <span>{n_topics}</span> Topics</span>
                  <span class="em-stat-pill">⚡ <span>{n_contradicts}</span> Conflicts</span>
                  <span class="em-stat-pill">⏱️ <span>{total_str}</span></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.divider()

            # ── Main Tabs ──────────────────────────────────────────────────
            tab_names = [
                "🧠 Verified Intelligence",
                "📋 Summary",
                "✨ Cleaned Text",
                "💬 Smart Transcript",
                "⏱️ Processing Performance",
            ]
            tab_intel, tab_summary, tab_cleaned, tab_transcript, tab_perf = st.tabs(
                tab_names,
                default=st.session_state.get("main_tab_selection", "🧠 Verified Intelligence"),
            )

            # ══════════════════════════════════════════════════════════════
            # TAB 1 — VERIFIED INTELLIGENCE
            # ══════════════════════════════════════════════════════════════
            with tab_intel:
                st.subheader("Verified Meeting Intelligence")
                st.caption("Extracted meeting intelligence grounded with verified transcript citations and quotes.")

                # Navigation notice or active highlight banner
                if st.session_state.get("nav_warning"):
                    st.warning(f"⚠️ Navigation notice: {st.session_state['nav_warning']}")

                hi_id = st.session_state.get("highlighted_segment_id")
                if hi_id is not None:
                    col_nav1, col_nav2 = st.columns([4, 1])
                    with col_nav1:
                        st.info(f"📍 Currently focused on **Segment #{hi_id}** in Smart Transcript.")
                    with col_nav2:
                        if st.button("✖️ Clear focus", key="clear_focus_intel"):
                            st.session_state["highlighted_segment_id"] = None
                            st.session_state["nav_origin"] = None
                            st.rerun()

                # Manual speaker name mapping expander
                if unique_speakers:
                    with st.expander("🏷️ Edit Speaker Names (Optional)", expanded=False):
                        st.caption("Map detected speaker IDs to names for evidence citations and transcripts:")
                        for spk in unique_speakers:
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.markdown(f"**{spk}** →")
                            with col2:
                                st.text_input(
                                    f"Name for {spk}",
                                    key=f"intel_speaker_input_{spk}",
                                    placeholder="Enter name",
                                    label_visibility="collapsed",
                                    on_change=_on_intel_speaker_change,
                                    args=(spk,),
                                )

                if intelligence_error:
                    st.warning(f"⚠️ Notice: {intelligence_error}")

                has_items = intel and (
                    getattr(intel, "claims", None)
                    or getattr(intel, "decisions", None)
                    or getattr(intel, "action_items", None)
                    or getattr(intel, "topics", None)
                    or getattr(intel, "contradictions", None)
                )

                if not has_items:
                    st.info("No verified intelligence items detected for this meeting.")
                else:
                    # ── Display Evidence Helper ────────────────────────────
                    def _display_evidence(evidence_list, item_key_prefix="", is_partial=False, origin_meta=None):
                        if evidence_list:
                            label = "Partial Evidence (unverified)" if is_partial else "Verified Evidence"
                            st.markdown(f"**{label}:** *({len(evidence_list)} segment{'s' if len(evidence_list) > 1 else ''})*")
                            for ev_idx, ev in enumerate(evidence_list):
                                speaker = getattr(ev, "speaker", "UNKNOWN")
                                start = getattr(ev, "start", 0.0)
                                end = getattr(ev, "end", 0.0)
                                seg_id = getattr(ev, "segment_id", None)
                                quote = getattr(ev, "quote", "")
                                seg_tag = f"Segment #{seg_id}" if seg_id is not None else "Segment"
                                display_name = speaker_name_map.get(speaker, speaker)
                                if display_name != speaker:
                                    speaker_label = f"**{display_name}** ({speaker})"
                                else:
                                    speaker_label = f"**{speaker}**"

                                col_cap, col_btn = st.columns([3, 2])
                                with col_cap:
                                    st.caption(f"🗣️ {speaker_label} • ⏱️ `{start:05.2f}s - {end:05.2f}s` • {seg_tag}")
                                with col_btn:
                                    btn_key = f"jump_btn_{item_key_prefix}_{seg_id}_{ev_idx}"
                                    if st.button(
                                        f"⏱️ Jump to {seg_tag}",
                                        key=btn_key,
                                        help=f"Navigate to {seg_tag} in Smart Transcript and highlight it",
                                    ):
                                        nav_origin_data = {
                                            "tab": "🧠 Verified Intelligence",
                                            "category": origin_meta.get("category", "") if origin_meta else "",
                                            "category_label": origin_meta.get("category_label", "") if origin_meta else "",
                                            "item_idx": origin_meta.get("item_idx", 1) if origin_meta else 1,
                                            "item_title": origin_meta.get("item_title", "") if origin_meta else "",
                                            "segment_id": seg_id,
                                        }
                                        _navigate_to_segment(seg_id, aligned_transcript, origin=nav_origin_data)
                                        st.rerun()

                                st.markdown(f"> *\"{quote}\"*")
                        else:
                            st.markdown("⚠️ *No verified transcript evidence found (unsupported)*")

                    def _format_item_badge(item):
                        has_ev = bool(getattr(item, "evidence", []))
                        status = getattr(item, "verification_status", None)
                        score = getattr(item, "support_score", 0.0)
                        if has_ev or status == "SUPPORTED":
                            score_str = f" • Strongly Supported (Support: {score:.0%})" if score > 0 else ""
                            return f"🟢 Verified{score_str}"
                        elif status == "PARTIALLY_SUPPORTED":
                            return f"🟡 Partially Supported (Support: {score:.0%})"
                        else:
                            score_str = f" • Unsupported (Support: {score:.0%})" if score > 0 else ""
                            return f"🔴 Unverified{score_str}"

                    def _display_score_caption(item):
                        score = getattr(item, "support_score", 0.0)
                        status = getattr(item, "verification_status", "UNSUPPORTED")
                        conf = getattr(item, "confidence", None)
                        if conf is not None:
                            st.caption(
                                f"📊 **Evidence Support Score:** `{score:.2f}` ({status}, lexical grounding) &nbsp;|&nbsp; "
                                f"🤖 **Model Confidence:** `{conf:.2f}` (model certainty)"
                            )
                        else:
                            st.caption(
                                f"📊 **Evidence Support Score:** `{score:.2f}` ({status}, lexical grounding)"
                            )

                    # ── REQUIREMENT 4: COLLAPSIBLE MEETING INTELLIGENCE ────
                    expanded_cat = st.session_state.get("expanded_category")

                    action_items = getattr(intel, "action_items", []) or []
                    decisions = getattr(intel, "decisions", []) or []
                    claims = getattr(intel, "claims", []) or []
                    topics = getattr(intel, "topics", []) or []
                    contradictions = getattr(intel, "contradictions", []) or []

                    # Category 1: Action Items
                    with st.expander(f"✅ Action Items ({len(action_items)})", expanded=(expanded_cat == "action_items")):
                        st.markdown(f"### ✅ Action Items ({len(action_items)})")
                        if not action_items:
                            st.caption("No action items detected.")
                        else:
                            for idx, a in enumerate(action_items, start=1):
                                with st.container():
                                    commitment = getattr(a, "commitment_status", "committed")
                                    commit_label = "Accepted Commitment" if commitment == "committed" else "Request / Proposal (Uncommitted)"
                                    owner = getattr(a, "owner", None) or "Unassigned"
                                    deadline = getattr(a, "deadline", None) or "Not specified"
                                    display_owner = speaker_name_map.get(owner, owner)
                                    if display_owner != owner:
                                        owner_label = f"{display_owner} ({owner})"
                                    else:
                                        owner_label = owner
                                    badge = _format_item_badge(a)
                                    st.markdown(f"**Action {idx}:** {a.task} &nbsp; `{badge}`")
                                    st.caption(f"👤 **Owner:** {owner_label} &nbsp;|&nbsp; 📅 **Deadline:** {deadline} &nbsp;|&nbsp; 📋 **Type:** {commit_label}")
                                    _display_score_caption(a)
                                    origin_meta = {
                                        "category": "action_items",
                                        "category_label": "Action Items",
                                        "item_idx": idx,
                                        "item_title": a.task,
                                    }
                                    _display_evidence(getattr(a, "evidence", []), item_key_prefix=f"action_{idx}", origin_meta=origin_meta)
                                    if getattr(a, "verification_status", None) == "PARTIALLY_SUPPORTED" and getattr(a, "partial_evidence", []):
                                        _display_evidence(a.partial_evidence, item_key_prefix=f"act_part_{idx}", is_partial=True, origin_meta=origin_meta)
                                    st.markdown("---")

                    # Category 2: Decisions
                    with st.expander(f"🎯 Decisions ({len(decisions)})", expanded=(expanded_cat == "decisions")):
                        st.markdown(f"### 🎯 Decisions ({len(decisions)})")
                        if not decisions:
                            st.caption("No decisions detected.")
                        else:
                            for idx, d in enumerate(decisions, start=1):
                                with st.container():
                                    status = getattr(d, "status", "confirmed")
                                    status_label = "Confirmed Decision" if status == "confirmed" else f"Tentative Proposal ({status})"
                                    badge = _format_item_badge(d)
                                    st.markdown(f"**Decision {idx}:** {d.text} &nbsp; `{badge}`")
                                    st.caption(f"📋 **Classification:** {status_label}")
                                    _display_score_caption(d)
                                    origin_meta = {
                                        "category": "decisions",
                                        "category_label": "Decisions",
                                        "item_idx": idx,
                                        "item_title": d.text,
                                    }
                                    _display_evidence(getattr(d, "evidence", []), item_key_prefix=f"decision_{idx}", origin_meta=origin_meta)
                                    if getattr(d, "verification_status", None) == "PARTIALLY_SUPPORTED" and getattr(d, "partial_evidence", []):
                                        _display_evidence(d.partial_evidence, item_key_prefix=f"dec_part_{idx}", is_partial=True, origin_meta=origin_meta)
                                    st.markdown("---")

                    # Category 3: Claims
                    with st.expander(f"📌 Factual Claims ({len(claims)})", expanded=(expanded_cat == "claims")):
                        st.markdown(f"### 📌 Factual Claims ({len(claims)})")
                        if not claims:
                            st.caption("No factual claims detected.")
                        else:
                            for idx, c in enumerate(claims, start=1):
                                with st.container():
                                    badge = _format_item_badge(c)
                                    st.markdown(f"**Claim {idx}:** {c.text} &nbsp; `{badge}`")
                                    _display_score_caption(c)
                                    origin_meta = {
                                        "category": "claims",
                                        "category_label": "Factual Claims",
                                        "item_idx": idx,
                                        "item_title": c.text,
                                    }
                                    _display_evidence(getattr(c, "evidence", []), item_key_prefix=f"claim_{idx}", origin_meta=origin_meta)
                                    if getattr(c, "verification_status", None) == "PARTIALLY_SUPPORTED" and getattr(c, "partial_evidence", []):
                                        _display_evidence(c.partial_evidence, item_key_prefix=f"claim_part_{idx}", is_partial=True, origin_meta=origin_meta)
                                    st.markdown("---")

                    # Category 4: Topics
                    with st.expander(f"🏷️ Topics Discussed ({len(topics)})", expanded=(expanded_cat == "topics")):
                        st.markdown(f"### 🏷️ Topics Discussed ({len(topics)})")
                        if not topics:
                            st.caption("No topics detected.")
                        else:
                            for idx, t in enumerate(topics, start=1):
                                with st.container():
                                    badge = _format_item_badge(t)
                                    st.markdown(f"**Topic {idx}:** {t.name} &nbsp; `{badge}`")
                                    _display_score_caption(t)
                                    origin_meta = {
                                        "category": "topics",
                                        "category_label": "Topics Discussed",
                                        "item_idx": idx,
                                        "item_title": t.name,
                                    }
                                    _display_evidence(getattr(t, "evidence", []), item_key_prefix=f"topic_{idx}", origin_meta=origin_meta)
                                    if getattr(t, "verification_status", None) == "PARTIALLY_SUPPORTED" and getattr(t, "partial_evidence", []):
                                        _display_evidence(t.partial_evidence, item_key_prefix=f"topic_part_{idx}", is_partial=True, origin_meta=origin_meta)
                                    st.markdown("---")

                    # Category 5: Contradictions
                    with st.expander(f"⚡ Contradictions & Conflicts ({len(contradictions)})", expanded=(expanded_cat == "contradictions")):
                        st.markdown(f"### ⚡ Contradictions & Conflicts ({len(contradictions)})")
                        if not contradictions:
                            st.caption("No contradictions detected.")
                        else:
                            for idx, x in enumerate(contradictions, start=1):
                                with st.container():
                                    badge = _format_item_badge(x)
                                    st.markdown(f"**Contradiction {idx}:** {x.description} &nbsp; `{badge}`")
                                    _display_score_caption(x)
                                    origin_meta = {
                                        "category": "contradictions",
                                        "category_label": "Contradictions & Conflicts",
                                        "item_idx": idx,
                                        "item_title": x.description,
                                    }
                                    _display_evidence(getattr(x, "evidence", []), item_key_prefix=f"contra_{idx}", origin_meta=origin_meta)
                                    if getattr(x, "verification_status", None) == "PARTIALLY_SUPPORTED" and getattr(x, "partial_evidence", []):
                                        _display_evidence(x.partial_evidence, item_key_prefix=f"contra_part_{idx}", is_partial=True, origin_meta=origin_meta)
                                    st.markdown("---")

            # ══════════════════════════════════════════════════════════════
            # TAB 2 — SUMMARY
            # ══════════════════════════════════════════════════════════════
            with tab_summary:
                st.subheader("Meeting Summary")
                st.success(summary_text)
                st.download_button(
                    "📥 Download Summary",
                    summary_text,
                    file_name="meeting_summary.txt",
                    key="download_summary_btn",
                    use_container_width=True,
                )

            # ══════════════════════════════════════════════════════════════
            # TAB 3 — CLEANED TEXT
            # ══════════════════════════════════════════════════════════════
            with tab_cleaned:
                st.subheader("Cleaned Meeting Transcript")
                st.text_area(
                    label="Final Output",
                    value=cleaned_text,
                    height=350,
                    key="cleaned_text_area",
                    label_visibility="collapsed",
                )
                st.download_button(
                    "📥 Download Cleaned Text",
                    cleaned_text,
                    file_name="cleaned_transcript.txt",
                    key="download_cleaned_btn",
                    use_container_width=True,
                )

            # ══════════════════════════════════════════════════════════════
            # TAB 4 — SMART TRANSCRIPT
            # ══════════════════════════════════════════════════════════════
            with tab_transcript:
                st.subheader("Diarized Meeting Log")

                # Speaker Count
                st.markdown(f"#### 👥 {len(unique_speakers)} Speakers Detected")

                # ── REQUIREMENT 5: HIGHLIGHT BANNER & RETURN TO PREVIOUS VIEW ──
                hi_seg = st.session_state.get("highlighted_segment_id")
                nav_origin = st.session_state.get("nav_origin")

                if hi_seg is not None:
                    col_hi, col_ret, col_clr = st.columns([4, 2, 1])
                    with col_hi:
                        origin_info = ""
                        if nav_origin:
                            origin_info = f" from **{nav_origin.get('category_label', 'Intelligence')}** (Item #{nav_origin.get('item_idx', 1)})"
                        st.success(f"🎯 **Highlighted Segment #{hi_seg}**{origin_info} (visible below in chat log)")
                    with col_ret:
                        if nav_origin is not None:
                            if st.button("↩️ Return to Previous View", key="btn_return_to_origin"):
                                prev_tab = nav_origin.get("tab", "🧠 Verified Intelligence")
                                prev_cat = nav_origin.get("category")
                                st.session_state["main_tab_selection"] = prev_tab
                                st.session_state["expanded_category"] = prev_cat
                                # Clear origin to prevent navigation loops
                                st.session_state["nav_origin"] = None
                                st.rerun()
                    with col_clr:
                        if st.button("✖️ Clear highlight", key="clear_transcript_hi"):
                            st.session_state["highlighted_segment_id"] = None
                            st.session_state["nav_origin"] = None
                            st.rerun()

                    # Anchor link & JavaScript smooth scrolling into view
                    st.markdown(
                        f'<a href="#segment-{hi_seg}" class="em-anchor-jump">⬇️ Scroll directly to Segment #{hi_seg}</a>',
                        unsafe_allow_html=True,
                    )
                    components.html(
                        f"""
                        <script>
                          (function() {{
                            function scrollToTarget() {{
                              try {{
                                var el = window.parent.document.getElementById("segment-{hi_seg}");
                                if (el) {{
                                  el.scrollIntoView({{ behavior: "smooth", block: "center" }});
                                }}
                              }} catch (e) {{}}
                            }}
                            setTimeout(scrollToTarget, 100);
                            setTimeout(scrollToTarget, 400);
                          }})();
                        </script>
                        """,
                        height=0,
                        width=0,
                    )

                # Manual speaker name mapping expander
                with st.expander("🏷️ Edit Speaker Names (Optional)", expanded=False):
                    st.caption("Map detected speaker IDs to names for transcript display:")
                    for spk in unique_speakers:
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.markdown(f"**{spk}** →")
                        with col2:
                            st.text_input(
                                f"Name for {spk}",
                                key=f"speaker_input_{spk}",
                                placeholder="Enter name",
                                label_visibility="collapsed",
                                on_change=_on_transcript_speaker_change,
                                args=(spk,),
                            )

                avatars = ["🧑‍💼", "👩‍💻", "👨‍🏫", "👩‍🔬", "🕵️", "🧑‍⚕️"]
                speaker_avatars = {}

                for idx, block in enumerate(aligned_transcript, start=1):
                    seg_id = block.get("id", idx)
                    is_highlighted = (hi_seg is not None and seg_id == hi_seg)
                    speaker_id = block["speaker"]
                    display_name = speaker_name_map.get(speaker_id, speaker_id)
                    if speaker_id not in speaker_avatars:
                        speaker_avatars[speaker_id] = avatars[len(speaker_avatars) % len(avatars)]

                    if is_highlighted:
                        st.markdown(
                            f'<div id="segment-{seg_id}" class="em-highlight-segment">'
                            f'⭐ <strong>TARGET EVIDENCE SEGMENT #{seg_id}</strong> • ⏱️ {block["start"]:05.2f}s - {block["end"]:05.2f}s'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(f'<div id="segment-{seg_id}"></div>', unsafe_allow_html=True)

                    with st.chat_message(name=display_name, avatar=speaker_avatars[speaker_id]):
                        highlight_label = "⭐ **HIGHLIGHTED** • " if is_highlighted else ""
                        st.caption(f"{highlight_label}{display_name} • {block['start']:05.2f}s - {block['end']:05.2f}s")
                        st.write(block["text"])

            # ══════════════════════════════════════════════════════════════
            # TAB 5 — PROCESSING PERFORMANCE
            # ══════════════════════════════════════════════════════════════
            with tab_perf:
                st.subheader("Processing Performance")
                st.caption("Granular latency and timing breakdown across speech and intelligence pipeline stages.")

                total_val = timing.get("total", 0.0)
                t_trans   = timing.get("transcription", 0.0)
                t_diar    = timing.get("diarization", 0.0)
                t_align   = timing.get("alignment", 0.0)
                t_prep    = timing.get("preprocessing", 0.0)
                t_bart    = timing.get("bart_summary", 0.0)
                t_qwen    = timing.get("qwen_extraction", 0.0)
                t_ev      = timing.get("evidence", 0.0)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Duration", format_duration(total_val))
                with col2:
                    speech_dur = t_trans + t_diar + t_align
                    st.metric("Speech Pipeline", format_duration(speech_dur))
                with col3:
                    st.metric("BART Summary", format_duration(t_bart))
                with col4:
                    intel_dur = t_prep + t_bart + t_qwen + t_ev
                    st.metric("Intelligence Stage", format_duration(intel_dur))

                st.markdown("#### Stage Timing Breakdown")
                perf_rows = [
                    {"Stage": "1. Transcription", "Component": "Whisper (Speech-to-Text)", "Duration": format_duration(t_trans)},
                    {"Stage": "2. Speaker Diarization", "Component": "PyAnnote Audio", "Duration": format_duration(t_diar)},
                    {"Stage": "3. Speaker Alignment", "Component": "Member 1 Aligner", "Duration": format_duration(t_align)},
                    {"Stage": "4. Text Preprocessing", "Component": "Regex Normalizer", "Duration": format_duration(t_prep)},
                    {"Stage": "5. Baseline Summary (BART)", "Component": "BART-large-cnn", "Duration": format_duration(t_bart)},
                    {"Stage": "6. Qwen Extraction", "Component": "Qwen3 (Local Ollama)", "Duration": format_duration(t_qwen)},
                    {"Stage": "7. Evidence Retrieval & Verification", "Component": "Deterministic Verifier", "Duration": format_duration(t_ev)},
                    {"Stage": "Total Pipeline Execution", "Component": "Full End-to-End", "Duration": format_duration(total_val)},
                ]
                st.table(perf_rows)

                dt = data.get("diar_timings")
                if dt:
                    st.markdown("##### ⏱️ Speaker Diarization Breakdown")
                    st.caption(
                        f"• **Model Initialization:** {format_duration(dt.get('model_load', 0.0))} &nbsp;|&nbsp; "
                        f"• **Audio Prep (WAV Conversion):** {format_duration(dt.get('conversion', 0.0))} &nbsp;|&nbsp; "
                        f"• **Inference:** {format_duration(dt.get('inference', 0.0))}"
                    )

    else:
        # ── EMPTY / INITIAL STATE ──────────────────────────────────────────
        st.info("Upload an audio file to start the transcription and summarization pipeline.")


if __name__ == "__main__":
    main()