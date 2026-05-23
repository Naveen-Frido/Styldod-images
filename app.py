"""
Styldod AI Image Pipeline
=========================
A Streamlit application for AI-powered real estate image processing.

Pipeline:
1. Upload Reference Image → Analyze with Gemini → Generate Prompt
2. Generate 5 Images with Gemini
3. Select Image(s) → Process Occupied to Vacant
4. Optional: Upload Reference Furniture → Virtual Staging
5. Rename & Download Results
"""

import io
import base64
import streamlit as st
from PIL import Image

from config import load_config, save_config, get_api_keys, get_msp, DEFAULT_MSP
from ai_services import (
    analyze_image_with_gemini,
    generate_images_with_gemini,
    process_occupied_to_vacant,
    stage_with_furniture,
)

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Styldod AI Image Pipeline",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS for Premium Look
# ──────────────────────────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
                    radial-gradient(circle at 70% 80%, rgba(168, 85, 247, 0.1) 0%, transparent 50%);
        animation: shimmer 8s ease-in-out infinite alternate;
    }
    @keyframes shimmer {
        0% { transform: translateX(-5%) translateY(-5%); }
        100% { transform: translateX(5%) translateY(5%); }
    }
    .main-header h1 {
        color: #ffffff;
        font-weight: 800;
        font-size: 2rem;
        margin: 0;
        position: relative;
        z-index: 1;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.7);
        font-weight: 400;
        font-size: 1rem;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }

    /* Step Cards */
    .step-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 14px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    .step-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.1);
    }
    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-radius: 10px;
        color: white;
        font-weight: 700;
        font-size: 0.95rem;
        margin-right: 12px;
    }
    .step-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #e2e8f0;
        display: inline;
        vertical-align: middle;
    }
    .step-desc {
        color: rgba(226, 232, 240, 0.6);
        font-size: 0.875rem;
        margin-top: 0.5rem;
        margin-left: 48px;
    }

    /* Active step highlight */
    .step-active {
        border-color: rgba(99, 102, 241, 0.6) !important;
        background: linear-gradient(145deg, #1e1e3f, #1a2744) !important;
        box-shadow: 0 0 25px rgba(99, 102, 241, 0.15) !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #c7d2fe;
    }

    /* Image containers */
    .image-container {
        border: 2px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        overflow: hidden;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .image-container:hover {
        border-color: #6366f1;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.2);
    }
    .image-selected {
        border-color: #22c55e !important;
        box-shadow: 0 0 20px rgba(34, 197, 94, 0.3) !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }

    /* Download button */
    .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 600;
        background: linear-gradient(135deg, #22c55e, #16a34a) !important;
        color: white !important;
        border: none !important;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #16a34a, #15803d) !important;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.3);
    }

    /* Prompt display */
    .prompt-box {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 1.2rem;
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.6;
        max-height: 300px;
        overflow-y: auto;
    }

    /* Progress indicator */
    .pipeline-progress {
        display: flex;
        gap: 8px;
        margin-bottom: 1.5rem;
    }
    .progress-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #334155;
        transition: all 0.3s ease;
    }
    .progress-dot.completed {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        box-shadow: 0 0 8px rgba(34, 197, 94, 0.4);
    }
    .progress-dot.active {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        box-shadow: 0 0 8px rgba(99, 102, 241, 0.4);
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.3); }
    }

    /* Success/Info badges */
    .badge-success {
        display: inline-block;
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .badge-info {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    /* Hide Streamlit branding */
    footer { visibility: hidden; }

    /* Text area styling */
    .stTextArea textarea {
        border-radius: 10px;
        border-color: rgba(99, 102, 241, 0.3);
        font-family: 'Inter', monospace;
        font-size: 0.875rem;
    }
    .stTextArea textarea:focus {
        border-color: #6366f1;
        box-shadow: 0 0 10px rgba(99, 102, 241, 0.2);
    }
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────
def image_to_bytes(image: Image.Image, format: str = "JPEG") -> bytes:
    """Convert PIL Image to bytes."""
    buf = io.BytesIO()
    if image.mode == "RGBA" and format == "JPEG":
        image = image.convert("RGB")
    image.save(buf, format=format, quality=95)
    return buf.getvalue()


def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "current_step": 1,
        "config": load_config(),
        # Step 1
        "reference_image": None,
        "reference_image_bytes": None,
        "generated_prompt": None,
        # Step 2
        "generated_images": [],
        # Step 3
        "selected_images": [],
        "vacant_images": [],
        "selected_vacant": None,
        "base_image": None,
        "base_image_bytes": None,
        # Step 4
        "furniture_image": None,
        "furniture_image_bytes": None,
        "staged_images": [],
        "selected_staged": None,
        # Step 5
        "final_image": None,
        "final_image_name": "final_result",
        "base_image_name": "base_image",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


def render_progress_bar():
    """Render the pipeline progress indicator."""
    steps = [
        "Upload & Analyze",
        "Generate Images",
        "Occupied → Vacant",
        "Virtual Staging",
        "Download",
    ]
    current = st.session_state.current_step

    cols = st.columns(len(steps))
    for i, (col, step_name) in enumerate(zip(cols, steps), 1):
        with col:
            if i < current:
                st.markdown(
                    f'<div style="text-align:center">'
                    f'<div class="progress-dot completed" style="margin:0 auto"></div>'
                    f'<div style="color:#4ade80;font-size:0.7rem;margin-top:4px;font-weight:600">{step_name}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            elif i == current:
                st.markdown(
                    f'<div style="text-align:center">'
                    f'<div class="progress-dot active" style="margin:0 auto"></div>'
                    f'<div style="color:#a5b4fc;font-size:0.7rem;margin-top:4px;font-weight:600">{step_name}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="text-align:center">'
                    f'<div class="progress-dot" style="margin:0 auto"></div>'
                    f'<div style="color:#64748b;font-size:0.7rem;margin-top:4px;font-weight:600">{step_name}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ──────────────────────────────────────────────
# Sidebar: API Keys & MSP Management
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # ── API Keys (from Streamlit Secrets) ──
    with st.expander("🔑 API Keys", expanded=False):
        api_keys = get_api_keys(st.session_state.config)

        gemini_status = "✅ Configured" if api_keys.get("gemini") else "❌ Not set"

        st.markdown(f"**Gemini:** {gemini_status}")
        st.caption("Edit `.streamlit/secrets.toml` to update API keys:")
        st.code(
            '[api_keys]\ngemini = "your-gemini-key-here"',
            language="toml",
        )
        if not api_keys.get("gemini"):
            st.warning("⚠️ Add your Gemini API key to `.streamlit/secrets.toml` and restart the app.")

    # ── Master System Prompts ──
    st.markdown("---")
    st.markdown("## 📝 Master System Prompts")

    msp = get_msp(st.session_state.config)

    msp_tabs = st.tabs(["🔍 Analysis", "🏠 Vacant", "🛋️ Staging"])

    with msp_tabs[0]:
        st.caption("Prompt used when analyzing reference images with Gemini")
        msp_analysis = st.text_area(
            "Image Analysis MSP",
            value=msp.get("image_analysis", DEFAULT_MSP["image_analysis"]),
            height=200,
            key="msp_analysis_input",
            label_visibility="collapsed",
        )

    with msp_tabs[1]:
        st.caption("Prompt used for occupied-to-vacant conversion")
        msp_vacant = st.text_area(
            "Occupied to Vacant MSP",
            value=msp.get("occupied_to_vacant", DEFAULT_MSP["occupied_to_vacant"]),
            height=200,
            key="msp_vacant_input",
            label_visibility="collapsed",
        )

    with msp_tabs[2]:
        st.caption("Prompt used for virtual staging with furniture")
        msp_staging = st.text_area(
            "Virtual Staging MSP",
            value=msp.get("virtual_staging", DEFAULT_MSP["virtual_staging"]),
            height=200,
            key="msp_staging_input",
            label_visibility="collapsed",
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save MSPs", key="save_msp_btn", use_container_width=True):
            config = st.session_state.config
            config["msp"] = {
                "image_analysis": msp_analysis,
                "occupied_to_vacant": msp_vacant,
                "virtual_staging": msp_staging,
            }
            st.session_state.config = config
            save_config(config)
            st.success("✅ Saved!")

    with col2:
        if st.button("🔄 Reset MSPs", key="reset_msp_btn", use_container_width=True):
            config = st.session_state.config
            config["msp"] = DEFAULT_MSP.copy()
            st.session_state.config = config
            save_config(config)
            st.rerun()

    # ── Navigation ──
    st.markdown("---")
    st.markdown("## 🧭 Navigation")

    if st.button("🔄 Reset Pipeline", key="reset_pipeline_btn", use_container_width=True, type="secondary"):
        for key in list(st.session_state.keys()):
            if key not in ("config",):
                del st.session_state[key]
        st.session_state.current_step = 1
        st.rerun()


# ──────────────────────────────────────────────
# Main Content Area
# ──────────────────────────────────────────────
# render_progress_bar() # Moved inside steps or kept above as needed


render_progress_bar()

config = st.session_state.config
api_keys = get_api_keys(config)
msp = get_msp(config)




# ══════════════════════════════════════════════
# STEP 1: Upload Reference Image & Analyze
# ══════════════════════════════════════════════
if st.session_state.current_step == 1:
    st.markdown(
        """
    <div class="step-card step-active">
        <span class="step-number">1</span>
        <span class="step-title">Upload Reference Image & Generate Prompt</span>
        <div class="step-desc">Upload a reference room image. It will be analyzed by Gemini to generate a detailed recreation prompt.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col_upload, col_preview = st.columns([1, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "📸 Upload Reference Image",
            type=["jpg", "jpeg", "png", "webp"],
            key="ref_image_uploader",
            help="Upload the room image you want to recreate",
        )

        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(image_bytes))
            st.session_state.reference_image = image
            st.session_state.reference_image_bytes = image_bytes
            
            # AUTOMATIC TRIGGER: Analyze immediately if not already done
            if st.session_state.generated_prompt is None:
                if not api_keys.get("gemini"):
                    st.warning("⚠️ Please add your Gemini API key in the sidebar settings.")
                else:
                    with st.spinner("🤖 Automatically analyzing image with Gemini..."):
                        try:
                            prompt = analyze_image_with_gemini(
                                api_key=api_keys["gemini"],
                                image_bytes=st.session_state.reference_image_bytes,
                                system_prompt=msp.get("image_analysis", DEFAULT_MSP["image_analysis"]),
                            )
                            st.session_state.generated_prompt = prompt
                            # AUTOMATIC PROCEED: Go to step 2
                            st.session_state.current_step = 2
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error analyzing image: {e}")

    with col_preview:
        if st.session_state.reference_image is not None:
            st.image(
                st.session_state.reference_image,
                caption="📷 Reference Image",
                use_container_width=True,
            )

    st.markdown("---")

    # Generate prompt button
    if st.session_state.reference_image is not None:
        if not api_keys.get("gemini"):
            st.warning("⚠️ Please add your Gemini API key in the sidebar settings.")
        else:
            gen_col1, gen_col2 = st.columns([1, 3])
            with gen_col1:
                analyze_btn = st.button(
                    "🔍 Analyze Image & Generate Prompt",
                    key="analyze_btn",
                    type="primary",
                    use_container_width=True,
                )

            if analyze_btn:
                with st.spinner("🤖 Analyzing image with Gemini..."):
                    try:
                        prompt = analyze_image_with_gemini(
                            api_key=api_keys["gemini"],
                            image_bytes=st.session_state.reference_image_bytes,
                            system_prompt=msp.get("image_analysis", DEFAULT_MSP["image_analysis"]),
                        )
                        st.session_state.generated_prompt = prompt
                        st.success("✅ Prompt generated successfully!")
                    except Exception as e:
                        st.error(f"❌ Error analyzing image: {e}")

    # Show generated prompt
    if st.session_state.generated_prompt:
        st.markdown("### 📝 Generated Prompt")
        edited_prompt = st.text_area(
            "Edit the generated prompt if needed:",
            value=st.session_state.generated_prompt,
            height=250,
            key="prompt_editor",
        )
        st.session_state.generated_prompt = edited_prompt

        st.markdown("")
        if st.button(
            "✅ Proceed to Image Generation →",
            key="proceed_step2",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.current_step = 2
            st.rerun()


# ══════════════════════════════════════════════
# STEP 2: Generate 5 Images with Gemini
# ══════════════════════════════════════════════
elif st.session_state.current_step == 2:
    st.markdown(
        """
    <div class="step-card step-active">
        <span class="step-number">2</span>
        <span class="step-title">Generate Images with Gemini</span>
        <div class="step-desc">Generate 5 image variations using the prompt. Select 1 or 2 images to proceed.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Show the reference image and prompt
    col_ref, col_prompt = st.columns([1, 2])
    with col_ref:
        if st.session_state.reference_image:
            st.image(st.session_state.reference_image, caption="📸 Original Reference", use_container_width=True)
    
    with col_prompt:
        st.markdown("**📝 Current Prompt**")
        st.markdown(
            f'<div class="prompt-box">{st.session_state.generated_prompt}</div>',
            unsafe_allow_html=True,
        )

    # AUTOMATIC TRIGGER: Generate 5 images if not already done
    if not st.session_state.generated_images:
        if not api_keys.get("gemini"):
            st.warning("⚠️ Please add your Gemini API key in the sidebar settings.")
        else:
            with st.container():
                st.info("🎨 Automatically generating 5 variations using Nano Banana Pro...")
                progress = st.progress(0, text="Starting generation...")
                
                def update_progress(current, total, text):
                    progress.progress(int((current / total) * 100), text=text)

                try:
                    images = generate_images_with_gemini(
                        api_key=api_keys["gemini"],
                        prompt=st.session_state.generated_prompt,
                        num_images=5,
                        progress_callback=update_progress,
                    )
                    progress.progress(100, text="Complete!")
                    if images:
                        st.session_state.generated_images = images
                        st.success(f"✅ Generated {len(images)} image(s)!")
                        st.rerun()
                    else:
                        st.error("❌ No images were generated. Check your Gemini API key and safety settings.")
                except Exception as e:
                    st.error(f"❌ Error generating images: {e}")

    # Automatic generation logic moved above

    # Display generated images for selection
    if st.session_state.generated_images:
        st.markdown("### 🖼️ Generated Images — Select 1 or 2")
        st.caption("Click the checkboxes below each image to select")

        # Initialize selection
        if "selected_indices" not in st.session_state:
            st.session_state.selected_indices = []

        cols = st.columns(min(5, len(st.session_state.generated_images)))

        for idx, (col, img) in enumerate(
            zip(cols, st.session_state.generated_images)
        ):
            with col:
                st.image(img, caption=f"Image {idx + 1}", use_container_width=True)
                selected = st.checkbox(
                    f"Select #{idx + 1}",
                    key=f"select_img_{idx}",
                    value=idx in st.session_state.get("selected_indices", []),
                )
                if selected and idx not in st.session_state.selected_indices:
                    if len(st.session_state.selected_indices) < 2:
                        st.session_state.selected_indices.append(idx)
                elif not selected and idx in st.session_state.selected_indices:
                    st.session_state.selected_indices.remove(idx)

        # Show selection count
        num_selected = len(st.session_state.get("selected_indices", []))
        if num_selected > 0:
            st.markdown(
                f'<span class="badge-success">✅ {num_selected} image(s) selected</span>',
                unsafe_allow_html=True,
            )

        st.markdown("")

        col_back, col_regen, col_proceed = st.columns([1, 1, 2])
        with col_back:
            if st.button("← Back", key="back_to_step1"):
                st.session_state.current_step = 1
                st.session_state.generated_images = []
                st.session_state.selected_indices = []
                st.rerun()

        with col_regen:
            if st.button("🔄 Regenerate", key="regenerate_btn"):
                st.session_state.generated_images = []
                st.session_state.selected_indices = []
                st.rerun()

        with col_proceed:
            if num_selected > 0:
                if st.button(
                    "✅ Proceed to Occupied → Vacant →",
                    key="proceed_step3",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state.selected_images = [
                        st.session_state.generated_images[i]
                        for i in st.session_state.selected_indices
                    ]
                    st.session_state.current_step = 3
                    st.rerun()


# ══════════════════════════════════════════════
# STEP 3: Occupied to Vacant
# ══════════════════════════════════════════════
elif st.session_state.current_step == 3:
    st.markdown(
        """
    <div class="step-card step-active">
        <span class="step-number">3</span>
        <span class="step-title">Occupied → Vacant Conversion</span>
        <div class="step-desc">The selected image(s) will be processed to remove all furniture and create empty room versions.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Display selected images
    st.markdown("### 📸 Selected Images")
    sel_cols = st.columns(len(st.session_state.selected_images))
    for idx, (col, img) in enumerate(zip(sel_cols, st.session_state.selected_images)):
        with col:
            st.image(img, caption=f"Selected Image {idx + 1}", use_container_width=True)

    st.markdown("---")

    # Process each selected image
    if not st.session_state.vacant_images:
        if not api_keys.get("gemini"):
            st.warning("⚠️ Please add your Gemini API key in the sidebar settings.")
        else:
            if st.button(
                "🏠 Process: Make Room Vacant",
                key="vacant_btn",
                type="primary",
            ):
                all_vacant = []
                for idx, sel_img in enumerate(st.session_state.selected_images):
                    with st.spinner(
                        f"🏠 Processing image {idx + 1}/{len(st.session_state.selected_images)} — Making room vacant..."
                    ):
                        try:
                            img_bytes = image_to_bytes(sel_img)
                            vacant_imgs = process_occupied_to_vacant(
                                api_key=api_keys["gemini"],
                                image_bytes=img_bytes,
                                system_prompt=msp.get(
                                    "occupied_to_vacant",
                                    DEFAULT_MSP["occupied_to_vacant"],
                                ),
                            )
                            all_vacant.extend(vacant_imgs)
                        except Exception as e:
                            st.error(f"❌ Error processing image {idx + 1}: {e}")

                if all_vacant:
                    st.session_state.vacant_images = all_vacant
                    st.success(f"✅ Generated {len(all_vacant)} vacant room image(s)!")
                    st.rerun()

    # Display vacant images for selection
    if st.session_state.vacant_images:
        st.markdown("### 🏠 Vacant Room Results — Select One as Base Image")

        if "selected_vacant_idx" not in st.session_state:
            st.session_state.selected_vacant_idx = None

        v_cols = st.columns(min(5, len(st.session_state.vacant_images)))
        for idx, (col, img) in enumerate(zip(v_cols, st.session_state.vacant_images)):
            with col:
                st.image(img, caption=f"Vacant {idx + 1}", use_container_width=True)
                if st.button(
                    f"✅ Select #{idx + 1}",
                    key=f"select_vacant_{idx}",
                    use_container_width=True,
                    type="primary" if st.session_state.selected_vacant_idx == idx else "secondary",
                ):
                    st.session_state.selected_vacant_idx = idx
                    st.session_state.selected_vacant = img
                    st.session_state.base_image = img
                    st.session_state.base_image_bytes = image_to_bytes(img)
                    st.rerun()

        if st.session_state.selected_vacant is not None:
            st.markdown("---")
            st.markdown("### ✅ Selected Base Image")
            st.image(
                st.session_state.selected_vacant,
                caption="This is now your base (vacant) image",
                use_container_width=True,
            )

            st.markdown("")
            col_back3, col_proceed3 = st.columns([1, 3])
            with col_back3:
                if st.button("← Back", key="back_to_step2"):
                    st.session_state.current_step = 2
                    st.session_state.vacant_images = []
                    st.session_state.selected_vacant = None
                    st.session_state.selected_vacant_idx = None
                    st.rerun()

            with col_proceed3:
                if st.button(
                    "✅ Proceed to Virtual Staging →",
                    key="proceed_step4",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state.current_step = 4
                    st.rerun()


# ══════════════════════════════════════════════
# STEP 4: Virtual Staging (Optional Furniture Upload)
# ══════════════════════════════════════════════
elif st.session_state.current_step == 4:
    st.markdown(
        """
    <div class="step-card step-active">
        <span class="step-number">4</span>
        <span class="step-title">Virtual Staging (Optional)</span>
        <div class="step-desc">Upload reference furniture to stage the vacant room. Or skip to download the base image directly.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Show base image
    col_base, col_furniture = st.columns([1, 1])

    with col_base:
        st.markdown("#### 🏠 Base (Vacant) Image")
        if st.session_state.base_image is not None:
            st.image(
                st.session_state.base_image,
                caption="Base Image (Vacant Room)",
                use_container_width=True,
            )

    with col_furniture:
        st.markdown("#### 🛋️ Upload Reference Furniture")
        furniture_file = st.file_uploader(
            "Upload furniture reference image",
            type=["jpg", "jpeg", "png", "webp"],
            key="furniture_uploader",
            help="Upload an image of the furniture style you want",
            label_visibility="collapsed",
        )

        if furniture_file is not None:
            furniture_bytes = furniture_file.getvalue()
            furniture_img = Image.open(io.BytesIO(furniture_bytes))
            st.session_state.furniture_image = furniture_img
            st.session_state.furniture_image_bytes = furniture_bytes
            st.image(furniture_img, caption="Reference Furniture", use_container_width=True)

    st.markdown("---")

    # Virtual staging
    if st.session_state.furniture_image is not None and not st.session_state.staged_images:
        if not api_keys.get("gemini"):
            st.warning("⚠️ Please add your Gemini API key in the sidebar settings.")
        else:
            if st.button(
                "🛋️ Generate Staged Images",
                key="stage_btn",
                type="primary",
            ):
                with st.spinner("🛋️ Staging room with reference furniture... This may take a minute."):
                    try:
                        staged = stage_with_furniture(
                            api_key=api_keys["gemini"],
                            base_image_bytes=st.session_state.base_image_bytes,
                            furniture_image_bytes=st.session_state.furniture_image_bytes,
                            system_prompt=msp.get(
                                "virtual_staging",
                                DEFAULT_MSP["virtual_staging"],
                            ),
                        )
                        if staged:
                            st.session_state.staged_images = staged
                            st.success(f"✅ Generated {len(staged)} staged image(s)!")
                            st.rerun()
                        else:
                            st.error("❌ No staged images generated. Please try again.")
                    except Exception as e:
                        st.error(f"❌ Error staging images: {e}")

    # Display staged images
    if st.session_state.staged_images:
        st.markdown("### 🛋️ Staged Room Results — Select One")

        if "selected_staged_idx" not in st.session_state:
            st.session_state.selected_staged_idx = None

        s_cols = st.columns(min(5, len(st.session_state.staged_images)))
        for idx, (col, img) in enumerate(zip(s_cols, st.session_state.staged_images)):
            with col:
                st.image(img, caption=f"Staged {idx + 1}", use_container_width=True)
                if st.button(
                    f"✅ Select #{idx + 1}",
                    key=f"select_staged_{idx}",
                    use_container_width=True,
                    type="primary" if st.session_state.selected_staged_idx == idx else "secondary",
                ):
                    st.session_state.selected_staged_idx = idx
                    st.session_state.selected_staged = img
                    st.session_state.final_image = img
                    st.rerun()

        if st.session_state.selected_staged is not None:
            st.markdown("---")
            st.markdown("### ✅ Selected Final Image")
            st.image(
                st.session_state.selected_staged,
                caption="Final Staged Image",
                use_container_width=True,
            )

    st.markdown("---")

    col_back4, col_skip, col_proceed4 = st.columns([1, 1, 2])

    with col_back4:
        if st.button("← Back", key="back_to_step3"):
            st.session_state.current_step = 3
            st.session_state.staged_images = []
            st.session_state.selected_staged = None
            st.session_state.selected_staged_idx = None
            st.session_state.furniture_image = None
            st.session_state.furniture_image_bytes = None
            st.rerun()

    with col_skip:
        if st.button("⏭️ Skip Staging", key="skip_staging"):
            # If no staging done, the final image is the base image
            st.session_state.final_image = st.session_state.base_image
            st.session_state.current_step = 5
            st.rerun()

    with col_proceed4:
        can_proceed = (
            st.session_state.selected_staged is not None
            or st.session_state.furniture_image is None
        )
        if st.session_state.selected_staged is not None:
            if st.button(
                "✅ Proceed to Download →",
                key="proceed_step5",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.current_step = 5
                st.rerun()


# ══════════════════════════════════════════════
# STEP 5: Rename & Download
# ══════════════════════════════════════════════
elif st.session_state.current_step == 5:
    st.markdown(
        """
    <div class="step-card step-active">
        <span class="step-number">5</span>
        <span class="step-title">Rename & Download Results</span>
        <div class="step-desc">Rename your images and download both the base image and final result as separate JPG files.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Side by side comparison
    col_base_dl, col_final_dl = st.columns(2)

    with col_base_dl:
        st.markdown("#### 🏠 Base Image (Vacant)")
        if st.session_state.base_image is not None:
            st.image(
                st.session_state.base_image,
                caption="Base Image",
                use_container_width=True,
            )
            base_name = st.text_input(
                "📝 Base Image Filename",
                value=st.session_state.base_image_name,
                key="base_name_input",
                help="Enter filename without extension",
            )
            st.session_state.base_image_name = base_name

            base_bytes = image_to_bytes(st.session_state.base_image)
            st.download_button(
                label=f"📥 Download {base_name}.jpg",
                data=base_bytes,
                file_name=f"{base_name}.jpg",
                mime="image/jpeg",
                key="download_base",
                use_container_width=True,
            )

    with col_final_dl:
        st.markdown("#### 🎨 Final Result")
        if st.session_state.final_image is not None:
            st.image(
                st.session_state.final_image,
                caption="Final Result",
                use_container_width=True,
            )
            final_name = st.text_input(
                "📝 Final Image Filename",
                value=st.session_state.final_image_name,
                key="final_name_input",
                help="Enter filename without extension",
            )
            st.session_state.final_image_name = final_name

            final_bytes = image_to_bytes(st.session_state.final_image)
            st.download_button(
                label=f"📥 Download {final_name}.jpg",
                data=final_bytes,
                file_name=f"{final_name}.jpg",
                mime="image/jpeg",
                key="download_final",
                use_container_width=True,
            )
        else:
            st.info("ℹ️ No staged image was generated. The base image is your final result.")
            if st.session_state.base_image is not None:
                st.image(
                    st.session_state.base_image,
                    caption="Final Result (Same as Base)",
                    use_container_width=True,
                )
                final_name = st.text_input(
                    "📝 Final Image Filename",
                    value=st.session_state.final_image_name,
                    key="final_name_input_alt",
                    help="Enter filename without extension",
                )
                final_bytes = image_to_bytes(st.session_state.base_image)
                st.download_button(
                    label=f"📥 Download {final_name}.jpg",
                    data=final_bytes,
                    file_name=f"{final_name}.jpg",
                    mime="image/jpeg",
                    key="download_final_alt",
                    use_container_width=True,
                )

    st.markdown("---")

    # Summary
    st.markdown(
        """
    <div class="step-card">
        <span class="step-title">📊 Pipeline Summary</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    summary_cols = st.columns(4)
    with summary_cols[0]:
        st.metric("Reference Image", "✅ Uploaded" if st.session_state.reference_image else "—")
    with summary_cols[1]:
        st.metric("Images Generated", len(st.session_state.generated_images))
    with summary_cols[2]:
        st.metric("Vacant Images", len(st.session_state.vacant_images))
    with summary_cols[3]:
        st.metric(
            "Staged Images",
            len(st.session_state.staged_images) if st.session_state.staged_images else "Skipped",
        )

    st.markdown("")
    if st.button("🔄 Start New Pipeline", key="restart_pipeline", type="primary", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key not in ("config",):
                del st.session_state[key]
        st.session_state.current_step = 1
        st.rerun()
