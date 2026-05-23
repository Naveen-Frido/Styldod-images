"""
AI Service integrations using Google Gemini for both image analysis and image generation/editing.
Uses the google-genai SDK.
"""
import base64
import io
import time
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types


import concurrent.futures

# Model for Gemini image generation (Nano Banana Pro - high quality)
GEMINI_IMAGE_MODEL = "gemini-3-pro-image-preview"


def encode_image_to_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


GEMINI_TEXT_MODEL = "gemini-2.5-flash"


def analyze_image_with_gemini(api_key: str, image_bytes: bytes, system_prompt: str) -> str:
    """
    Send image to Gemini to analyze and generate a recreation prompt.
    """
    if not api_key:
        api_key = st.secrets.get("api_keys", {}).get("gemini", "")
    
    if not api_key:
        raise ValueError("Gemini API key not found in secrets or passed as argument.")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=GEMINI_TEXT_MODEL,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg",
            ),
            "Analyze this image and write a detailed prompt to recreate it with an AI image generator.",
        ],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
        ),
    )
    return response.text


def _extract_image_from_response(response) -> Image.Image | None:
    """Extract PIL Image from a Gemini response, or return None."""
    try:
        # Try the simpler response.parts API first (official SDK pattern)
        if hasattr(response, "parts") and response.parts:
            for part in response.parts:
                if hasattr(part, "inline_data") and part.inline_data is not None:
                    img = Image.open(io.BytesIO(part.inline_data.data))
                    return img
                # Also try as_image() helper
                if hasattr(part, "as_image"):
                    try:
                        return part.as_image()
                    except Exception:
                        pass

        # Fallback: try candidates path
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]

            # Check for safety blocks
            if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                finish = str(candidate.finish_reason)
                if "SAFETY" in finish.upper():
                    return None

            if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "inline_data") and part.inline_data is not None:
                        img = Image.open(io.BytesIO(part.inline_data.data))
                        return img

        # If we got here, check for text response
        if hasattr(response, "text") and response.text:
            # We don't want to spam info here if it's parallel
            pass

        return None

    except Exception:
        return None


def generate_single_image(client, model, prompt):
    """Helper for parallel execution."""
    try:
        response = client.models.generate_content(
            model=model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )
        img = _extract_image_from_response(response)
        if img is None:
            error_details = "No image found in response."
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                    error_details += f" Finish reason: {candidate.finish_reason}."
            if hasattr(response, "text") and response.text:
                error_details += f" Text response: {response.text[:200]}"
            raise ValueError(error_details)
        return img
    except Exception as e:
        raise e


def generate_images_with_gemini(
    api_key: str, prompt: str, num_images: int = 5, progress_callback=None
) -> list:
    """
    Generate images using Gemini's native image generation capability in parallel.
    """
    if not api_key:
        api_key = st.secrets.get("api_keys", {}).get("gemini", "")
    
    if not api_key:
        st.error("❌ Gemini API key not found in secrets.")
        return []

    client = genai.Client(api_key=api_key)
    images = []

    if progress_callback:
        progress_callback(0, num_images, f"Firing {num_images} parallel requests...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_images) as executor:
        future_to_idx = {
            executor.submit(generate_single_image, client, GEMINI_IMAGE_MODEL, prompt): i 
            for i in range(num_images)
        }
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                img = future.result()
                if img:
                    images.append(img)
                if progress_callback:
                    completed = len(images)
                    progress_callback(completed, num_images, f"Received {completed}/{num_images} images...")
            except Exception as e:
                st.error(f"❌ Worker thread failed for image {idx+1}: {e}")

    return images


def process_occupied_to_vacant(
    api_key: str, image_bytes: bytes, system_prompt: str, progress_callback=None
) -> list:
    """
    Send an occupied room image to Gemini and get back vacant versions.
    """
    if not api_key:
        api_key = st.secrets.get("api_keys", {}).get("gemini", "")
    
    if not api_key:
        st.error("❌ Gemini API key not found in secrets.")
        return []

    client = genai.Client(api_key=api_key)
    images = []
    errors = []

    full_prompt = (
        f"{system_prompt}\n\n"
        "Please process the attached image of an occupied room. "
        "Remove all movable furniture, decorations, and personal items. "
        "Keep the room's architecture, walls, floors, windows, doors, and built-in fixtures intact. "
        "Generate a photorealistic image of the same room but completely empty and clean."
    )

    for i in range(5):
        try:
            if progress_callback:
                progress_callback(i, 5, f"Processing vacant version {i+1}/5...")

            response = client.models.generate_content(
                model=GEMINI_IMAGE_MODEL,
                contents=[
                    types.Content(
                        parts=[
                            types.Part.from_text(text=full_prompt),
                            types.Part.from_bytes(
                                data=image_bytes,
                                mime_type="image/jpeg",
                            ),
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                ),
            )

            img = _extract_image_from_response(response)
            if img is not None:
                images.append(img)

            if i < 4:
                time.sleep(3)

        except Exception as e:
            error_msg = f"Vacant {i+1}: {str(e)}"
            errors.append(error_msg)
            st.warning(f"⚠️ Error processing vacant image {i+1}: {e}")
            time.sleep(5)
            continue

    if errors and not images:
        st.error("❌ All vacant processing attempts failed. Errors:\n" + "\n".join(errors))

    return images


def stage_with_furniture(
    api_key: str,
    base_image_bytes: bytes,
    furniture_image_bytes: bytes,
    system_prompt: str,
    progress_callback=None,
) -> list:
    """
    Send base (vacant) image + reference furniture to Gemini.
    """
    if not api_key:
        api_key = st.secrets.get("api_keys", {}).get("gemini", "")
    
    if not api_key:
        st.error("❌ Gemini API key not found in secrets.")
        return []

    client = genai.Client(api_key=api_key)
    images = []
    errors = []

    full_prompt = (
        f"{system_prompt}\n\n"
        "I'm providing two images:\n"
        "1. The first image is an empty room that needs to be furnished.\n"
        "2. The second image shows reference furniture that should be used for staging.\n\n"
        "Please analyze the reference furniture (style, color, material, design) and "
        "place similar or matching furniture appropriately in the empty room. "
        "Maintain proper scale, perspective, lighting, and shadows. "
        "The result should look photorealistic."
    )

    for i in range(5):
        try:
            if progress_callback:
                progress_callback(i, 5, f"Staging image {i+1}/5...")

            response = client.models.generate_content(
                model=GEMINI_IMAGE_MODEL,
                contents=[
                    types.Content(
                        parts=[
                            types.Part.from_text(text=full_prompt),
                            types.Part.from_bytes(
                                data=base_image_bytes,
                                mime_type="image/jpeg",
                            ),
                            types.Part.from_bytes(
                                data=furniture_image_bytes,
                                mime_type="image/jpeg",
                            ),
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                ),
            )

            img = _extract_image_from_response(response)
            if img is not None:
                images.append(img)

            if i < 4:
                time.sleep(3)

        except Exception as e:
            error_msg = f"Staged {i+1}: {str(e)}"
            errors.append(error_msg)
            st.warning(f"⚠️ Error staging image {i+1}: {e}")
            time.sleep(5)
            continue

    if errors and not images:
        st.error("❌ All staging attempts failed. Errors:\n" + "\n".join(errors))

    return images
