"""
Configuration management for API keys and Master System Prompts.
API keys are stored in Streamlit secrets (.streamlit/secrets.toml).
MSPs are persisted to a local JSON file.
"""
import json
import os
import streamlit as st

CONFIG_FILE = "app_config.json"

DEFAULT_MSP = {
    "image_analysis": (
        "You are an expert interior design and real estate photography analyst. "
        "Analyze the uploaded reference image in extreme detail. Describe the room type, "
        "camera angle, lighting conditions, color palette, textures, materials, furniture style, "
        "architectural features, and overall aesthetic. Write a comprehensive prompt that could "
        "be used to recreate this exact image using an AI image generator. "
        "The prompt should be detailed enough to capture every visual element."
    ),
    "occupied_to_vacant": (
        "You are an expert at virtual staging and real estate image editing. "
        "Your task is to take an occupied room image and remove ALL furniture, decorations, "
        "and personal items while preserving the room's architecture, walls, floors, windows, "
        "doors, built-in fixtures, and lighting. The result should look like a clean, empty, "
        "move-in ready room. Maintain the same camera angle, perspective, and lighting. "
        "Keep the walls, flooring, ceiling, and any built-in elements exactly as they are. "
        "The room should look photorealistic and professionally photographed."
    ),
    "virtual_staging": (
        "You are an expert virtual staging specialist. Your task is to furnish an empty room "
        "using the reference furniture provided. Analyze the reference furniture images for "
        "style, color, material, and design. Place appropriate furniture in the empty room "
        "maintaining proper scale, perspective, and lighting consistency. "
        "The result should look photorealistic, as if the furniture was actually in the room. "
        "Ensure proper shadows, reflections, and lighting interaction with the placed furniture."
    ),
}


def get_api_keys_from_secrets():
    """Get API keys from Streamlit secrets (.streamlit/secrets.toml)."""
    try:
        return {
            "openai": st.secrets.get("api_keys", {}).get("openai", ""),
            "gemini": st.secrets.get("api_keys", {}).get("gemini", ""),
        }
    except Exception:
        return {"openai": "", "gemini": ""}


def load_config():
    """Load configuration from file, or return defaults. API keys come from secrets."""
    config = {"msp": DEFAULT_MSP.copy()}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                saved = json.load(f)
                config["msp"] = saved.get("msp", DEFAULT_MSP.copy())
        except (json.JSONDecodeError, IOError):
            pass
    # Always read API keys from Streamlit secrets
    config["api_keys"] = get_api_keys_from_secrets()
    return config


def save_config(config):
    """Save MSP configuration to file. API keys are NOT saved here."""
    save_data = {"msp": config.get("msp", DEFAULT_MSP.copy())}
    with open(CONFIG_FILE, "w") as f:
        json.dump(save_data, f, indent=2)


def get_api_keys(config):
    """Get API keys - always from Streamlit secrets."""
    return get_api_keys_from_secrets()


def get_msp(config):
    """Get Master System Prompts from config."""
    return config.get("msp", DEFAULT_MSP.copy())
