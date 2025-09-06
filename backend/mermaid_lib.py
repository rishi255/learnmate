import base64
import requests

# Local imports
# from .utils import load_config
# from backend.paths import CONFIG_FILE_PATH

# app_cfg: dict = load_config(CONFIG_FILE_PATH)


def call_mermaid_api(mermaid_code: str) -> tuple[requests.Response | None, str | None]:
    """
    Call the Mermaid rendering API to generate an SVG from Mermaid code.

    Args:
            mermaid_code: Mermaid diagram code (e.g., 'graph TD; A-->B;').
            base_url: Optional base URL for the Mermaid rendering service.

    Returns:
            (response, error):
                    - response: The requests.Response object if successful, otherwise None.
                    - error: Error message string if failed, otherwise None.
    """
    base_url = "http://localhost:3000"  # (app_cfg["mermaid_api_base_url"]).rstrip("/")
    resp = None
    try:
        # Encode Mermaid code as URL-safe Base64 per mermaid.ink API requirements
        graph_bytes = mermaid_code.encode("utf-8")
        base64_string = base64.urlsafe_b64encode(graph_bytes).decode("ascii")

        # Request SVG from configured base URL (fallback to public mermaid.ink)

        resp = requests.get(f"{base_url}/svg/{base64_string}", timeout=20)
        resp.raise_for_status()
        return resp, None
    except Exception as e:
        return resp, f"Error generating Mermaid SVG: {e}"


def render_mermaid_svg_bytes(mermaid_code: str) -> tuple[bytes | None, str | None]:
    """
    Render Mermaid code to SVG via the public mermaid.ink API.

    Args:
        mermaid_code: Mermaid diagram code (e.g., 'graph TD; A-->B;').

    Returns:
        (svg_bytes, error):
            - svg_bytes: The raw SVG bytes if successful, otherwise None.
            - error: Error message string if failed, otherwise None.
    """
    # Encode Mermaid code as URL-safe Base64 per mermaid.ink API requirements
    resp, error = call_mermaid_api(mermaid_code)
    if error:
        return None, error
    return resp.content, None


def svg_bytes_to_data_uri(svg_bytes: bytes) -> str:
    """
    Convert SVG bytes into a data URI suitable for embedding in HTML/Markdown.

    Note: Many renderers support data URIs. In Streamlit, you may need to pass
    unsafe_allow_html=True to st.markdown when embedding via <img src="...">.

    Args:
        svg_bytes: Raw SVG bytes.

    Returns:
        A data URI string like 'data:image/svg+xml;base64,...'
    """
    b64 = base64.b64encode(svg_bytes).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"
