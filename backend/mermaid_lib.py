import base64
import requests


def render_mermaid_svg_bytes(
    graph: str, base_url: str | None = None
) -> tuple[bytes | None, str | None]:
    """
    Render Mermaid code to SVG via the public mermaid.ink API.

    Args:
        graph: Mermaid diagram code (e.g., 'graph TD; A-->B;').

    Returns:
        (svg_bytes, error):
            - svg_bytes: The raw SVG bytes if successful, otherwise None.
            - error: Error message string if failed, otherwise None.
    """
    try:
        # Encode Mermaid code as URL-safe Base64 per mermaid.ink API requirements
        graph_bytes = graph.encode("utf-8")
        base64_string = base64.urlsafe_b64encode(graph_bytes).decode("ascii")

        # Request SVG from configured base URL (fallback to public mermaid.ink)
        base = (base_url or "https://mermaid.ink").rstrip("/")
        resp = requests.get(f"{base}/svg/{base64_string}", timeout=30)
        resp.raise_for_status()
        return resp.content, None
    except Exception as e:
        return None, f"Error generating Mermaid SVG: {e}"


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
