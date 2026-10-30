"""Figures, images and captions."""

import re
from urllib.parse import urljoin

from bs4 import Tag

from source.parsing.latexml.dom import alive, collapse, new_tag, new_text, text_of

_GENERIC_ALT = "refer to caption"


def convert_images(root: Tag, base_url: str) -> None:
    """Make image links absolute, drop inline data and decoration attributes."""
    base = base_url.rstrip("/") + "/"
    for image in alive(root.find_all("img")):
        source = (image.get("src") or "").strip()
        if not source or source.startswith("data:"):
            image.decompose()
            continue
        label = _figure_label(image)
        _own_paragraph(image)
        alt = collapse(image.get("alt") or "")
        image.attrs = {
            "src": urljoin(base, source),
            "alt": label or ("" if alt.lower() == _GENERIC_ALT else alt),
        }


def _own_paragraph(image: Tag) -> None:
    """Stand-alone images get their own paragraph (so they do not run together)."""
    if image.find_parent(["td", "th", "p", "h1", "h2", "h3", "h4", "h5", "h6"]):
        return
    paragraph = new_tag("p")
    image.replace_with(paragraph)
    paragraph.append(image)


def _figure_label(image: Tag) -> str:
    """``Figure 3 (a)`` built from the captions of the enclosing figures."""
    labels = []
    for figure in image.find_parents("figure"):
        caption = figure.find("figcaption", recursive=False)
        tag = caption.find(class_="ltx_tag") if caption else None
        if tag is not None:
            labels.append(_tidy_label(text_of(tag)))
    return " ".join(reversed(labels))


def convert_figures(root: Tag) -> None:
    """Figures/tables/algorithms become a div with a bold-labelled caption."""
    for figure in reversed(alive(root.find_all("figure"))):
        caption = figure.find("figcaption", recursive=False)
        if caption is not None:
            _caption_to_paragraph(caption)
        figure.name = "div"


def _tidy_label(label: str, keep_colon: bool = False) -> str:
    label = re.sub(r"\s+([:.])", r"\1", label)
    return label if keep_colon else label.rstrip(":. ")


def _caption_to_paragraph(caption: Tag) -> None:
    tag = caption.find(class_="ltx_tag")
    if tag is not None:
        label = _tidy_label(text_of(tag), keep_colon=True)
        if label and label[-1] not in ":.)":
            label += ":"
        strong = new_tag("strong")
        strong.append(new_text(label))
        tag.replace_with(strong)
        strong.insert_after(new_text(" "))
    caption.name = "p"
