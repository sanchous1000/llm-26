"""Reference list: one list item per bibliography entry."""

from bs4 import BeautifulSoup, Tag

from source.parsing.latexml.dom import alive, collapse, new_tag, new_text, text_of


def convert_bibliography(root: Tag, include: bool = True) -> None:
    """Render ``ltx_bibliography`` sections as plain lists (or drop them)."""
    for section in alive(root.select("section.ltx_bibliography")):
        if not include:
            section.decompose()
            continue
        for item in alive(section.select("li.ltx_bibitem")):
            item.replace_with(_entry(item))
        for container in section.select("ul.ltx_biblist"):
            container.attrs = {}


def _entry(item: Tag) -> Tag:
    entry = new_tag("li")
    label = item.find("span", class_="ltx_tag_bibitem")
    if label is not None:
        strong = new_tag("strong")
        strong.append(new_text(text_of(label)))
        entry.append(strong)
        entry.append(new_text(" "))
    blocks = item.select("span.ltx_bibblock")
    for block in blocks:
        for anchor in block.find_all("a"):
            if not (anchor.get("href") or "").startswith("http"):
                anchor.unwrap()
        inner = collapse(block.decode_contents())
        if inner:
            fragment = new_tag("span")
            fragment.append(_parse_fragment(inner))
            entry.append(fragment)
            entry.append(new_text(" "))
    if not blocks:
        entry.append(new_text(collapse(item.get_text(" "))))
    return entry


def _parse_fragment(html: str) -> Tag:
    return BeautifulSoup(f"<span>{html}</span>", "html.parser").span
